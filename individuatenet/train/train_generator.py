"""IndividuateNet training loop (REINFORCE on the individuation reward).

The generator samples captions, they are embedded by the frozen text encoder,
scored by `IndividuationReward` (identity alignment - attribute leakage +
fluency), and the policy is updated by REINFORCE with a moving-average baseline.

This module is cluster-only (imports torch and the generator). See
`scripts/train_generator.py` for the CLI entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import torch

from .losses import IndividuationReward, RewardWeights


@dataclass
class TrainConfig:
    lr: float = 1e-4
    steps: int = 5000
    batch_size: int = 32
    max_new_tokens: int = 40
    grad_clip: float = 1.0
    log_every: int = 100
    baseline_momentum: float = 0.9
    ckpt_path: str = ""
    save_every: int = 500
    resume: bool = True


class GeneratorTrainer:
    def __init__(
        self,
        model,
        reward: IndividuationReward,
        encode_text: Callable[[list[str]], torch.Tensor],
        cfg: TrainConfig,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.reward = reward
        self.encode_text = encode_text
        self.cfg = cfg
        self.device = device
        self.opt = torch.optim.Adam(model.trainable_parameters, lr=cfg.lr)
        self._baseline = 0.0

    def _seq_logprob(self, token_ids: torch.Tensor, prefix=None) -> torch.Tensor:
        """Sum log-prob of tok_1..tok_{L-1} given an optional (differentiable) prefix.

        With ``prefix`` the log-prob depends on the context encoder, so the
        REINFORCE gradient flows into the only trainable module. Without it we
        get the frozen base-LM reference used for the KL / fluency term.
        """
        tok_emb = self.model._embed_tokens(token_ids)
        if prefix is not None:
            inputs_embeds = torch.cat([prefix, tok_emb], dim=1)
            p = prefix.shape[1]
        else:
            inputs_embeds = tok_emb
            p = 0
        out = self.model.lm(inputs_embeds=inputs_embeds)
        logp = torch.log_softmax(out.logits, dim=-1)
        length = token_ids.shape[1]
        # token at absolute position p+j (j>=1) is predicted by logits[p+j-1];
        # slice [p : p+length-1] -> predictions for tok_1..tok_{L-1}.
        pred = logp[:, p:p + length - 1, :]
        targets = token_ids[:, 1:]
        gathered = pred.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return gathered.sum(dim=1)

    def step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """One REINFORCE update. `batch` has context `c`, id_anchor, attr, attr_net."""
        c = batch["context"].to(self.device)
        id_anchor = batch["id_anchor"].to(self.device)
        attr_true = batch["attr"].to(self.device)
        attr_net = batch["attr_net"]  # frozen CLUB net q(a|t)

        token_ids = self.model.generate(c, self.cfg.max_new_tokens)
        captions = self.model.decode(token_ids)
        text_emb = self.encode_text(captions).to(self.device)

        with torch.no_grad():
            attr_pred = attr_net(text_emb)
        prefix = self.model._prefix(c)
        policy_logprob = self._seq_logprob(token_ids, prefix=prefix)
        with torch.no_grad():
            base_logprob = self._seq_logprob(token_ids, prefix=None)  # frozen-LM ref

        # per-token logprobs for the fluency/KL term (scale-stable vs length)
        seq_len = max(int(token_ids.shape[1]) - 1, 1)
        reward, comps = self.reward(
            text_emb, id_anchor, attr_pred, attr_true,
            base_logprob / seq_len, (policy_logprob / seq_len).detach(),
        )

        # REINFORCE with moving-average baseline + per-batch advantage
        # standardisation (keeps the policy-gradient scale stable; without it the
        # unbounded reward made the loss explode into the hundreds).
        self._baseline = (
            self.cfg.baseline_momentum * self._baseline
            + (1 - self.cfg.baseline_momentum) * reward.mean().item()
        )
        advantage = (reward - self._baseline).detach()
        advantage = advantage / (advantage.std() + 1e-6)
        loss = -(advantage * policy_logprob).mean()

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters, self.cfg.grad_clip)
        self.opt.step()

        comps["loss"] = float(loss.item())
        comps["baseline"] = self._baseline
        return comps

    def fit(self, data_iter: Callable[[], Dict[str, torch.Tensor]]):
        import os

        history = []
        start = 0
        if self.cfg.resume and self.cfg.ckpt_path and os.path.exists(self.cfg.ckpt_path):
            start = self._load_ckpt(self.cfg.ckpt_path)
            print(f"[resume] loaded {self.cfg.ckpt_path} at step {start}")
        for it in range(start, self.cfg.steps):
            comps = self.step(data_iter())
            if it % self.cfg.log_every == 0:
                history.append({"step": it, **comps})
                print(f"[{it:5d}] " + " ".join(f"{k}={v:.3f}" for k, v in comps.items()))
            if (
                self.cfg.ckpt_path
                and self.cfg.save_every
                and (it + 1) % self.cfg.save_every == 0
            ):
                self._save_ckpt(self.cfg.ckpt_path, it + 1)
        if self.cfg.ckpt_path:
            self._save_ckpt(self.cfg.ckpt_path, self.cfg.steps)
            open(self.cfg.ckpt_path + ".done", "w").close()
        return history

    def _save_ckpt(self, path: str, step: int) -> None:
        """Atomic checkpoint (context encoder + optimiser + step) to durable FS."""
        import dataclasses
        import os

        tmp = path + ".tmp"
        torch.save(
            {
                "context": self.model.context.state_dict(),
                "gen_cfg": dataclasses.asdict(self.model.cfg),
                "opt": self.opt.state_dict(),
                "baseline": self._baseline,
                "step": step,
            },
            tmp,
        )
        os.replace(tmp, path)

    def _load_ckpt(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.model.context.load_state_dict(ckpt["context"])
        if "opt" in ckpt:
            self.opt.load_state_dict(ckpt["opt"])
        self._baseline = ckpt.get("baseline", 0.0)
        return int(ckpt.get("step", 0))


def train_generator(model, reward, encode_text, data_iter, cfg: TrainConfig, device="cpu"):
    trainer = GeneratorTrainer(model, reward, encode_text, cfg, device)
    return trainer.fit(data_iter)
