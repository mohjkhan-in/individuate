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

    def _policy_logprob(self, token_ids: torch.Tensor, prefix_len: int) -> torch.Tensor:
        """Sum log-prob of sampled tokens under the current policy."""
        emb = self.model.lm.get_input_embeddings()(token_ids)
        prefix = None  # already reflected in sampling; recompute for gradient
        out = self.model.lm(inputs_embeds=emb)
        logits = out.logits[:, :-1, :]
        targets = token_ids[:, 1:]
        logp = torch.log_softmax(logits, dim=-1)
        gathered = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
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
        base_logprob = torch.zeros(c.shape[0], device=self.device)  # KL baseline ref
        policy_logprob = self._policy_logprob(token_ids, self.model.cfg.prefix_tokens)

        reward, comps = self.reward(
            text_emb, id_anchor, attr_pred, attr_true, base_logprob, policy_logprob.detach()
        )

        # REINFORCE with moving-average baseline
        self._baseline = (
            self.cfg.baseline_momentum * self._baseline
            + (1 - self.cfg.baseline_momentum) * reward.mean().item()
        )
        advantage = (reward - self._baseline).detach()
        loss = -(advantage * policy_logprob).mean()

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters, self.cfg.grad_clip)
        self.opt.step()

        comps["loss"] = float(loss.item())
        comps["baseline"] = self._baseline
        return comps

    def fit(self, data_iter: Callable[[], Dict[str, torch.Tensor]]):
        history = []
        for it in range(self.cfg.steps):
            comps = self.step(data_iter())
            if it % self.cfg.log_every == 0:
                history.append({"step": it, **comps})
                print(f"[{it:5d}] " + " ".join(f"{k}={v:.3f}" for k, v in comps.items()))
        return history


def train_generator(model, reward, encode_text, data_iter, cfg: TrainConfig, device="cpu"):
    trainer = GeneratorTrainer(model, reward, encode_text, cfg, device)
    return trainer.fit(data_iter)
