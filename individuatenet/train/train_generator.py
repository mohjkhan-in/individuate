"""IndividuateNet training loop (REINFORCE on the individuation reward).

The generator samples captions, they are embedded by the frozen text encoder,
scored by `IndividuationReward` (identity alignment - attribute leakage +
fluency), and the policy is updated by REINFORCE with a moving-average baseline.

This module is cluster-only (imports torch and the generator). See
`scripts/train_generator.py` for the CLI entry point.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict

import torch

from .losses import IndividuationReward, RewardWeights

_WORD_RE = re.compile(r"[A-Za-z']+|[0-9]+")


def count_propnoun_tokens(captions: list[str]) -> list[float]:
    """Per-caption count of mid-sentence capitalised + digit-bearing words.

    A cheap, tokenizer-free proxy for confabulated named entities (invented
    names/places/ages/dates). A word is counted if it contains a digit, or if it
    is capitalised and not the first word of a sentence (so a leading ``A`` or a
    word after ``.``/``!``/``?`` is not penalised).
    """
    counts = []
    for cap in captions:
        words = cap.split()
        n = 0.0
        for i, w in enumerate(words):
            core = w.strip(".,!?;:\"'()")
            if not core:
                continue
            if any(ch.isdigit() for ch in core):
                n += 1.0
                continue
            sentence_start = i == 0 or words[i - 1][-1:] in ".!?"
            if core[:1].isupper() and not sentence_start:
                n += 1.0
        counts.append(n)
    return counts


def count_repetition(captions: list[str]) -> list[float]:
    """Per-caption degeneracy score in [0, 1] (unigram + bigram repetition).

    Greedy decoding of the prefix-tuned LM tends to loop ("beautiful ...
    beautiful ... beautiful", "passionate and passionate"). This tokenizer-free
    proxy averages the fraction of repeated unigrams and repeated bigrams so the
    reward can penalise looping without constraining the decoder itself.
    """
    fracs = []
    for cap in captions:
        words = [w.lower() for w in cap.split()]
        n = len(words)
        if n < 2:
            fracs.append(0.0)
            continue
        uni_rep = 1.0 - len(set(words)) / n
        bigrams = [(words[i], words[i + 1]) for i in range(n - 1)]
        bi_rep = 1.0 - len(set(bigrams)) / len(bigrams)
        fracs.append(0.5 * uni_rep + 0.5 * bi_rep)
    return fracs


# Off-topic / unsafe lexicon: biographical persona filler + explicit terms the
# CLIP grounding reward can hack toward. Penalised so captions stay on the face.
_OFFTOPIC_WORDS = frozenset({
    "actor", "actress", "singer", "musician", "career", "industry", "star",
    "famous", "popular", "celebrity", "pioneer", "legend", "icon", "married",
    "divorced", "husband", "wife", "hobbies", "sports", "movies", "humor",
    "humour", "personality",
    "sexy", "nipples", "tits", "boobs", "breasts", "cock", "dick", "pussy",
    "naked", "nude", "porn", "horny", "cum", "bombshell", "babe", "erotic",
    "orgasm",
})


def count_offtopic_tokens(captions: list[str]) -> list[float]:
    """Per-caption count of off-topic persona / explicit words (0 = on-face).

    The CLIP grounding reward can raise cosine similarity by drifting into
    biographical persona filler ("actress", "musician") or CLIP's explicit-text
    subspace. This tokenizer-free lexical count penalises both so the generator
    is pushed back toward describable facial content. No-op when
    offtopic_weight == 0.
    """
    counts = []
    for cap in captions:
        words = [w.strip(".,!?;:'()").lower() for w in cap.split()]
        counts.append(float(sum(1 for w in words if w in _OFFTOPIC_WORDS)))
    return counts


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
        propnoun_counts = torch.as_tensor(
            count_propnoun_tokens(captions), dtype=torch.float32, device=self.device
        )
        rep_fracs = torch.as_tensor(
            count_repetition(captions), dtype=torch.float32, device=self.device
        )
        offtopic_counts = torch.as_tensor(
            count_offtopic_tokens(captions), dtype=torch.float32, device=self.device
        )
        reward, comps = self.reward(
            text_emb, id_anchor, attr_pred, attr_true,
            base_logprob / seq_len, (policy_logprob / seq_len).detach(),
            propnoun_counts=propnoun_counts,
            rep_fracs=rep_fracs,
            offtopic_counts=offtopic_counts,
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
        import signal

        done_marker = self.cfg.ckpt_path + ".done" if self.cfg.ckpt_path else ""
        if done_marker and os.path.exists(done_marker):
            print(f"[skip] training already complete ({done_marker} exists)")
            return []

        history = []
        start = 0
        if self.cfg.resume and self.cfg.ckpt_path and os.path.exists(self.cfg.ckpt_path):
            start = self._load_ckpt(self.cfg.ckpt_path)
            print(f"[resume] loaded {self.cfg.ckpt_path} at step {start}")

        # Track the latest committed step so a kill-signal handler can flush an
        # up-to-date checkpoint (SLURM wall-time sends SIGTERM/SIGUSR1 before
        # SIGKILL; chained jobs then resume with no lost work).
        self._step = start

        def _on_signal(signum, _frame):
            if self.cfg.ckpt_path:
                try:
                    self._save_ckpt(self.cfg.ckpt_path, self._step)
                    print(f"[signal {signum}] checkpoint saved at step "
                          f"{self._step}; exiting")
                except Exception as exc:  # best-effort: never mask the shutdown
                    print(f"[signal {signum}] checkpoint save failed: {exc}")
            raise SystemExit(0)

        prev_handlers = {}
        for sig_name in ("SIGTERM", "SIGINT", "SIGUSR1"):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            try:
                prev_handlers[sig] = signal.signal(sig, _on_signal)
            except (ValueError, OSError):
                pass  # not main thread / unsupported platform (e.g. Windows SIGUSR1)

        try:
            for it in range(start, self.cfg.steps):
                self._step = it
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
                    self._step = it + 1
            if self.cfg.ckpt_path:
                self._save_ckpt(self.cfg.ckpt_path, self.cfg.steps)
                open(done_marker, "w").close()
        finally:
            for sig, handler in prev_handlers.items():
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):
                    pass
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
