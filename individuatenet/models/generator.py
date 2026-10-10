"""IndividuateNet: a prefix-tuned language model q_phi(T | c).

A frozen causal LM (default GPT-2) is steered by a small context encoder that
maps an identity-context feature ``c`` (pooled from the identity's own images at
*training* time only) into ``prefix_tokens`` soft-prompt embeddings prepended to
the token stream. Only the context encoder (and optional LoRA adapters) is
trained. At deployment no per-identity image gallery is required.

Torch / transformers are imported at module top; this module is only used on the
cluster (the offline path never imports it).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GeneratorConfig:
    base_lm: str = "gpt2"
    prefix_tokens: int = 16
    context_dim: int = 512
    hidden: int = 512
    max_new_tokens: int = 40
    offline: bool = False


class ContextEncoder(nn.Module):
    """Maps a context vector c -> (prefix_tokens x n_embd) soft prompt."""

    def __init__(self, context_dim: int, n_embd: int, prefix_tokens: int, hidden: int):
        super().__init__()
        self.prefix_tokens = prefix_tokens
        self.n_embd = n_embd
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, prefix_tokens * n_embd),
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        b = c.shape[0]
        return self.net(c).view(b, self.prefix_tokens, self.n_embd)


class IndividuateNet(nn.Module):
    def __init__(self, cfg: GeneratorConfig):
        super().__init__()
        self.cfg = cfg
        from transformers import AutoModelForCausalLM, AutoTokenizer

        local_only = cfg.offline
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_lm, local_files_only=local_only
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(
            cfg.base_lm, local_files_only=local_only
        )
        for p in self.lm.parameters():   # freeze the base LM
            p.requires_grad_(False)

        n_embd = self.lm.get_input_embeddings().embedding_dim
        self.context = ContextEncoder(
            cfg.context_dim, n_embd, cfg.prefix_tokens, cfg.hidden
        )

    # -- helpers --------------------------------------------------------------
    def _prefix(self, c: torch.Tensor) -> torch.Tensor:
        return self.context(c)  # (B, prefix, n_embd)

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm.get_input_embeddings()(input_ids)

    # -- training forward: LM loss on reference captions ----------------------
    def forward(self, c: torch.Tensor, input_ids: torch.Tensor, attention_mask=None):
        prefix = self._prefix(c)
        tok_emb = self._embed_tokens(input_ids)
        inputs_embeds = torch.cat([prefix, tok_emb], dim=1)

        b, pfx = prefix.shape[0], prefix.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, device=input_ids.device)
        pfx_mask = torch.ones(b, pfx, device=input_ids.device)
        full_mask = torch.cat([pfx_mask, attention_mask], dim=1)

        # labels: ignore the prefix positions
        ignore = torch.full((b, pfx), -100, device=input_ids.device, dtype=input_ids.dtype)
        labels = torch.cat([ignore, input_ids], dim=1)
        return self.lm(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=labels)

    # -- sampling generation --------------------------------------------------
    @torch.no_grad()
    def generate(self, c: torch.Tensor, max_new_tokens: int | None = None,
                 temperature: float = 1.0, greedy: bool = False):
        max_new_tokens = max_new_tokens or self.cfg.max_new_tokens
        prefix = self._prefix(c)
        b = prefix.shape[0]
        device = prefix.device
        bos = torch.full((b, 1), self.tokenizer.bos_token_id or self.tokenizer.eos_token_id,
                         device=device, dtype=torch.long)
        generated = bos
        inputs_embeds = torch.cat([prefix, self._embed_tokens(generated)], dim=1)

        for _ in range(max_new_tokens):
            out = self.lm(inputs_embeds=inputs_embeds)
            logits = out.logits[:, -1, :] / max(temperature, 1e-5)
            if greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, nxt], dim=1)
            inputs_embeds = torch.cat([prefix, self._embed_tokens(generated)], dim=1)
        return generated

    def decode(self, token_ids: torch.Tensor):
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

    @property
    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)


# -- checkpoint + inference helpers -------------------------------------------
def save_generator(model: "IndividuateNet", path: str) -> None:
    """Atomically save the trainable context encoder + its GeneratorConfig."""
    import dataclasses
    import os

    tmp = str(path) + ".tmp"
    torch.save(
        {"context": model.context.state_dict(), "gen_cfg": dataclasses.asdict(model.cfg)},
        tmp,
    )
    os.replace(tmp, path)


def load_generator(path: str, device: str = "cpu") -> "IndividuateNet":
    """Rebuild an IndividuateNet and load the trained context encoder weights.

    Reads only ``context`` + ``gen_cfg`` so it also accepts the richer training
    checkpoint (which additionally carries optimiser/step state).
    """
    ckpt = torch.load(path, map_location=device)
    cfg = GeneratorConfig(**ckpt["gen_cfg"])
    model = IndividuateNet(cfg).to(device)
    model.context.load_state_dict(ckpt["context"])
    model.eval()
    return model


@torch.no_grad()
def generate_captions(
    model: "IndividuateNet",
    context,
    device: str = "cpu",
    max_new_tokens: int | None = None,
    batch_size: int = 64,
    greedy: bool = True,
) -> list:
    """Generate one caption per context row (deployment / eval helper)."""
    import numpy as np

    model.eval()
    ctx = torch.as_tensor(np.asarray(context, dtype="float32"), device=device)
    caps: list = []
    for i in range(0, ctx.shape[0], batch_size):
        tok = model.generate(ctx[i:i + batch_size], max_new_tokens, greedy=greedy)
        caps.extend(model.decode(tok))
    out = []
    for c in caps:
        c = " ".join(c.split()).strip()
        out.append(c if c else "a person")
    return out
