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


def _stub_broken_torch_companions() -> None:
    """Neutralise broken torch companion libs so GPT-2 loading still works.

    Recent ``transformers`` versions eagerly import torchvision and torchaudio
    (via their object-detection loss / audio processing utilities) the moment a
    causal LM is loaded. If those libs have broken compiled extensions (a
    torch/torchvision/torchaudio ABI mismatch -- common with nightly torch,
    e.g. ``operator torchvision::nms does not exist`` or
    ``libtorchaudio.so: undefined symbol``), that import aborts GPT-2 loading
    even though the text-generation path never touches them. For each installed
    companion that fails to import, install a lazy stub so the unused imports
    resolve harmlessly. Healthy libs are left completely untouched.
    """
    import sys
    import types
    import importlib
    import importlib.abc
    import importlib.machinery

    broken: set = set()
    for _lib in ("torchvision", "torchaudio"):
        try:
            importlib.import_module(_lib)
        except ModuleNotFoundError:
            continue  # genuinely not installed -> let transformers' guards handle it
        except Exception:
            broken.add(_lib)
            for _name in list(sys.modules):  # drop partial modules
                if _name == _lib or _name.startswith(_lib + "."):
                    del sys.modules[_name]
    if not broken:
        return

    class _Dummy:
        """Permissive placeholder for any stubbed symbol (never actually used).

        Supports arbitrary attribute access, calls, subscripting and iteration so
        that ``transformers`` import-time expressions (e.g. building an
        ``InterpolationMode`` lookup table) succeed. Distinct instances per access
        keep dict keys/values from colliding.
        """

        def __getattr__(self, name):
            return _Dummy()

        def __call__(self, *args, **kwargs):
            return _Dummy()

        def __getitem__(self, key):
            return _Dummy()

        def __iter__(self):
            return iter(())

    class _StubModule(types.ModuleType):
        __path__: list = []  # mark as a package so submodule imports resolve

        def __getattr__(self, name):
            return _Dummy()

    class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in broken:
                return importlib.machinery.ModuleSpec(name, self)
            return None

        def create_module(self, spec):
            mod = _StubModule(spec.name)
            mod.__path__ = []
            mod.__version__ = "0.0.0+stub"
            return mod

        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, _StubFinder())


_stub_broken_torch_companions()


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
                 temperature: float = 1.0, greedy: bool = False,
                 no_repeat_ngram_size: int = 0, repetition_penalty: float = 1.0,
                 bad_token_ids=None):
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
            if no_repeat_ngram_size > 0 and generated.shape[1] >= no_repeat_ngram_size:
                self._apply_no_repeat(logits, generated, no_repeat_ngram_size)
            if repetition_penalty and repetition_penalty != 1.0:
                self._apply_repetition_penalty(logits, generated, repetition_penalty)
            if bad_token_ids:
                logits[:, bad_token_ids] = float("-inf")
            if greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, nxt], dim=1)
            inputs_embeds = torch.cat([prefix, self._embed_tokens(generated)], dim=1)
        return generated

    @staticmethod
    def _apply_no_repeat(logits: torch.Tensor, generated: torch.Tensor, n: int) -> None:
        """Mask tokens that would complete an already-seen n-gram (in place)."""
        for i in range(generated.shape[0]):
            seq = generated[i].tolist()
            cur = tuple(seq[-(n - 1):]) if n > 1 else ()
            for start in range(len(seq) - n + 1):
                ngram = tuple(seq[start:start + n])
                if ngram[:-1] == cur:
                    logits[i, ngram[-1]] = float("-inf")

    @staticmethod
    def _apply_repetition_penalty(logits: torch.Tensor, generated: torch.Tensor,
                                  penalty: float) -> None:
        """CTRL-style repetition penalty on already-generated tokens (in place).

        Down-weights tokens the sequence has already emitted so greedy decoding
        stops looping ("beautiful ... beautiful ..."). penalty > 1 discourages
        repeats; 1.0 is a no-op.
        """
        for i in range(generated.shape[0]):
            for tok in set(generated[i].tolist()):
                if logits[i, tok] > 0:
                    logits[i, tok] /= penalty
                else:
                    logits[i, tok] *= penalty

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


def _safe_device(device: str) -> str:
    """Fall back to CPU when CUDA is requested but unavailable (old driver / CPU node)."""
    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        import warnings
        warnings.warn(f"CUDA requested ('{device}') but unavailable -> using CPU")
        return "cpu"
    return device


def load_generator(path: str, device: str = "cpu") -> "IndividuateNet":
    """Rebuild an IndividuateNet and load the trained context encoder weights.

    Reads only ``context`` + ``gen_cfg`` so it also accepts the richer training
    checkpoint (which additionally carries optimiser/step state).
    """
    device = _safe_device(device)
    ckpt = torch.load(path, map_location=device)
    cfg = GeneratorConfig(**ckpt["gen_cfg"])
    model = IndividuateNet(cfg).to(device)
    model.context.load_state_dict(ckpt["context"])
    model.eval()
    return model


_NSFW_WORDS = (
    "sexy", "nipples", "tits", "boobs", "breasts", "cock", "dick", "pussy",
    "naked", "nude", "porn", "horny", "cum", "erotic", "orgasm", "bombshell",
    "babe", "butthole", "asshole", "shit", "fuck", "bitch",
)


def _nsfw_bad_token_ids(tokenizer) -> list:
    """Single-token ids for explicit words (+ space/case variants) to hard-block."""
    ids = set()
    for w in _NSFW_WORDS:
        for v in (w, " " + w, w.capitalize(), " " + w.capitalize(),
                  w.upper(), " " + w.upper()):
            toks = tokenizer.encode(v)
            if len(toks) == 1:
                ids.add(int(toks[0]))
    return sorted(ids)


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
    device = _safe_device(device)
    bad_ids = _nsfw_bad_token_ids(model.tokenizer)
    ctx = torch.as_tensor(np.asarray(context, dtype="float32"), device=device)
    caps: list = []
    for i in range(0, ctx.shape[0], batch_size):
        tok = model.generate(ctx[i:i + batch_size], max_new_tokens, greedy=greedy,
                             no_repeat_ngram_size=3, repetition_penalty=1.3,
                             bad_token_ids=bad_ids)
        caps.extend(model.decode(tok))
    out = []
    for c in caps:
        c = " ".join(c.split()).strip()
        out.append(c if c else "a person")
    return out
