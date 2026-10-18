"""Precompute free-text VLM captions for a real dataset (the ``rich`` E1 baseline).

Runs a frozen image-captioning VLM (BLIP by default) over each identity's images
and writes ``{image_path: caption}`` to JSON. ``run_e1.py --rich_captions <json>``
then uses these as a genuine free-text VLM enrolment baseline for the ``rich``
source, instead of the attribute-text fallback.

The captions are per-image, so ``run_e1`` can pick a caption from a *context*
image of each identity (never a held-out probe) -- no probe leakage, parallel to
how the learned ``generated`` source builds its context.

Cluster-only: needs torch + transformers + Pillow and the images on disk. Writes
incrementally and resumes from an existing ``--out`` file.

Example:
  python scripts/caption_vlm.py --config configs/vggface2.yaml \
    --out results/rich_vggface2.json --min_imgs 6 --max_ids 300 \
    --max_images_per_id 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from individuatenet.utils import load_config
from individuatenet.data.real import load_dataset


def _load_captioner(model_name: str, device: str):
    """Load a BLIP conditional-captioning model + processor (frozen, eval)."""
    from transformers import BlipForConditionalGeneration, BlipProcessor

    proc = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device).eval()
    return proc, model


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Precompute free-text VLM captions (rich E1 baseline)."
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results/rich_captions.json")
    ap.add_argument("--model", default="Salesforce/blip-image-captioning-large",
                    help="HF image-captioning model id (BLIP conditional)")
    ap.add_argument("--prompt", default="a photo of the face of",
                    help="conditional caption prefix; '' for unconditional captioning")
    ap.add_argument("--min_imgs", type=int, default=6,
                    help="only caption identities with at least this many images")
    ap.add_argument("--max_ids", type=int, default=0, help="cap identities (0 = all)")
    ap.add_argument("--max_images_per_id", type=int, default=20,
                    help="cap images captioned per identity (bounds cost)")
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default=None,
                    help="override device (default: cfg.device or cuda if available)")
    args = ap.parse_args()

    import torch
    from PIL import Image

    cfg = load_config(args.config)
    device = args.device or cfg.get(
        "device", "cuda" if torch.cuda.is_available() else "cpu"
    )
    ds = load_dataset(cfg)

    by_id: dict = {}
    for s in ds:
        by_id.setdefault(s.identity, []).append(s)
    ids = sorted(i for i, v in by_id.items() if len(v) >= args.min_imgs)
    if args.max_ids:
        ids = ids[: args.max_ids]

    paths: list = []
    for i in ids:
        lim = args.max_images_per_id or None  # 0 -> all images
        imgs = sorted(by_id[i], key=lambda s: s.image_path)[:lim]
        paths.extend(s.image_path for s in imgs)
    print(f"[caption_vlm] {len(ids)} identities, {len(paths)} images "
          f"-> {args.out}  (model={args.model}, device={device})")

    out = Path(args.out)
    caps: dict = {}
    if out.exists():
        caps = json.loads(out.read_text(encoding="utf-8"))
        paths = [p for p in paths if p not in caps]
        print(f"[caption_vlm] resuming: {len(caps)} cached, {len(paths)} remaining")

    proc, model = _load_captioner(args.model, device)

    def _flush():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(caps), encoding="utf-8")

    with torch.no_grad():
        for b in range(0, len(paths), args.batch_size):
            batch = paths[b : b + args.batch_size]
            images = [Image.open(p).convert("RGB") for p in batch]
            texts = [args.prompt] * len(images) if args.prompt else None
            inputs = proc(images=images, text=texts,
                          return_tensors="pt", padding=True).to(device)
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            decoded = proc.batch_decode(gen, skip_special_tokens=True)
            for p, c in zip(batch, decoded):
                caps[p] = c.strip()
            if (b // args.batch_size) % 20 == 0:
                _flush()
                print(f"  {b + len(batch)}/{len(paths)} captioned ...")

    _flush()
    print(f"[caption_vlm] saved {len(caps)} captions -> {out}")


if __name__ == "__main__":
    main()
