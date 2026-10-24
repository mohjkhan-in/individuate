#!/usr/bin/env python3
"""Upload IndividuateNet training checkpoints to the Hugging Face Hub.

Uploads every checkpoint under ``--ckpt_dir`` (default ``~/inet_ckpt``) to a
single HF *model* repo, preserving the sub-directory layout
(``celeba/generator.pt``, ``e5_full/generator.pt``, ``vggface2/generator.pt`` …).

Setup (once, on the cluster)::

    pip install -U huggingface_hub
    # authenticate as mohjkhan@alumni.iu.edu — either:
    huggingface-cli login            # paste a WRITE token from hf.co/settings/tokens
    #   …or export the token for this shell:
    export HF_TOKEN=hf_xxx

Run::

    python scripts/upload_checkpoints_hf.py \\
        --repo_id <your-hf-username>/individuatenet-checkpoints \\
        --ckpt_dir ~/inet_ckpt --private

``--repo_id`` must start with your HF username or an org you can write to
(NOT the email). The repo is created automatically if it does not exist.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--repo_id",
        default=os.environ.get("HF_CKPT_REPO", "mohjkhan-in/individuatenet-checkpoints"),
        help="target HF repo '<username-or-org>/<name>' (default: $HF_CKPT_REPO or "
        "mohjkhan-in/individuatenet-checkpoints)",
    )
    ap.add_argument(
        "--ckpt_dir",
        default=os.path.expanduser("~/inet_ckpt"),
        help="local checkpoint directory to upload (default ~/inet_ckpt)",
    )
    ap.add_argument("--private", action="store_true", help="create the repo as private")
    ap.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF write token (default: $HF_TOKEN or the cached huggingface-cli login)",
    )
    ap.add_argument(
        "--path_in_repo",
        default=".",
        help="sub-path inside the repo to upload into (default: repo root)",
    )
    ap.add_argument(
        "--allow",
        nargs="*",
        default=None,
        help="optional allow-patterns, e.g. '*.pt' '*.json' (default: everything)",
    )
    ap.add_argument(
        "--commit_message",
        default="Upload IndividuateNet checkpoints",
    )
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir).expanduser()
    if not ckpt_dir.is_dir():
        sys.exit(f"[error] checkpoint directory not found: {ckpt_dir}")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("[error] huggingface_hub not installed. Run: pip install -U huggingface_hub")

    api = HfApi(token=args.token)

    # Report what will be uploaded.
    files = [p for p in ckpt_dir.rglob("*") if p.is_file()]
    total_gb = sum(p.stat().st_size for p in files) / (1024 ** 3)
    print(f"[upload] {len(files)} files, {total_gb:.2f} GiB from {ckpt_dir}")
    print(f"[upload] -> https://huggingface.co/{args.repo_id} "
          f"({'private' if args.private else 'public'})")

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    # Prefer the resumable large-folder uploader when available (big *.pt files).
    try:
        api.upload_large_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(ckpt_dir),
            allow_patterns=args.allow,
        )
        print("[upload] done (upload_large_folder).")
    except AttributeError:
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(ckpt_dir),
            path_in_repo=args.path_in_repo,
            allow_patterns=args.allow,
            commit_message=args.commit_message,
        )
        print("[upload] done (upload_folder).")

    print(f"[upload] view at https://huggingface.co/{args.repo_id}/tree/main")


if __name__ == "__main__":
    main()
