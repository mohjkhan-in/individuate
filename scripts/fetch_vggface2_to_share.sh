#!/usr/bin/env bash
# =============================================================================
# ONE-TIME: download the full VGGFace2 train tar to DURABLE storage (/share3).
#
# Run this on the Ada LOGIN / GATEWAY node (the only place that can reach BOTH
# the internet AND /share3). Do NOT sbatch this -- gnodes cannot see /share3.
#
# Source: Hugging Face mirror  ProgramComputer/VGGFace2
#   data/vggface2_train.tar.gz   37.9 GB   8,631 identities
# The repo is GATED: you must (1) have an HF account, (2) click "Agree and
# access repository" on https://huggingface.co/datasets/ProgramComputer/VGGFace2
# and (3) create a READ token at https://huggingface.co/settings/tokens.
#
# Usage:
#   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
#   bash scripts/fetch_vggface2_to_share.sh
#
# Env knobs:
#   HF_TOKEN   (required) your HF read token
#   SHARE_DIR  durable dir for the tar (default /share3/$USER/data)
#   MAAD_SRC   optional path to MAAD_Face.csv to copy alongside (for attributes)
# =============================================================================
set -euo pipefail

: "${HF_TOKEN:?set HF_TOKEN to your Hugging Face read token}"
SHARE_DIR="${SHARE_DIR:-/share3/$USER/data}"
mkdir -p "$SHARE_DIR"

echo "[fetch] installing huggingface_hub + hf_xet"
pip install -q -U "huggingface_hub[cli]" hf_xet

echo "[fetch] downloading vggface2_train.tar.gz (37.9 GB) -> $SHARE_DIR"
# 'hf download' resumes partial downloads, so re-running after an interruption is safe.
hf download ProgramComputer/VGGFace2 data/vggface2_train.tar.gz \
  --repo-type dataset \
  --local-dir "$SHARE_DIR/hf_vgg" \
  --token "$HF_TOKEN"

# Stable path the SLURM job expects.
ln -sf "$SHARE_DIR/hf_vgg/data/vggface2_train.tar.gz" "$SHARE_DIR/vggface2_train.tar.gz"

# Optionally place MAAD_Face.csv on durable storage too (needed for attributes).
if [ -n "${MAAD_SRC:-}" ] && [ -f "$MAAD_SRC" ]; then
  cp -f "$MAAD_SRC" "$SHARE_DIR/MAAD_Face.csv"
  echo "[fetch] copied MAAD_Face.csv -> $SHARE_DIR/MAAD_Face.csv"
fi

echo "[fetch] DONE."
ls -lh "$SHARE_DIR/vggface2_train.tar.gz"
echo
echo "Next, from ~/Individuate, submit the chained job:"
echo "    SHARE_TAR=$SHARE_DIR/vggface2_train.tar.gz \\"
echo "    MAAD_SRC=$SHARE_DIR/MAAD_Face.csv \\"
echo "    sbatch scripts/slurm_vggface2_e1.sh"
