#!/bin/bash
# Resumable, self-chaining IndividuateNet training on IIIT-H Ada (SLURM).
#
#   sbatch scripts/train_celeba.sh
#
# Why chained + resumable: a gnode allocation has a hard ~6h wall-time, so the
# ONLY durable filesystem reachable inside the job is NFS $HOME. We therefore:
#   * read CelebA DIRECTLY from the durable copy under $HOME (no ssd_scratch copy),
#   * checkpoint atomically to $HOME every save_every steps,
#   * queue a follower job (afterany) that resumes from the last checkpoint,
#     and exits immediately once generator.pt.done exists.
#
# EDIT the #SBATCH -p line if your account needs a different partition.

#SBATCH -J inet_train
#SBATCH -p u22
#SBATCH -G 1
#SBATCH -c 9
#SBATCH -t 06:00:00
#SBATCH -o inet_train_%j.out

set -euo pipefail

REPO="$HOME/Individuate"                 # durable repo clone
CKPT_DIR="$HOME/inet_ckpt/celeba"        # durable checkpoints (NFS home, has quota)

# CelebA is read DIRECTLY from the durable copy under $HOME (no ssd_scratch copy).
# Override with:  SRC_CELEBA=/path/to/celeba sbatch $0
SRC_CELEBA="${SRC_CELEBA:-}"
if [ -z "$SRC_CELEBA" ]; then
  for cand in "$HOME/dataset/celeb" "$HOME/dataset/celeba" "$HOME/dataset" "$HOME/datasets/celeba"; do
    if [ -e "$cand/list_attr_celeba.txt" ]; then SRC_CELEBA="$cand"; break; fi
  done
fi

mkdir -p "$CKPT_DIR"

# Nothing to do if a previous run already finished.
if [ -f "$CKPT_DIR/generator.pt.done" ]; then
  echo "[done] $CKPT_DIR/generator.pt.done present; stopping chain"
  exit 0
fi

# --- validate dataset BEFORE queueing any follower, so a missing-data / startup
#     failure exits cleanly instead of crash-looping ---
if [ -z "$SRC_CELEBA" ] || [ ! -e "$SRC_CELEBA/list_attr_celeba.txt" ]; then
  echo "[fatal] CelebA (list_attr_celeba.txt) not found under \$HOME/dataset[s]" >&2
  echo "        Submit with:  SRC_CELEBA=/path/to/celeba sbatch $0" >&2
  exit 1
fi

# Point the repo at the dataset directly on NFS home (no node-local copy).
echo "[data] using CelebA directly from $SRC_CELEBA"
mkdir -p "$REPO/data_local"
ln -sfn "$SRC_CELEBA" "$REPO/data_local/celeba"

# Staging succeeded -> queue the follower NOW so it resumes if THIS job is killed
# at wall-time. It re-checks the DONE marker at the top and exits early if done.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  sbatch --dependency=afterany:"$SLURM_JOB_ID" "$0" || true
fi

cd "$REPO"
# conda's activate/deactivate scripts reference unset vars (e.g. CONDA_BACKUP_CXX),
# which are fatal under `set -u`; relax it just around activation.
set +u
# shellcheck disable=SC1090
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate torch310
set -u
export HF_HOME="$HOME/.cache/huggingface"
export PYTHONUNBUFFERED=1   # stream [warmup]/[step] prints live into the .out

# --- train (resumable); writes $CKPT_DIR/generator.pt + .done on completion ---
python -u scripts/train_generator.py --config configs/celeba.yaml --out "$CKPT_DIR"

echo "[exit] job $SLURM_JOB_ID finished its slice"
