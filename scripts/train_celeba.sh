#!/bin/bash
# Resumable, self-chaining IndividuateNet training on IIIT-H Ada (SLURM).
#
#   sbatch scripts/train_celeba.sh
#
# Why chained + resumable: a gnode allocation has a hard ~6h wall-time and
# ssd_scratch is node-local + wiped at job end, so the ONLY durable filesystem
# reachable inside the job is NFS $HOME. We therefore:
#   * stage CelebA to node-local ssd_scratch for fast IO,
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
SRC_CELEBA="$HOME/datasets/celeba"       # durable source copy of CelebA
SCRATCH="/ssd_scratch/$USER/celeba"      # node-local staging (fast, ephemeral)

mkdir -p "$CKPT_DIR"

# Nothing to do if a previous run already finished.
if [ -f "$CKPT_DIR/generator.pt.done" ]; then
  echo "[done] $CKPT_DIR/generator.pt.done present; stopping chain"
  exit 0
fi

# Queue the follower NOW so it runs even if this job is killed at wall-time.
# It re-checks the DONE marker at the top and exits early if training finished.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  sbatch --dependency=afterany:"$SLURM_JOB_ID" "$0" || true
fi

# --- stage dataset to node-local ssd_scratch ---
mkdir -p "$SCRATCH"
if [ ! -f "$SCRATCH/list_attr_celeba.txt" ]; then
  echo "[stage] copying CelebA -> $SCRATCH"
  rsync -a "$SRC_CELEBA/" "$SCRATCH/"
fi
ln -sfn "$SCRATCH" "$REPO/data_local/celeba"

cd "$REPO"
# shellcheck disable=SC1090
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate torch310
export HF_HOME="$HOME/.cache/huggingface"

# --- train (resumable); writes $CKPT_DIR/generator.pt + .done on completion ---
python scripts/train_generator.py --config configs/celeba.yaml --out "$CKPT_DIR"

echo "[exit] job $SLURM_JOB_ID finished its slice"
