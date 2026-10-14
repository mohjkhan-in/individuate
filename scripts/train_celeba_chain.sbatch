#!/bin/bash
# Resumable, self-chaining CelebA generator training for SLURM (IIIT-H Ada).
#
# The gnode wall-time is a hard ~6h. This job:
#   1. asks SLURM to send SIGUSR1 ~120s before SIGKILL (--signal=B:USR1@120),
#      which the trainer catches to flush an up-to-date checkpoint;
#   2. self-resubmits (sbatch) until $CKPT/generator.pt.done exists, so training
#      resumes from the last checkpoint with no lost work.
#
# Checkpoints go to durable NFS $HOME (NOT ssd_scratch, which is wiped at job end).
#
# Submit with:   sbatch scripts/train_celeba_chain.sbatch
#
#SBATCH -p u22
#SBATCH --gres=gpu:1
#SBATCH -c 9
#SBATCH -t 06:00:00
#SBATCH -x gnode077
#SBATCH -J inet_train
#SBATCH -o inet_train_%j.out
#SBATCH --signal=B:USR1@120

set -euo pipefail

REPO="$HOME/Individuate"
CKPT="$HOME/inet_ckpt/celeba"
CONFIG="configs/celeba.yaml"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch310
cd "$REPO"
mkdir -p "$CKPT"

# Already finished? nothing to do.
if [ -f "$CKPT/generator.pt.done" ]; then
    echo "[chain] $CKPT/generator.pt.done exists; training complete."
    exit 0
fi

# Launch training in the background so this script can forward SIGUSR1 to it
# (SLURM's --signal=B targets the batch shell, not the python child).
python -u scripts/train_generator.py --config "$CONFIG" --out "$CKPT" &
TRAIN_PID=$!

# Relay SLURM's pre-kill SIGUSR1 to the trainer so it checkpoints and exits.
trap 'echo "[chain] forwarding SIGUSR1 to $TRAIN_PID"; kill -USR1 "$TRAIN_PID" 2>/dev/null || true' USR1

# Wait for training; tolerate the SystemExit(0) raised by the signal handler.
wait "$TRAIN_PID" || true

# Resubmit a continuation job unless training finished.
if [ -f "$CKPT/generator.pt.done" ]; then
    echo "[chain] training complete; not resubmitting."
else
    echo "[chain] wall-time/kill hit before completion; resubmitting to resume."
    sbatch "$REPO/scripts/train_celeba_chain.sbatch"
fi
