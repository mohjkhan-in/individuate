#!/usr/bin/env bash
#SBATCH --job-name=vgg_e1
#SBATCH --output=vgg_e1_%j.out
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --signal=B:USR1@120
# =============================================================================
# Chained, resumable VGGFace2 -> tab:main \method{} cell.
#
# Survives the Ada 6h wall-time + node-local ssd_scratch wipe by:
#   1. staging the 38GB tar from DURABLE /share3 to node-local ssd_scratch,
#   2. training the generator RESUMABLY to DURABLE $HOME/inet_ckpt (not scratch),
#      (train_generator.py checkpoints every save_every steps + handles USR1),
#   3. running the unified open-set E1 once the generator is done,
#   4. self-resubmitting (sbatch) until an E1 .DONE marker exists.
#
# Prereq: run scripts/fetch_vggface2_to_share.sh on the GATEWAY node first so
# $SHARE_TAR (and ideally $MAAD_SRC) live on /share3.
#
# Submit (from ~/Individuate):
#   SHARE_TAR=/share3/$USER/data/vggface2_train.tar.gz \
#   MAAD_SRC=/share3/$USER/data/MAAD_Face.csv \
#   sbatch scripts/slurm_vggface2_e1.sh
#
# Env knobs (all optional, sane defaults):
#   REPO       repo root            (default $HOME/Individuate)
#   SHARE_TAR  durable VGGFace2 tar  (default /share3/$USER/data/vggface2_train.tar.gz)
#   MAAD_SRC   durable MAAD_Face.csv (default /share3/$USER/data/MAAD_Face.csv)
#   SCRATCH    node-local dest       (default /ssd_scratch/$USER/vggface2; MUST match configs/vggface2.yaml:data.root)
#   CKDIR      durable ckpt root     (default $HOME/inet_ckpt)
#   CHAIN_N / MAX_CHAIN  resubmission counter / cap (default 0 / 10)
# =============================================================================
set -uo pipefail

REPO="${REPO:-$HOME/Individuate}"
SHARE_TAR="${SHARE_TAR:-/share3/$USER/data/vggface2_train.tar.gz}"
MAAD_SRC="${MAAD_SRC:-/share3/$USER/data/MAAD_Face.csv}"
SCRATCH="${SCRATCH:-/ssd_scratch/$USER/vggface2}"
CKDIR="${CKDIR:-$HOME/inet_ckpt}"
CHAIN_N="${CHAIN_N:-0}"
MAX_CHAIN="${MAX_CHAIN:-10}"

VGG_CK="$CKDIR/vggface2/generator.pt"
DONE_MARK="$CKDIR/vggface2/e1_vggface2.DONE"
RESULT="$REPO/results/e1_unified_vggface2.json"

echo "[vgg_e1] chain=$CHAIN_N/$MAX_CHAIN  host=$(hostname)  $(date)"

# ---- activate env ----------------------------------------------------------
cd "$REPO"
source "$HOME/.bashrc" 2>/dev/null || true
# shellcheck disable=SC1091
conda activate torch310 2>/dev/null || source activate torch310

resubmit() {
  if [ -f "$DONE_MARK" ]; then echo "[vgg_e1] done, no resubmit."; return; fi
  if [ "$CHAIN_N" -ge "$MAX_CHAIN" ]; then
    echo "[vgg_e1] hit MAX_CHAIN=$MAX_CHAIN, NOT resubmitting (investigate)."; return
  fi
  echo "[vgg_e1] not finished -> resubmitting (chain $((CHAIN_N+1)))"
  sbatch --export=ALL,CHAIN_N=$((CHAIN_N+1)),SHARE_TAR="$SHARE_TAR",MAAD_SRC="$MAAD_SRC",SCRATCH="$SCRATCH",CKDIR="$CKDIR",REPO="$REPO",MAX_CHAIN="$MAX_CHAIN" "$0"
}

# ---- 0. already finished? --------------------------------------------------
if [ -f "$DONE_MARK" ]; then
  echo "[vgg_e1] DONE marker present ($DONE_MARK) -> nothing to do."
  exit 0
fi

# ---- 1. stage data to node-local scratch (wiped per allocation) ------------
mkdir -p "$SCRATCH"
[ -f "$SCRATCH/MAAD_Face.csv" ] || { [ -f "$MAAD_SRC" ] && cp -f "$MAAD_SRC" "$SCRATCH/MAAD_Face.csv" || echo "[vgg_e1] WARN: no MAAD_Face.csv ($MAAD_SRC) -> attribute arm will be degraded"; }

if [ -z "$(find "$SCRATCH/train" -mindepth 1 -maxdepth 1 -type d -name 'n[0-9]*' -print -quit 2>/dev/null)" ]; then
  [ -f "$SHARE_TAR" ] || { echo "[vgg_e1] ERROR: tar not found: $SHARE_TAR (run fetch_vggface2_to_share.sh on the gateway first)"; exit 1; }
  echo "[vgg_e1] staging VGGFace2 from $SHARE_TAR -> $SCRATCH"
  VGGFACE2_TRAIN_TAR="$SHARE_TAR" DEST="$SCRATCH" bash scripts/prepare_vggface2.sh
fi

n_ids=$(find "$SCRATCH/train" -mindepth 1 -maxdepth 1 -type d -name 'n[0-9]*' | wc -l)
echo "[vgg_e1] staged $n_ids identities under $SCRATCH/train"
if [ "$n_ids" -lt 2000 ]; then
  echo "[vgg_e1] ERROR: only $n_ids ids (<2000) -> split cannot yield 400 novel. Aborting, NO resubmit."
  exit 1
fi

# ---- 2. export SemanticBio split (idempotent) ------------------------------
export VGGFACE2_ROOT="$SCRATCH/train"
if [ ! -f splits/vggface2_semanticbio_test_ids.txt ]; then
  echo "[vgg_e1] exporting SemanticBio VGGFace2 split"
  python -u scripts/export_semanticbio_splits.py --vggface2_root "$VGGFACE2_ROOT" --out_dir splits
fi
n_novel=$(wc -l < splits/vggface2_semanticbio_test_ids.txt || echo 0)
echo "[vgg_e1] split novel ids: $n_novel (expect ~400)"

# ---- 3. train generator (resumable; short-circuits if <ckpt>.done) ---------
mkdir -p "$CKDIR/vggface2"
echo "[vgg_e1] training/resuming generator -> $CKDIR/vggface2"
python -u scripts/train_generator.py --config configs/vggface2.yaml --out "$CKDIR/vggface2" &
TRAIN_PID=$!
# Forward SLURM's pre-kill USR1 to the trainer for a graceful final checkpoint.
trap 'echo "[vgg_e1] caught USR1 -> forwarding to trainer $TRAIN_PID"; kill -USR1 "$TRAIN_PID" 2>/dev/null || true' USR1
wait "$TRAIN_PID" || true
trap - USR1

# ---- 4. run E1 if the generator finished -----------------------------------
if [ ! -f "$CKDIR/vggface2/generator.pt.done" ]; then
  echo "[vgg_e1] generator not finished this window -> will resume next job."
  resubmit
  exit 0
fi

echo "[vgg_e1] generator complete -> running unified open-set E1"
rich_arg=()
[ -f results/rich_vggface2.json ] && rich_arg=(--rich_captions results/rich_vggface2.json)
split_arg=()
[ -f splits/vggface2_semanticbio_test_ids.txt ] && split_arg=(--split_ids splits/vggface2_semanticbio_test_ids.txt)

python -u scripts/run_e1_unified.py \
  --config configs/vggface2.yaml \
  --generator_ckpt "$VGG_CK" \
  --seed_list 7 42 123 --min_imgs 6 --max_probe_per_id 4 \
  "${rich_arg[@]}" "${split_arg[@]}" \
  --out "$RESULT"
rc=$?

if [ "$rc" -eq 0 ] && [ -f "$RESULT" ]; then
  cp -f "$RESULT" "$CKDIR/vggface2/e1_unified_vggface2.json"   # durable copy
  touch "$DONE_MARK"
  echo "[vgg_e1] SUCCESS -> $RESULT  (durable copy + DONE marker written)"
  echo "----- result -----"; cat "$RESULT"
else
  echo "[vgg_e1] E1 did not complete (rc=$rc) -> resubmitting"
  resubmit
fi
