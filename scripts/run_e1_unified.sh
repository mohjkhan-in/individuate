#!/usr/bin/env bash
# E1 unified open-set enrolment -> the "\method{} (ours)" row of tab:main.
# Runs the descriptive generator (config of record) under the SemanticBio
# shared open-set split on CelebA and VGGFace2, 3 seeds each.
#
# Env overrides:
#   REPO        repo root            (default $HOME/Individuate/individuatenet)
#   CKDIR       checkpoint root      (default $HOME/inet_ckpt)
#   CELEBA_CK   CelebA generator.pt  (default $CKDIR/e5_full/generator.pt)
#   VGG_CK      VGGFace2 generator.pt(default $CKDIR/vggface2/generator.pt)
#   CELEBA_SPLIT / VGG_SPLIT  SemanticBio test-id lists (exact parity).
#                             Default to splits/*.txt; auto-generated below from
#                             the staged data if missing (CelebA ~= 1,674 ids).
#   CELEBA_IDFILE  identity_CelebA.txt  (for auto-export, optional)
#   CELEBA_IMGDIR  img_align_celeba/    (for auto-export, optional)
#   VGGFACE2_ROOT  VGGFace2 train/      (for auto-export, optional)
set -euo pipefail

REPO=${REPO:-$HOME/Individuate/individuatenet}
CKDIR=${CKDIR:-$HOME/inet_ckpt}
CELEBA_CK=${CELEBA_CK:-$CKDIR/e5_full/generator.pt}
VGG_CK=${VGG_CK:-$CKDIR/vggface2/generator.pt}
cd "$REPO"

CELEBA_SPLIT=${CELEBA_SPLIT:-splits/celeba_semanticbio_test_ids.txt}
VGG_SPLIT=${VGG_SPLIT:-splits/vggface2_semanticbio_test_ids.txt}

# Materialise the SemanticBio splits from staged data if they are not present.
if [ ! -f "$CELEBA_SPLIT" ] && [ -n "${CELEBA_IDFILE:-}" ]; then
  echo "=== exporting CelebA SemanticBio split ==="
  python -u scripts/export_semanticbio_splits.py \
    --celeba_identity_file "$CELEBA_IDFILE" \
    ${CELEBA_IMGDIR:+--celeba_image_dir "$CELEBA_IMGDIR"} \
    --out_dir "$(dirname "$CELEBA_SPLIT")"
fi
if [ ! -f "$VGG_SPLIT" ] && [ -n "${VGGFACE2_ROOT:-}" ]; then
  echo "=== exporting VGGFace2 SemanticBio split ==="
  python -u scripts/export_semanticbio_splits.py \
    --vggface2_root "$VGGFACE2_ROOT" \
    --out_dir "$(dirname "$VGG_SPLIT")"
fi

celeba_split_arg=()
[ -f "$CELEBA_SPLIT" ] && celeba_split_arg=(--split_ids "$CELEBA_SPLIT")
vgg_split_arg=()
[ -f "$VGG_SPLIT" ] && vgg_split_arg=(--split_ids "$VGG_SPLIT")

echo "=== CelebA (unified open-set) ==="
python -u scripts/run_e1_unified.py \
  --config configs/celeba.yaml \
  --generator_ckpt "$CELEBA_CK" \
  --seed_list 7 42 123 --min_imgs 6 --max_probe_per_id 4 \
  --rich_captions results/rich_celeba.json \
  "${celeba_split_arg[@]}" \
  --out results/e1_unified_celeba.json

echo "=== VGGFace2 (unified open-set) ==="
python -u scripts/run_e1_unified.py \
  --config configs/vggface2.yaml \
  --generator_ckpt "$VGG_CK" \
  --seed_list 7 42 123 --min_imgs 6 --max_probe_per_id 4 \
  --rich_captions results/rich_vggface2.json \
  "${vgg_split_arg[@]}" \
  --out results/e1_unified_vggface2.json

echo "[done] wrote results/e1_unified_{celeba,vggface2}.json"
echo "       -> fill the \\method{} (ours) row of tab:main from these."
