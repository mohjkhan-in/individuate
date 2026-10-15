#!/usr/bin/env bash
# =============================================================================
# Stage VGGFace2 into the layout IndividuateNet expects:
#
#   $DEST/
#     train/
#       n000002/0001_01.jpg ...
#       n000003/0001_01.jpg ...
#     MAAD_Face.csv            (optional, enables the attribute baseline/penalty)
#
# The loader (individuatenet/data/real.py::VGGFace2Dataset) derives the identity
# from the folder name (int(name.lstrip("n"))) and keys MAAD attributes by the
# relative path "nXXXXXX/img.jpg" (first CSV column is the filename key).
#
# VGGFace2 is registration-gated; there is no single canonical public URL, so
# this script is source-agnostic. Point it at whatever you already have:
#
#   Local archive you downloaded:
#     VGGFACE2_TRAIN_TAR=/path/vggface2_train.tar.gz \
#     MAAD_FACE_CSV=/path/MAAD_Face.csv \
#     bash scripts/prepare_vggface2.sh
#
#   Remote mirror (wget/curl):
#     VGGFACE2_TRAIN_TAR="https://host/vggface2_train.tar.gz" \
#     MAAD_FACE_CSV="https://host/MAAD_Face.csv" \
#     bash scripts/prepare_vggface2.sh
#
#   Kaggle mirror (needs ~/.kaggle/kaggle.json):
#     VGGFACE2_KAGGLE="owner/dataset-slug" \
#     MAAD_FACE_KAGGLE="owner/maad-slug" \
#     bash scripts/prepare_vggface2.sh
#
# Env knobs:
#   DEST                 destination dir           (default: data_local/vggface2)
#   VGGFACE2_TRAIN_TAR   local path OR http(s) URL to the train tar(.gz)
#   MAAD_FACE_CSV        local path OR http(s) URL to MAAD_Face.csv
#   VGGFACE2_KAGGLE      kaggle dataset slug for the images (owner/name)
#   MAAD_FACE_KAGGLE     kaggle dataset slug for MAAD-Face (owner/name)
#   MAX_IDS              keep only the first N identity folders (0 = all)
# =============================================================================
set -euo pipefail

DEST="${DEST:-data_local/vggface2}"
TRAIN_DIR="$DEST/train"
MAAD_DST="$DEST/MAAD_Face.csv"
MAX_IDS="${MAX_IDS:-0}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$DEST"

log()  { printf '[prepare_vggface2] %s\n' "$*"; }
die()  { printf '[prepare_vggface2] ERROR: %s\n' "$*" >&2; exit 1; }
is_url() { case "$1" in http://*|https://*) return 0;; *) return 1;; esac; }

fetch() {  # fetch <src-url-or-path> <dst-file>
  local src="$1" dst="$2"
  if is_url "$src"; then
    log "downloading $src"
    if command -v wget >/dev/null 2>&1; then
      wget --no-verbose --show-progress -O "$dst" "$src"
    else
      curl -fL --progress-bar -o "$dst" "$src"
    fi
  else
    [ -f "$src" ] || die "local source not found: $src"
    log "copying $src"
    cp -f "$src" "$dst"
  fi
}

kaggle_download() {  # kaggle_download <slug> <out-dir>
  command -v kaggle >/dev/null 2>&1 || die "kaggle CLI not found (pip install kaggle)"
  [ -f "$HOME/.kaggle/kaggle.json" ] || die "missing ~/.kaggle/kaggle.json"
  log "kaggle datasets download $1"
  kaggle datasets download -d "$1" -p "$2" --unzip
}

# ---------------------------------------------------------------------------
# 1) Images -> $DEST/train/nXXXXXX/*.jpg
# ---------------------------------------------------------------------------
if [ -d "$TRAIN_DIR" ] && [ -n "$(find "$TRAIN_DIR" -mindepth 1 -maxdepth 1 -type d -print -quit)" ]; then
  log "train/ already populated, skipping image download"
else
  if [ -n "${VGGFACE2_TRAIN_TAR:-}" ]; then
    arc="$TMP/vggface2_train.tar"
    case "$VGGFACE2_TRAIN_TAR" in *.gz|*.tgz) arc="$arc.gz";; esac
    fetch "$VGGFACE2_TRAIN_TAR" "$arc"
    log "extracting archive"
    mkdir -p "$TMP/extract"
    tar -xf "$arc" -C "$TMP/extract"
  elif [ -n "${VGGFACE2_KAGGLE:-}" ]; then
    mkdir -p "$TMP/extract"
    kaggle_download "$VGGFACE2_KAGGLE" "$TMP/extract"
  else
    die "no image source. Set VGGFACE2_TRAIN_TAR=<tar|url> or VGGFACE2_KAGGLE=<slug>."
  fi

  # Locate the directory that directly contains the nXXXXXX identity folders.
  src_train="$(find "$TMP/extract" -type d -name 'n[0-9]*' -printf '%h\n' 2>/dev/null | sort -u | head -n1 || true)"
  [ -n "$src_train" ] || die "could not find nXXXXXX identity folders in the archive"
  log "identity folders found under: $src_train"
  mkdir -p "$TRAIN_DIR"
  # Move identity dirs into place (works whether archive root was train/ or vggface2/train/).
  find "$src_train" -mindepth 1 -maxdepth 1 -type d -name 'n[0-9]*' -print0 \
    | xargs -0 -I{} mv -n "{}" "$TRAIN_DIR/"
fi

# Optional: trim to the first MAX_IDS identities (smaller cluster footprint).
if [ "$MAX_IDS" -gt 0 ]; then
  log "trimming to first $MAX_IDS identities"
  mapfile -t keep < <(find "$TRAIN_DIR" -mindepth 1 -maxdepth 1 -type d -name 'n[0-9]*' | sort | head -n "$MAX_IDS")
  find "$TRAIN_DIR" -mindepth 1 -maxdepth 1 -type d -name 'n[0-9]*' | sort | tail -n +"$((MAX_IDS+1))" \
    | xargs -r rm -rf
fi

# ---------------------------------------------------------------------------
# 2) MAAD-Face attributes -> $DEST/MAAD_Face.csv (optional)
# ---------------------------------------------------------------------------
if [ -f "$MAAD_DST" ]; then
  log "MAAD_Face.csv already present, skipping"
elif [ -n "${MAAD_FACE_CSV:-}" ]; then
  fetch "$MAAD_FACE_CSV" "$MAAD_DST"
elif [ -n "${MAAD_FACE_KAGGLE:-}" ]; then
  kaggle_download "$MAAD_FACE_KAGGLE" "$TMP/maad"
  found="$(find "$TMP/maad" -iname '*.csv' | head -n1 || true)"
  [ -n "$found" ] || die "no CSV found in MAAD kaggle download"
  cp -f "$found" "$MAAD_DST"
else
  log "no MAAD source given -> running without attributes (attr baseline/penalty disabled)."
  log "      set MAAD_FACE_CSV / MAAD_FACE_KAGGLE and set data.attr_csv: MAAD_Face.csv to enable."
fi

# ---------------------------------------------------------------------------
# 3) Verify layout
# ---------------------------------------------------------------------------
n_ids=$(find "$TRAIN_DIR" -mindepth 1 -maxdepth 1 -type d -name 'n[0-9]*' | wc -l)
n_jpg=$(find "$TRAIN_DIR" -type f -name '*.jpg' | head -n 5000 | wc -l)
[ "$n_ids" -gt 0 ] || die "no identity folders staged under $TRAIN_DIR"
[ "$n_jpg" -gt 0 ] || die "no .jpg images found under $TRAIN_DIR"
log "staged $n_ids identities, >=$n_jpg jpgs (sampled) under $TRAIN_DIR"

sample_img="$(find "$TRAIN_DIR" -type f -name '*.jpg' -print -quit)"
sample_rel="${sample_img#"$TRAIN_DIR"/}"
log "example image key: $sample_rel"

if [ -f "$MAAD_DST" ]; then
  header="$(head -n1 "$MAAD_DST")"
  key_col="${header%%,*}"
  log "MAAD_Face.csv present; key column = '$key_col', $(($(wc -l < "$MAAD_DST") - 1)) rows"
  if grep -q -F "$sample_rel" "$MAAD_DST"; then
    log "OK: sample image key found in MAAD_Face.csv"
  else
    log "WARN: '$sample_rel' not found in MAAD_Face.csv -> check the CSV filename column uses 'nXXXXXX/img.jpg' keys"
  fi
fi

cat <<EOF

[prepare_vggface2] DONE.
  Images : $TRAIN_DIR  ($n_ids identities)
  Attrs  : $([ -f "$MAAD_DST" ] && echo "$MAAD_DST" || echo "<none>")

Config  (configs/vggface2.yaml):
  data:
    root: $DEST
    image_dir: train$([ -f "$MAAD_DST" ] && printf '\n    attr_csv: MAAD_Face.csv' || true)

Next:
  python scripts/train_generator.py --config configs/vggface2.yaml --out \$HOME/inet_ckpt/vggface2
  python scripts/run_e1.py --config configs/vggface2.yaml \\
    --generator_ckpt \$HOME/inet_ckpt/vggface2/generator.pt \\
    --max_ids 300 --min_imgs 6 --max_probe_per_id 4 --dump_captions 20
EOF
