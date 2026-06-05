#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_distill_convnextv2_nano_vitl16_1gpu.sh \
    --data-root /path/to/imagenet_root \
    --data-extra /path/to/imagenet_extra \
    --teacher-ckpt /path/to/vitl16_teacher.pth \
    --output-dir /path/to/output_dir \
    [--config-file dinov3/configs/train/distillation_convnext/convnextv2_nano_distill_vitl16_1gpu.yaml] \
    [--python-bin /path/to/python]

Notes:
  - Intended for Amazon Linux / bash.
  - Single-GPU launch (nproc_per_node=1).
  - Expects ImageNet-style dataset metadata (root + extra).
EOF
}

CONFIG_FILE="dinov3/configs/train/distillation_convnext/convnextv2_nano_distill_vitl16_1gpu.yaml"
PYTHON_BIN=""
DATA_ROOT=""
DATA_EXTRA=""
TEACHER_CKPT=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --data-extra)
      DATA_EXTRA="$2"
      shift 2
      ;;
    --teacher-ckpt)
      TEACHER_CKPT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --config-file)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$DATA_ROOT" || -z "$DATA_EXTRA" || -z "$TEACHER_CKPT" || -z "$OUTPUT_DIR" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Data root does not exist: $DATA_ROOT" >&2
  exit 1
fi
if [[ ! -d "$DATA_EXTRA" ]]; then
  echo "Data extra does not exist: $DATA_EXTRA" >&2
  exit 1
fi
if [[ ! -f "$TEACHER_CKPT" ]]; then
  echo "Teacher checkpoint does not exist: $TEACHER_CKPT" >&2
  exit 1
fi
if [[ ! -f "$REPO_ROOT/$CONFIG_FILE" ]]; then
  echo "Config file does not exist: $REPO_ROOT/$CONFIG_FILE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="$REPO_ROOT"
DATASET_PATH="ImageNet:split=TRAIN:root=${DATA_ROOT}:extra=${DATA_EXTRA}"

echo "Starting distillation run"
echo "Repo root         : $REPO_ROOT"
echo "Python            : $PYTHON_BIN"
echo "Config file       : $CONFIG_FILE"
echo "Dataset path      : $DATASET_PATH"
echo "Teacher checkpoint: $TEACHER_CKPT"
echo "Output dir        : $OUTPUT_DIR"

cd "$REPO_ROOT"
"$PYTHON_BIN" -m torch.distributed.run --nproc_per_node=1 dinov3/train/train.py \
  --config-file "$CONFIG_FILE" \
  --output-dir "$OUTPUT_DIR" \
  "train.dataset_path=$DATASET_PATH" \
  "distillation.checkpoint_path=$TEACHER_CKPT"
