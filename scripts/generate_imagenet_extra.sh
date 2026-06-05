#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/generate_imagenet_extra.sh \
    [--data-root /home/ec2-user/datasets/imagenet] \
    [--extra-root /home/ec2-user/datasets/imagenet_extra] \
    [--python-bin /path/to/python]

Notes:
  - Generates ImageNet metadata files required by dinov3 ImageNet loader.
  - Requires labels.txt under --data-root.
  - You may pass the same path for --data-root and --extra-root if you want
    metadata files to be stored alongside the dataset.
EOF
}

DATA_ROOT="/home/ec2-user/datasets/imagenet"
EXTRA_ROOT="/home/ec2-user/datasets/imagenet_extra"
PYTHON_BIN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --extra-root)
      EXTRA_ROOT="$2"
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

if [[ ! -f "$DATA_ROOT/labels.txt" ]]; then
  echo "labels.txt not found under data root: $DATA_ROOT/labels.txt" >&2
  exit 1
fi

mkdir -p "$EXTRA_ROOT"
export PYTHONPATH="$REPO_ROOT"

echo "Generating ImageNet extra metadata"
echo "Repo root : $REPO_ROOT"
echo "Python    : $PYTHON_BIN"
echo "Data root : $DATA_ROOT"
echo "Extra root: $EXTRA_ROOT"

cd "$REPO_ROOT"
"$PYTHON_BIN" -c '
from dinov3.data.datasets import ImageNet

root = r"'"$DATA_ROOT"'"
extra = r"'"$EXTRA_ROOT"'"

for split in ImageNet.Split:
    print(f"dumping metadata for {split.name} -> {extra}")
    dataset = ImageNet(split=split, root=root, extra=extra)
    dataset.dump_extra()

print("done")
'
