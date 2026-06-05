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
  - If labels.txt is missing, the script will generate it from the vendored
    ImageNet synset map in ultralytics/ultralytics/cfg/datasets/ImageNet.yaml.
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

"$PYTHON_BIN" -c '
import sys

if sys.version_info < (3, 11):
    raise SystemExit(
        f"dinov3 requires Python >= 3.11, but got {sys.version.split()[0]}. "
        "Pass --python-bin /path/to/python3.11 or use a 3.11 virtual environment."
    )
'

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Data root does not exist: $DATA_ROOT" >&2
  exit 1
fi

if [[ ! -f "$DATA_ROOT/labels.txt" ]]; then
  LABEL_SOURCE="$REPO_ROOT/ultralytics/ultralytics/cfg/datasets/ImageNet.yaml"
  if [[ ! -f "$LABEL_SOURCE" ]]; then
    echo "labels.txt not found under data root: $DATA_ROOT/labels.txt" >&2
    echo "Also could not find fallback label source: $LABEL_SOURCE" >&2
    exit 1
  fi

  echo "labels.txt not found, generating from $LABEL_SOURCE"
  "$PYTHON_BIN" -c '
from pathlib import Path

label_source = Path(r"'"$LABEL_SOURCE"'")
labels_path = Path(r"'"$DATA_ROOT"'") / "labels.txt"

mapping = {}
in_map = False
for raw_line in label_source.read_text(encoding="utf-8").splitlines():
    line = raw_line.rstrip()
    if line == "map:":
        in_map = True
        continue
    if not in_map:
        continue
    if not line:
        continue
    if not raw_line.startswith("  "):
        break
    stripped = line.strip()
    if ": " not in stripped:
        continue
    class_id, class_name = stripped.split(": ", 1)
    mapping[class_id] = class_name.replace("_", " ")

if not mapping:
    raise RuntimeError(f"Failed to parse ImageNet labels from {label_source}")

with labels_path.open("w", encoding="utf-8", newline="") as handle:
    for class_id in sorted(mapping):
        handle.write(f"{class_id},{mapping[class_id]}\n")

print(f"generated {labels_path} with {len(mapping)} labels")
'
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
