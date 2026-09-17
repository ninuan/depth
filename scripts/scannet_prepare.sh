#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/scannet_prepare.sh RAW_FRAMES_DIR OUTPUT_DIR [converter options...]" >&2
  exit 2
fi
raw_dir=$1
processed_dir=$2
shift 2
split_dir=assets/scannet_splits
mkdir -p "$split_dir"
for split in train val; do
  split_file="$split_dir/scannetv2_${split}.txt"
  if [[ ! -s "$split_file" ]]; then
    curl -fL --retry 3 "https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark/scannetv2_${split}.txt" -o "$split_file"
  fi
done
python -m src.scripts.convert_scannet \
  --input_dir "$raw_dir" --output_dir "$processed_dir" \
  --train_split "$split_dir/scannetv2_train.txt" \
  --test_split "$split_dir/scannetv2_val.txt" \
  --frame_stride 1 --val_fraction 0.1 "$@"
python -m src.scripts.check_scannet --dataset_dir "$processed_dir" \
  --output_dir "${processed_dir}/checks"
