#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 3 ]]; then
  echo "Usage: bash scripts/arkitscenes_prepare.sh RAW_DIR PLAN_JSON OUTPUT_DIR [converter options...]" >&2
  exit 2
fi
raw_dir=$1
plan=$2
processed_dir=$3
shift 3
python -m src.scripts.convert_arkitscenes --input_dir "$raw_dir" --plan "$plan" \
  --output_dir "$processed_dir" "$@"
# Validate every retained scan in the converted splits, including test-only exports.
python - "$processed_dir" <<'PY'
import json
from pathlib import Path
import subprocess
import sys
root = Path(sys.argv[1])
manifest = json.loads((root / 'manifest.json').read_text())
splits = [s for s, n in manifest['scene_counts'].items() if n]
subprocess.run([sys.executable, '-m', 'src.scripts.check_scannet', '--dataset', 'arkitscenes',
                '--dataset_dir', str(root), '--output_dir', str(root / 'checks'),
                '--max_scenes', '0', '--splits', *splits], check=True)
PY
