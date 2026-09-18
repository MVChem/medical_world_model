#!/usr/bin/env bash

set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export VJEPA2_FULL_DATA_DIR=${VJEPA2_SMOKE_DATA_DIR:-/tmp/vjepa2-medical-balanced-smoke-data}
mimic_samples=${MIMIC_SAMPLES:-8}

"$repo_dir/scripts/prepare_medical_full.sh" --smoke

head -n "$mimic_samples" \
  /home/data2/chk/data/medical_pretraining/vjepa2_stage1/mimic_30k.csv \
  > "$VJEPA2_FULL_DATA_DIR/mimic.csv"
export VJEPA2_MIMIC_DATA="$VJEPA2_FULL_DATA_DIR/mimic.csv"

if [[ $# -eq 0 ]]; then
  devices=(cuda:4 cuda:5 cuda:6 cuda:7)
else
  devices=("$@")
fi
if [[ ${#devices[@]} -ne 4 ]]; then
  echo "balanced smoke test requires exactly four devices" >&2
  exit 2
fi

cd "$repo_dir"
python -m app.main \
  --fname configs/train_2_1/vitb16/medical-smoke-64px-8f.yaml \
  --devices "${devices[@]}"
