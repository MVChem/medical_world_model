#!/usr/bin/env bash

set -eu

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
stage_data_dir=${VJEPA2_STAGE1_DATA_DIR:-/home/data2/chk/data/medical_pretraining/vjepa2_stage1}

export VJEPA2_MEDICAL_DATA=${VJEPA2_MEDICAL_DATA:-$stage_data_dir/medical.csv}
export VJEPA2_MIMIC_DATA=${VJEPA2_MIMIC_DATA:-$stage_data_dir/mimic_30k.csv}

cd "$repo_dir"
python -m app.main \
  --fname configs/train_2_1/vitb16/medical-stage1-256px-16f.yaml \
  --devices "${1:-cuda:1}" "${2:-cuda:2}"
