#!/usr/bin/env bash

set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export VJEPA2_FULL_DATA_DIR=${VJEPA2_FULL_DATA_DIR:-/home/data2/chk/data/medical_pretraining/vjepa2_medical_full_v1}
export VJEPA2_MIMIC_DATA=${VJEPA2_MIMIC_DATA:-/home/data2/chk/data/medical_pretraining/vjepa2_stage1/mimic_30k.csv}

required_manifests=(
  medical.csv
  totalseg_ct.csv
  amos_ct.csv
  ixi_t1.csv
  ixi_t2.csv
  ixi_pd.csv
  ixi_mra.csv
  ixi_dti.csv
  totalseg_mri.csv
  amos_mri.csv
)
for manifest in "${required_manifests[@]}"; do
  if [[ ! -s "$VJEPA2_FULL_DATA_DIR/manifests/$manifest" ]]; then
    echo "missing manifest: $VJEPA2_FULL_DATA_DIR/manifests/$manifest" >&2
    echo "run scripts/prepare_medical_full.sh --full first" >&2
    exit 1
  fi
done

if [[ $# -eq 0 ]]; then
  devices=(cuda:4 cuda:5)
else
  devices=("$@")
fi
if [[ ${#devices[@]} -ne 2 ]]; then
  echo "medical-full training requires exactly two devices" >&2
  exit 2
fi

cd "$repo_dir"
python -m app.main \
  --fname configs/train_2_1/vitb16/medical-full-256px-16f.yaml \
  --devices "${devices[@]}"
