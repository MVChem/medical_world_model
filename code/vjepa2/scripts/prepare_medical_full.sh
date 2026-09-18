#!/usr/bin/env bash

set -euo pipefail

mode=${1:---full}
source_root=${MEDICAL_PRETRAINING_ROOT:-/home/data2/chk/data/medical_pretraining}
if [[ "$mode" == "--smoke" ]]; then
  output_dir=${VJEPA2_FULL_DATA_DIR:-/tmp/vjepa2-medical-balanced-smoke-data}
elif [[ "$mode" == "--full" ]]; then
  output_dir=${VJEPA2_FULL_DATA_DIR:-$source_root/vjepa2_medical_full_v1}
else
  echo "usage: $0 [--full|--smoke]" >&2
  exit 2
fi

ixi_dir=$output_dir/volumes/ixi
totalseg_ct_dir=$output_dir/volumes/totalseg_ct
totalseg_mri_dir=$output_dir/volumes/totalseg_mri
amos_dir=$output_dir/volumes/amos22
manifest_dir=$output_dir/manifests

ixi_source=$source_root/IXI/sessions
totalseg_ct_zip=$source_root/TotalSegmentator_CT_v2/Totalsegmentator_dataset_v2.zip
totalseg_mri_zip=$source_root/TotalSegmentator_MRI_v2/TotalsegmentatorMRI_dataset_v200.zip
amos_zip=$source_root/AMOS22/amos22.zip

mkdir -p \
  "$ixi_dir" \
  "$totalseg_ct_dir" \
  "$totalseg_mri_dir" \
  "$amos_dir" \
  "$manifest_dir"

if [[ "$mode" == "--smoke" ]]; then
  unzip -nq "$totalseg_ct_zip" \
    s0000/ct.nii.gz s0001/ct.nii.gz s0002/ct.nii.gz s0003/ct.nii.gz \
    -d "$totalseg_ct_dir"
  unzip -nq "$totalseg_mri_zip" \
    s0001/mri.nii.gz s0002/mri.nii.gz s0003/mri.nii.gz s0004/mri.nii.gz \
    -d "$totalseg_mri_dir"

  mapfile -t amos_ct_entries < <(
    unzip -Z1 "$amos_zip" \
      | awk -F/ '
          /^amos22\/images/ && /\/amos_[0-9]+\.nii\.gz$/ {
            id=$NF; sub(/^amos_/, "", id); sub(/\.nii\.gz$/, "", id)
            if ((id + 0) <= 500 && count < 4) { print; count++ }
          }'
  )
  mapfile -t amos_mri_entries < <(
    unzip -Z1 "$amos_zip" \
      | awk -F/ '
          /^amos22\/images/ && /\/amos_[0-9]+\.nii\.gz$/ {
            id=$NF; sub(/^amos_/, "", id); sub(/\.nii\.gz$/, "", id)
            if ((id + 0) > 500 && count < 4) { print; count++ }
          }'
  )
  unzip -nq "$amos_zip" \
    "${amos_ct_entries[@]}" "${amos_mri_entries[@]}" \
    -d "$amos_dir"

  ixi_archive=${IXI_SMOKE_ZIP:-$ixi_source/NITRC_IR_E10452.zip}
  session_id=$(basename "$ixi_archive" .zip)
  mkdir -p "$ixi_dir/$session_id"
  unzip -nq "$ixi_archive" '*/scans/*/resources/NIfTI/files/*.nii.gz' \
    -d "$ixi_dir/$session_id"
else
  if ! unzip -nq "$totalseg_ct_zip" 's*/ct.nii.gz' -d "$totalseg_ct_dir"; then
    echo "warning: validating extracted TotalSegmentator CT volumes after unzip errors" >&2
  fi
  find "$totalseg_ct_dir" -type f -name 'ct.nii.gz' -print0 \
    | xargs -0 -r -P 8 -n 1 bash -c '
        if ! gzip -t "$1" 2>/dev/null; then
          echo "excluding corrupt volume: $1" >&2
          rm -f -- "$1"
        fi
      ' _
  unzip -nq "$totalseg_mri_zip" 's*/mri.nii.gz' -d "$totalseg_mri_dir"
  unzip -nq "$amos_zip" \
    'amos22/imagesTr/*.nii.gz' \
    'amos22/imagesVa/*.nii.gz' \
    'amos22/imagesTs/*.nii.gz' \
    -d "$amos_dir"

  for ixi_archive in "$ixi_source"/*.zip; do
    session_id=$(basename "$ixi_archive" .zip)
    mkdir -p "$ixi_dir/$session_id"
    unzip -nq "$ixi_archive" '*/scans/*/resources/NIfTI/files/*.nii.gz' \
      -d "$ixi_dir/$session_id"
  done
fi

find "$totalseg_ct_dir" -type f -path '*/s*/ct.nii.gz' | sort \
  | awk '{ print $0, 0 }' > "$manifest_dir/totalseg_ct.csv"
find "$totalseg_mri_dir" -type f -path '*/s*/mri.nii.gz' | sort \
  | awk '{ print $0, 0 }' > "$manifest_dir/totalseg_mri.csv"

: > "$manifest_dir/amos_ct.csv"
: > "$manifest_dir/amos_mri.csv"
while IFS= read -r volume_path; do
  volume_id=$(basename "$volume_path")
  volume_id=${volume_id#amos_}
  volume_id=${volume_id%%.*}
  if ((10#$volume_id <= 500)); then
    printf '%s 0\n' "$volume_path" >> "$manifest_dir/amos_ct.csv"
  else
    printf '%s 0\n' "$volume_path" >> "$manifest_dir/amos_mri.csv"
  fi
done < <(find "$amos_dir" -type f -name 'amos_*.nii.gz' | sort)

for sequence in t1 t2 pd mra; do
  scan_name=${sequence^^}-${sequence^^}
  find "$ixi_dir" -type f \
    -path "*/scans/$scan_name/resources/NIfTI/files/*.nii.gz" \
    | sort | awk '{ print $0, 0 }' > "$manifest_dir/ixi_$sequence.csv"
done

find "$ixi_dir" -type d \
  -path '*/scans/DTI-DTI/resources/NIfTI/files' \
  | sort | awk '{ print $0 "/*.nii.gz", 0 }' > "$manifest_dir/ixi_dti.csv"

combined_manifest=$manifest_dir/medical.csv
: > "$combined_manifest"
append_weighted_manifest() {
  local source_manifest=$1
  local source_weight=$2
  local sample_count
  sample_count=$(wc -l < "$source_manifest")
  awk -v source_weight="$source_weight" -v sample_count="$sample_count" \
    '{ print $1, $2, source_weight / sample_count }' "$source_manifest" \
    >> "$combined_manifest"
}

# Volume mixture: 50% CT, 50% MRI. IXI is 22.5% of volume updates and
# each of its five sequence groups receives an equal share.
append_weighted_manifest "$manifest_dir/totalseg_ct.csv" 0.35
append_weighted_manifest "$manifest_dir/amos_ct.csv" 0.15
append_weighted_manifest "$manifest_dir/ixi_t1.csv" 0.045
append_weighted_manifest "$manifest_dir/ixi_t2.csv" 0.045
append_weighted_manifest "$manifest_dir/ixi_pd.csv" 0.045
append_weighted_manifest "$manifest_dir/ixi_mra.csv" 0.045
append_weighted_manifest "$manifest_dir/ixi_dti.csv" 0.045
append_weighted_manifest "$manifest_dir/totalseg_mri.csv" 0.20
append_weighted_manifest "$manifest_dir/amos_mri.csv" 0.075

for manifest in "$manifest_dir"/*.csv; do
  if [[ ! -s "$manifest" ]]; then
    echo "empty manifest: $manifest" >&2
    exit 1
  fi
  printf '%-24s %5d samples\n' "$(basename "$manifest")" "$(wc -l < "$manifest")"
done

echo "prepared medical dataset: $output_dir"
