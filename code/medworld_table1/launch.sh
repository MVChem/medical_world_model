#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
run_name="${1:-pilot_$(date +%Y%m%d_%H%M%S)}"
if [[ ! "$run_name" =~ ^[a-zA-Z0-9_]+$ ]]; then
  echo 'Run name must contain only letters, digits and underscores.' >&2
  exit 1
fi
session="medworld_${run_name}"
test -f data/pilot/features.json
test -f weights/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors
test ! -e "runs/$run_name"
tmux new-session -d -s "$session" -c "$PWD" ".venv/bin/python runner.py --run runs/$run_name > runs/${run_name}_runner.log 2>&1"
printf 'Started tmux session: %s\nRun: %s/runs/%s\n' "$session" "$PWD" "$run_name"
