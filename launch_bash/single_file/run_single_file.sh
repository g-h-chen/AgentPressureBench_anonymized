#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python run_single_file.py \
  --config configs/single_file.yaml \
  --parallel 20 \
  --results-dir results/single_file/tabular_clean_full/results \
  --logs-dir logs/single_file/tabular_clean_full \
  --datasets tabular \
  --models gpt_54 claude_opus46 \
  --eval-accesses full \
  --run-ids 0 1 2 3 4 \
  --max-rounds 10
