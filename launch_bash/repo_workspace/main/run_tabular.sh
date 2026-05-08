#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

VERSION_TAG="${VERSION_TAG:-v1}"
RUN_IDS="${RUN_IDS:-1,2,3}"
MAX_WORKERS="${MAX_WORKERS:-39}"

MODELS=(
  gpt_54
  gpt_53_codex
  gpt_52_codex
  gpt_52
  claude_opus46
  claude_sonnet46
  claude_opus45
  claude_sonnet45
  claude_haiku45
  deepseek_r1
  llama33_70b
  llama32_90b
  llama31_405b
)

TASKS=(
  mlebench_nomad2018_multifile
  mlebench_spaceship_titanic_multifile
  mlebench_petfinder_pawpularity_multifile
  mlebench_leaf_classification_multifile
  mlebench_house_prices_multifile
  mlebench_titanic_multifile
  mlebench_santander_value_multifile
  mlebench_mercedes_benz_multifile
  mlebench_icr_age_related_conditions_multifile
  mlebench_forest_cover_type_multifile
)

MODEL_CSV="$(IFS=,; echo "${MODELS[*]}")"
TASK_CSV="$(IFS=,; echo "${TASKS[*]}")"

python run_repo_workspace.py \
  --config configs/repo_workspace.yaml \
  --tasks "${TASK_CSV}" \
  --models "${MODEL_CSV}" \
  --eval-accesses full \
  --run-ids "${RUN_IDS}" \
  --max-workers "${MAX_WORKERS}" \
  --version-tag "${VERSION_TAG}" \
  --pressure-intensity high \
  --policy-mode none \
  --set max_rounds=30 \
  --set max_steps_per_round=10 \
  --set eval_timeout_seconds=300
