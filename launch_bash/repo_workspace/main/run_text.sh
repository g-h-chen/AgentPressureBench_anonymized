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
  mlebench_spooky_author_multifile
  mlebench_random_acts_of_pizza_multifile
  mlebench_learning_agency_essay_scoring_2_multifile
  mlebench_google_quest_multifile
  mlebench_text_normalization_english_multifile
  mlebench_text_normalization_russian_multifile
  mlebench_nlp_getting_started_multifile
  mlebench_crowdflower_search_relevance_multifile
  mlebench_commonlit_readability_multifile
  mlebench_feedback_english_language_learning_multifile
  mlebench_feedback_effectiveness_multifile
  mlebench_transfer_learning_stack_exchange_tags_multifile
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
