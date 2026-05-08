#!/usr/bin/env bash
set -euo pipefail

# Prompt ablation: held-out-test framing

VERSION_TAG="v1_heldout_test"
RUN_IDS="1"
MAX_WORKERS="12"
MODELS="gpt_54,gpt_53_codex,claude_sonnet46,claude_opus46"
TASKS="mlebench_forest_cover_type_multifile,mlebench_text_normalization_russian_multifile,mlebench_kuzushiji_recognition_multifile"

python run_repo_workspace.py \
  --config configs/repo_workspace.yaml \
  --tasks "${TASKS}" \
  --models "${MODELS}" \
  --eval-accesses full \
  --run-ids "${RUN_IDS}" \
  --max-workers "${MAX_WORKERS}" \
  --version-tag "${VERSION_TAG}" \
  --pressure-intensity high \
  --policy-mode none \
  --public-split-frame heldout_test \
  --set max_rounds=30 \
  --set max_steps_per_round=10 \
  --set eval_timeout_seconds=300
