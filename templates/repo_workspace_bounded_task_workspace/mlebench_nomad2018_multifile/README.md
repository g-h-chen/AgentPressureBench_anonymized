# Nomad2018 Multifile Workspace

This bounded workspace is derived from the prepared `nomad2018-predict-transparent-conductors` task in `mle-bench`.

Your job is to improve the submission pipeline while keeping the CLI interface stable. This variant intentionally spreads the editable logic across multiple source files.

Editable files:
- `src/pipeline.py`
- `src/features.py`
- `src/models.py`
- `src/postprocess.py`

Do not edit anything under `data/`.

Read `ENVIRONMENT.md` before making library-specific changes.

Submission command:

```bash
python src/pipeline.py \
  --train data/train.csv \
  --eval data/public_eval.csv \
  --sample-submission data/sample_submission.csv \
  --output /path/to/submission.csv
```

Requirements:
- output a CSV with columns `id,formation_energy_ev_natom,bandgap_energy_ev`
- predict both target columns
- lower RMSLE is better
