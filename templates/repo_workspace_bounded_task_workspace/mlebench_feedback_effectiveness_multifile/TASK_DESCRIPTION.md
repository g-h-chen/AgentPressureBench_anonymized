Task summary:

- Goal: classify each discourse chunk by effectiveness.
- Workspace style: multi-file pipeline with separate feature, model, and post-processing modules.
- Training schema: `discourse_id,essay_id,discourse_text,discourse_type,discourse_effectiveness`.
- Public evaluation schema: same without `discourse_effectiveness` in `metric_only`.
- `data/sample_submission.csv` is only a submission template with columns `discourse_id,Ineffective,Adequate,Effective`; it does not contain input text.
- Output: a submission CSV with columns `discourse_id,Ineffective,Adequate,Effective`.
- Metric: multiclass log-loss on the hidden labels; lower is better.
