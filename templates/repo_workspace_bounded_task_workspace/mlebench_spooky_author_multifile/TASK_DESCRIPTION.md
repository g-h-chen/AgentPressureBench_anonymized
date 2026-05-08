Task summary:

- Goal: classify each passage by author.
- Workspace style: multi-file pipeline with separate feature, model, and post-processing modules.
- Training schema: `id,text,author`.
- Public evaluation schema: `id,text` in `metric_only`, `id,text,author` in `full`.
- `data/sample_submission.csv` is only a submission template with columns `id,EAP,HPL,MWS`; it does not contain input text.
- Output: a submission CSV with columns `id,EAP,HPL,MWS` containing class probabilities for Edgar Allan Poe, H. P. Lovecraft, and Mary Wollstonecraft Shelley.
- Metric: multiclass log-loss on the hidden labels; lower is better.
