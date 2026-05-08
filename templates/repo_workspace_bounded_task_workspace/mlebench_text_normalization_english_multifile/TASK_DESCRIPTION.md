Task summary:

- Goal: predict the normalized written form for each English token.
- Input data: training rows contain `sentence_id`, `token_id`, `before`, and `after`; evaluation rows contain the same token identifiers plus `before` and may include `after` in `full` mode.
- Output: a submission CSV with columns `id,after`.
- Metric: exact-match accuracy; higher is better.
- Workspace structure: use the multifile layout to improve token preprocessing, normalization rules, and submission formatting separately.
