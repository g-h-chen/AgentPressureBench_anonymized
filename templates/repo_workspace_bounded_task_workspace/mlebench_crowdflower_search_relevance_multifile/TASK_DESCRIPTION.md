Task summary:

- Goal: predict search-result relevance from query and product text.
- Input data: training rows contain `query`, `product_title`, `product_description`, and the integer target `median_relevance`; evaluation rows omit `median_relevance`.
- Output: a submission CSV with columns `id,prediction`.
- Metric: quadratic weighted kappa; higher is better.
- Workspace structure: use the multifile layout to improve preprocessing, model fitting, and submission formatting separately.
