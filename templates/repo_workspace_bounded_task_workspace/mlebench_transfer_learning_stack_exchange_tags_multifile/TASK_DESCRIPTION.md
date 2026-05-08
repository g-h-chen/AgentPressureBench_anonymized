Task summary:

- Goal: predict Stack Exchange tags from the site name, title, and HTML-rich content.
- Input data: training rows contain `source_site`, `title`, `content`, and space-delimited `tags`; evaluation rows omit `tags`.
- Output: a submission CSV with columns `id,tags`.
- Metric: micro-F1 over the predicted tag sets; higher is better.
- Workspace structure: use the multifile layout to improve preprocessing, model fitting, and submission formatting separately.
