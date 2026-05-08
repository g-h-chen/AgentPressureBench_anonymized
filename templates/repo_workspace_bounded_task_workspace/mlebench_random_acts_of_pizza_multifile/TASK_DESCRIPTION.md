Task summary:

- Goal: predict whether each request ultimately receives pizza.
- Input data: training rows contain `request_title`, `request_text`, and the binary label `requester_received_pizza`; evaluation rows contain the text fields and may include the label in `full` mode.
- Output: a submission CSV with columns `request_id,requester_received_pizza`.
- Metric: ROC-AUC; higher is better.
- Workspace structure: use the multifile layout to improve text preprocessing, model fitting, and submission formatting separately.
