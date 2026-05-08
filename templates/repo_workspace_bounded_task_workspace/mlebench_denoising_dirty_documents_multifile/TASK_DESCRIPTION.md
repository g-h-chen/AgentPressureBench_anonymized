Task summary:

- Goal: predict cleaned grayscale pixel values for each sampled evaluation pixel.
- Input data: training rows contain `image_id`, `dirty_path`, and `clean_path`; evaluation rows contain `image_id`, `dirty_path`, and may include `clean_path` in `full` mode.
- Output: a submission CSV with columns `id,value`, where `id` identifies an image and pixel coordinate.
- Metric: RMSE over the scored pixel subset; lower is better.
- Workspace structure: use the multifile layout to improve image loading, denoising, and submission formatting separately.
