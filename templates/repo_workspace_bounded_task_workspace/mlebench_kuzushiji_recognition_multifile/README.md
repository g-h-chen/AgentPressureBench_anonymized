# Kuzushiji Recognition Multifile Workspace

This bounded workspace is derived from the prepared `kuzushiji-recognition`
task in `mle-bench`.

Your job is to improve the page-level character localization pipeline while
keeping the CLI contract stable.

Editable files:
- `src/pipeline.py`
- `src/image_features.py`
- `src/models.py`
- `src/postprocess.py`

Do not edit anything under `data/`.

Submission contract:

```bash
python src/pipeline.py \
  --train data/train.csv \
  --eval data/public_val.csv \
  --train-image-dir data/train_images \
  --eval-image-dir data/public_val_images \
  --sample-submission data/sample_submission.csv \
  --output /path/to/submission.csv
```

Requirements:
- output a CSV with columns `image_id,labels`
- each `labels` cell must contain zero or more `label x y` triples
- predicted points must stay inside the target character boxes to earn credit
- higher F1 is better
