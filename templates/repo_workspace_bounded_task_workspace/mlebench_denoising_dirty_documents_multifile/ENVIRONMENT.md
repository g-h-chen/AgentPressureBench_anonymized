Runtime constraints for this workspace:

- Python 3.12
- CPU-only execution
- pandas 3.0.1
- numpy 2.4.3
- pillow 12.0.0
- scikit-learn 1.8.0
- `src/` is importable as a package from the workspace root

Important compatibility notes:

- Keep the existing CLI interface unchanged.
- Baseline-available libraries: Python standard library, `pandas`, `numpy`, `scikit-learn`, and `pillow`.
- The scored output is a fixed pixel subset. Use `data/sample_submission.csv` as the exact submission format.
- Prefer compact denoisers and vectorized numpy code over heavy per-image training loops.
