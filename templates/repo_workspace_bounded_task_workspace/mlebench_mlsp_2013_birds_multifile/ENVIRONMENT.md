Runtime constraints for this workspace:

- Python 3.12
- CPU-only execution
- scikit-learn 1.8.0
- pandas 3.0.1
- numpy 2.4.3
- scipy 1.17.1
- `src/` is importable as a package from the workspace root
Important compatibility notes:

- Keep the existing CLI interface unchanged.
- Baseline-available libraries: Python standard library, `pandas`, `numpy`, `scipy`, and `scikit-learn`.
- Optional installed libraries in the synced benchmark env: `xgboost`, `lightgbm`, `catboost`, `torch`, `torchvision`, and `pillow`.
- Prefer compact linear or tree-based baselines over heavyweight audio models.
