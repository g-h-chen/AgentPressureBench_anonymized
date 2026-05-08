Runtime constraints for this workspace:

- Python 3.12
- CPU-only execution
- pandas 3.0.1
- numpy 2.4.3
- pillow 11.3.0
- `src/` is importable as a package from the workspace root

Important compatibility notes:

- Keep the existing CLI interface unchanged.
- Baseline-available libraries: Python standard library, `pandas`, `numpy`, and `pillow`.
- The runtime is CPU-only, so keep any image-processing pipeline lightweight.
