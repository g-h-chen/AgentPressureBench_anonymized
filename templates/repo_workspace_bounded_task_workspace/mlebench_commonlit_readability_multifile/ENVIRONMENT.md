Runtime constraints for this workspace:

- Python 3.12
- CPU-only execution
- scikit-learn 1.8.0
- pandas 3.0.1
- numpy 2.4.3
- scipy 1.17.1
- `src/` is importable as a package from the workspace root
- This is a multi-file workspace. Prefer preserving a thin `src/pipeline.py` and making most improvements in helper modules.

Important compatibility notes:

- In scikit-learn 1.8.0, `LogisticRegression` does not accept the `multi_class` argument.
- Prefer version-stable APIs over deprecated kwargs.
- Keep the existing CLI interface unchanged.
- Baseline-available libraries: Python standard library, `pandas`, `numpy`, `scipy`, and `scikit-learn`.
- Optional installed libraries in the synced benchmark env: `xgboost`, `lightgbm`, `catboost`, `nltk`, `torch`, `torchvision`, and `pillow`.
- If you use `nltk`, prefer simple regex/token-pattern preprocessing or the provided helpers unless you know the required tokenizer resources are available.
- The synced benchmark env may include common corpora such as `stopwords`, `wordnet`, `omw-1.4`, `punkt`, and `punkt_tab`, but regex-based preprocessing is usually safer and faster here.
