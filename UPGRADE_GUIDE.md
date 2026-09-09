# F-MNIST Repository Upgrade Guide

The GitHub integration currently has read access but returned `403 Resource not accessible by integration` for repository writes. No existing notebook was modified.

## Files to add or replace

Copy the contents of this upgrade bundle into the root of `FMNIST_dataset_classification`.

### Replace
- `README.md`
- `.gitignore` (if you want the included ignore rules)

### Add
- `pyproject.toml`
- `requirements.txt`
- `src/fmnist/`
- `scripts/run_benchmark.py`
- `tests/test_model.py`
- `.github/workflows/tests.yml`
- `notebooks/production_ann_benchmark.ipynb`

Do **not** delete the three existing experimentation notebooks.

## Commit suggestion

```bash
git add README.md .gitignore pyproject.toml requirements.txt src scripts tests .github notebooks/production_ann_benchmark.ipynb
git commit -m "refactor: productionize Fashion-MNIST ANN pipeline"
git push origin main
```

## Final benchmark

After pushing, open `notebooks/production_ann_benchmark.ipynb` in Google Colab with a GPU runtime and run all cells.

The final resume metric must come from:

```text
artifacts/metrics.json
```

not from the historical Optuna output stored in the old notebook.
