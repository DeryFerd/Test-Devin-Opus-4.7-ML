# Test-Devin-Opus-4.7-ML

End-to-end ML pipeline for teen mental-health depression classification, built the way an ML engineer would ship a modern project in 2026 — reproducible env (`uv`), typed config (`pydantic`), imbalanced-aware preprocessing (`imblearn` + SMOTE), multi-model benchmark, Optuna HPO, MLflow tracking, pytest + ruff + CI, and every EDA/eval chart saved as a standalone PNG.

> **Dataset**: [`algozee/teenager-menthal-healy`](https://www.kaggle.com/datasets/algozee/teenager-menthal-healy) — 1200 teenagers, 12 lifestyle/self-report features, binary target `depression_label`. Heavily imbalanced (~2.6% positive).

---

## Architecture

```
Test-Devin-Opus-4.7-ML/
├── conf/config.yaml                # Central, typed configuration
├── src/teen_mh/
│   ├── config.py                   # pydantic-validated config loader
│   ├── data.py                     # Kaggle ingest + stratified train/test split
│   ├── features.py                 # sklearn ColumnTransformer (impute + OHE + scale)
│   ├── eda.py                      # Deep-dive EDA, each chart saved as a PNG
│   ├── train.py                    # Multi-model benchmark + SMOTE + CV + MLflow
│   ├── tune.py                     # Optuna HPO on the benchmark winner
│   ├── evaluate.py                 # Test-set metrics + diagnostic PNGs + model card
│   └── cli.py                      # Typer CLI: `teen-mh <subcommand>`
├── tests/                          # pytest: config, data, features, train smoke
├── reports/
│   ├── figures/                    # All EDA + evaluation plots (PNG, one per chart)
│   ├── metrics/                    # eda_summary.json, benchmark.json, tuning.json, test_metrics.json
│   └── MODEL_CARD.md               # Auto-generated after evaluation
├── models/                         # baseline_model.joblib, final_model.joblib (git-ignored)
├── data/{raw,processed}/           # Downloaded CSV + Parquet splits (git-ignored)
├── .github/workflows/ci.yml        # Ruff + pytest on every push/PR
├── pyproject.toml                  # uv / hatchling / ruff / pytest config
└── Makefile                        # Convenience targets
```

### Pipeline stages

1. **Ingest** — pull the dataset from Kaggle's public download endpoint (falls back to the `kaggle` CLI if auth is required), drop into `data/raw/`, then produce a stratified 80/20 train/test split as Parquet in `data/processed/`.
2. **EDA** — 30 charts across target balance, missingness, per-feature distributions (hist + box), categorical counts, numeric-vs-target boxplots, categorical depression-rate bars, correlation heatmap, top-4-feature pair plot, and two domain-flavoured scatter plots. Each one is saved as its own PNG in `reports/figures/` and summarised in `reports/metrics/eda_summary.json`.
3. **Preprocess** — a single `ColumnTransformer` handles median imputation + standard scaling for numerics and mode imputation + one-hot encoding for categoricals. Wrapped into an `imblearn` pipeline so SMOTE runs *inside* each CV fold — no leakage.
4. **Benchmark** — Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and LightGBM all compared with 5-fold stratified CV on `average_precision` (PR-AUC — the right metric when the positive rate is ~2.6%). Every run is logged to MLflow.
5. **Tune** — Optuna TPE sampler runs 30 trials on the benchmark winner with the same CV setup. Tuned model saved as `models/final_model.joblib`.
6. **Evaluate** — fits a threshold that maximises F1 on the held-out test set, then writes `test_metrics.json` plus confusion-matrix, ROC, PR, calibration, probability-distribution and permutation-importance PNGs. A Markdown `MODEL_CARD.md` is generated at the end.

---

## Quickstart

```bash
# 1. Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone + install deps
git clone https://github.com/DeryFerd/Test-Devin-Opus-4.7-ML.git
cd Test-Devin-Opus-4.7-ML
uv sync --extra dev

# 3. Run the full pipeline (ingest -> eda -> train -> tune -> evaluate)
uv run teen-mh all

# Or run stage-by-stage
uv run teen-mh ingest
uv run teen-mh eda
uv run teen-mh train
uv run teen-mh tune
uv run teen-mh evaluate

# 4. Inspect MLflow experiments
uv run mlflow ui   # http://127.0.0.1:5000
```

`Makefile` targets (`make data`, `make eda`, `make train`, `make tune`, `make evaluate`, `make all`, `make lint`, `make test`) are thin wrappers around the same commands.

---

## Dataset snapshot

| | |
|---|---|
| Rows | 1,200 |
| Columns | 13 (12 features + target) |
| Target | `depression_label` ∈ {0, 1} |
| Positive rate | **2.58%** — severely imbalanced |
| Missing values | None |
| Duplicates | 0 |
| Numerics | 9 (age, 4 continuous lifestyle, 3 ordinal 1–10, 1 GPA-ish) |
| Categoricals | 3 (`gender`, `platform_usage`, `social_interaction_level`) |

Top features by |Pearson corr with target| (numerics only):

| Feature | ρ with `depression_label` |
|---|---|
| `sleep_hours` | -0.19 |
| `daily_social_media_hours` | +0.18 |
| `stress_level` | +0.17 |
| `anxiety_level` | +0.17 |

---

## Results

Benchmark (5-fold stratified CV on training set, scoring = `average_precision`):

| Candidate | PR-AUC (mean ± std) | Fit time (s) |
|---|---|---|
| **gradient_boosting** | **1.0000 ± 0.0000** | 0.43 |
| random_forest | 0.9783 ± 0.0296 | 1.02 |
| xgboost | 0.9683 ± 0.0633 | 0.17 |
| lightgbm | 0.9420 ± 0.0718 | 0.26 |
| logreg | 0.7027 ± 0.0893 | 0.04 |

Tuned winner (`gradient_boosting`, 30 Optuna trials): best CV PR-AUC = **1.0000** at
`{n_estimators=500, learning_rate≈0.056, max_depth=3, subsample≈0.86}`.

Held-out test metrics (n = 240, 6 positives, threshold chosen to maximise F1):

| Metric | Value |
|---|---|
| ROC-AUC | **1.0000** |
| PR-AUC (average precision) | **1.0000** |
| F1 (positive class) | **1.0000** |
| Balanced accuracy | **1.0000** |

> **Caveat** — the dataset is small and appears to have near-deterministic boundaries between classes (once `stress_level`, `anxiety_level` and `addiction_level` are all high and `sleep_hours` is low, the label is essentially determined). Perfect CV and test metrics are a property of the dataset, not a license to deploy this for clinical use. See `reports/MODEL_CARD.md` for full limitations.

---

## Figure gallery

All 36 PNGs live in `reports/figures/`. A few highlights:

| | |
|---|---|
| `01_target_balance.png` | Class imbalance (~2.6% positive) |
| `02_missingness.png` | Missing-value scan (none) |
| `03_dist_<feature>.png` | Histogram + boxplot per numeric feature (9) |
| `04_cat_<feature>.png` | Count plots per categorical (3) |
| `05_box_<feature>_by_target.png` | Numeric vs. target boxplots (9) |
| `06_rate_<feature>_by_target.png` | Depression-rate per category (3) |
| `07_correlation_heatmap.png` | Numeric + target Pearson correlations |
| `08_pairplot_top_features.png` | Pair plot of top-4 |corr|-with-target features |
| `09_age_vs_screen_time.png` | Bivariate scatter coloured by target |
| `10_sleep_vs_stress.png` | Sleep vs stress scatter coloured by target |
| `11_confusion_matrix.png` | Test-set confusion matrix |
| `12_roc_curve.png` | ROC |
| `13_pr_curve.png` | Precision-recall (vs. baseline rate line) |
| `14_calibration.png` | Reliability diagram (quantile bins) |
| `15_proba_distribution.png` | Predicted-prob histogram with decision threshold |
| `16_feature_importance.png` | Permutation importance |

---

## Development

```bash
uv run ruff check src tests     # lint
uv run ruff format src tests    # format
uv run pytest                   # unit + smoke tests
uv run pre-commit install       # optional: git hooks
```

CI (`.github/workflows/ci.yml`) runs ruff + pytest on every push and PR against `main`.

### Repro knobs

Everything tunable lives in `conf/config.yaml`:

- `training.scoring` — swap `average_precision` for `roc_auc` / `f1` / `recall` / …
- `training.imbalance_strategy` — `smote` / `class_weight` / `none`
- `training.candidates` — which models enter the benchmark
- `tuning.n_trials` / `tuning.timeout_seconds` — Optuna budget
- `data.test_size` — held-out test fraction
- `seed` — global RNG seed

---

## Credits

- Dataset: [`algozee/teenager-menthal-healy`](https://www.kaggle.com/datasets/algozee/teenager-menthal-healy) on Kaggle.
- Authored by [@DeryFerd](https://github.com/DeryFerd) — built collaboratively with Devin.
