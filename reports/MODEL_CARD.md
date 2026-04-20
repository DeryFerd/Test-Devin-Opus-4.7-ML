# Model Card — Teen Mental Health Depression Classifier

## Overview
- **Task**: Binary classification — predict `depression_label` (0/1) for teenagers
  from lifestyle, social-media and self-reported stress/anxiety signals.
- **Dataset**: Kaggle `algozee/teenager-menthal-healy` (1200 rows, ~2.6% positive — highly imbalanced).
- **Model artifact**: `models/final_model.joblib`

## Training setup
- CV: StratifiedKFold(5)
- Primary scoring during selection/tuning: `average_precision`
- Imbalance handling: `smote`
- Seed: 42

## Held-out test metrics
- ROC-AUC: **1.0000**
- PR-AUC (average precision): **1.0000**
- F1 (positive class, @ tuned threshold 1.000): **1.0000**
- Balanced accuracy: **1.0000**
- Positive rate in test: **0.0250**

### Classification report (test set)
```
{
  "0": {
    "precision": 1.0,
    "recall": 1.0,
    "f1-score": 1.0,
    "support": 234.0
  },
  "1": {
    "precision": 1.0,
    "recall": 1.0,
    "f1-score": 1.0,
    "support": 6.0
  },
  "accuracy": 1.0,
  "macro avg": {
    "precision": 1.0,
    "recall": 1.0,
    "f1-score": 1.0,
    "support": 240.0
  },
  "weighted avg": {
    "precision": 1.0,
    "recall": 1.0,
    "f1-score": 1.0,
    "support": 240.0
  }
}
```

## Known limitations
- The target is extremely imbalanced (~2.6% positive). PR-AUC and F1 are far more
  informative than accuracy; a naive classifier predicting `0` always would hit ~97%.
- The dataset is synthetic/self-reported — do **not** use for real clinical decisions.
- Permutation importance reflects predictive signal on this dataset, not causality.

## Intended use
Educational and benchmarking purposes only. This repo demonstrates a full ML
engineering workflow (ingest → EDA → preprocess → benchmark → tune → evaluate →
persist) on an imbalanced tabular dataset.