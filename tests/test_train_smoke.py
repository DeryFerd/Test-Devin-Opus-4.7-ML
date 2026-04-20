"""Smoke test: benchmark only cheap candidates on the toy dataset."""

from __future__ import annotations

import joblib

from teen_mh.data import persist_splits, split_dataset
from teen_mh.train import run_training


def test_run_training_smoke(tmp_cfg, toy_df):
    splits = split_dataset(toy_df, tmp_cfg)
    persist_splits(splits, tmp_cfg)

    # Keep runtime small.
    tmp_cfg.training.candidates = ["logreg"]
    tmp_cfg.training.cv_folds = 3
    tmp_cfg.mlflow.tracking_uri = f"file:{tmp_cfg.paths.mlruns}"

    summary = run_training(tmp_cfg)
    assert summary["best"]["name"] == "logreg"
    assert summary["best"]["mean_score"] >= 0.0
    model = joblib.load(tmp_cfg.paths.models / "baseline_model.joblib")
    assert hasattr(model, "predict_proba")
