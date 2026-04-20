"""Benchmark multiple candidate models with stratified CV + MLflow tracking.

The winner (by ``training.scoring`` metric) is persisted as the baseline model
and is the default starting point for hyperparameter tuning in ``tune.py``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from teen_mh.config import Config
from teen_mh.data import load_splits
from teen_mh.features import build_preprocessor

logger = logging.getLogger(__name__)


@dataclass
class CandidateResult:
    name: str
    mean_score: float
    std_score: float
    fit_time_s: float
    per_fold: list[float]


def _base_estimators(seed: int) -> dict[str, Any]:
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "logreg": LogisticRegression(max_iter=2000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        ),
    }


def build_pipeline(estimator, cfg: Config):
    preprocessor = build_preprocessor(cfg)
    strategy = cfg.training.imbalance_strategy
    if strategy == "smote":
        return ImbPipeline(
            steps=[
                ("preprocess", preprocessor),
                ("smote", SMOTE(random_state=cfg.seed, k_neighbors=3)),
                ("model", estimator),
            ]
        )
    if strategy == "class_weight":
        if hasattr(estimator, "class_weight"):
            estimator.set_params(class_weight="balanced")
        elif hasattr(estimator, "scale_pos_weight"):
            estimator.set_params(scale_pos_weight=10)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def _setup_mlflow(cfg: Config) -> None:
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)


def run_training(cfg: Config) -> dict:
    _setup_mlflow(cfg)
    splits = load_splits(cfg)
    X_train: pd.DataFrame = splits["X_train"]
    y_train: pd.Series = splits["y_train"][cfg.data.target]

    cv = StratifiedKFold(n_splits=cfg.training.cv_folds, shuffle=True, random_state=cfg.seed)
    estimators = _base_estimators(cfg.seed)
    results: list[CandidateResult] = []

    with mlflow.start_run(run_name="benchmark"):
        mlflow.log_params(
            {
                "cv_folds": cfg.training.cv_folds,
                "scoring": cfg.training.scoring,
                "imbalance_strategy": cfg.training.imbalance_strategy,
                "seed": cfg.seed,
            }
        )
        for name in cfg.training.candidates:
            if name not in estimators:
                logger.warning("Unknown candidate %s — skipping", name)
                continue
            logger.info("Training candidate: %s", name)
            pipe = build_pipeline(estimators[name], cfg)
            cv_out = cross_validate(
                pipe,
                X_train,
                y_train,
                cv=cv,
                scoring=cfg.training.scoring,
                n_jobs=1,
                return_train_score=False,
            )
            res = CandidateResult(
                name=name,
                mean_score=float(np.mean(cv_out["test_score"])),
                std_score=float(np.std(cv_out["test_score"])),
                fit_time_s=float(np.mean(cv_out["fit_time"])),
                per_fold=[float(v) for v in cv_out["test_score"]],
            )
            results.append(res)
            mlflow.log_metrics(
                {
                    f"{name}__{cfg.training.scoring}_mean": res.mean_score,
                    f"{name}__{cfg.training.scoring}_std": res.std_score,
                    f"{name}__fit_time_s": res.fit_time_s,
                }
            )

    results.sort(key=lambda r: r.mean_score, reverse=True)
    best = results[0]
    logger.info(
        "Best candidate: %s (score=%.4f ± %.4f)", best.name, best.mean_score, best.std_score
    )

    # Fit best candidate on the full training set and persist as baseline model.
    best_pipe = build_pipeline(_base_estimators(cfg.seed)[best.name], cfg)
    best_pipe.fit(X_train, y_train)

    models_dir = cfg.paths.models
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "baseline_model.joblib"
    joblib.dump(best_pipe, model_path)
    logger.info("Persisted baseline model to %s", model_path)

    # Write benchmark summary.
    summary = {
        "scoring": cfg.training.scoring,
        "imbalance_strategy": cfg.training.imbalance_strategy,
        "candidates": [r.__dict__ for r in results],
        "best": best.__dict__,
        "model_path": str(model_path),
    }
    _write_summary(summary, cfg.paths.metrics / "benchmark.json")
    return summary


def _write_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(summary, f, indent=2)
