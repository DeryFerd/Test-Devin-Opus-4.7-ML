"""Hyperparameter tuning with Optuna on the winning candidate from ``train.py``.

Why Optuna: modern default in 2026 — TPE sampler, pruning, parallelism, and
MLflow auto-logging all work out of the box.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, cross_val_score

from teen_mh.config import Config
from teen_mh.data import load_splits
from teen_mh.train import build_pipeline

logger = logging.getLogger(__name__)


def _load_best_candidate(cfg: Config) -> str:
    path = cfg.paths.metrics / "benchmark.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark summary not found at {path}. Run `teen-mh train` first."
        )
    with path.open() as f:
        summary = json.load(f)
    return summary["best"]["name"]


def _suggest_params(trial: optuna.Trial, name: str) -> tuple[Any, dict]:
    """Return (fresh estimator, params dict) for the named candidate."""
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression

        params = {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l2"]),
            "solver": "lbfgs",
            "max_iter": 5000,
        }
        return LogisticRegression(**params), params
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "n_jobs": -1,
        }
        return RandomForestClassifier(**params), params
    if name == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
        return GradientBoostingClassifier(**params), params
    if name == "xgboost":
        from xgboost import XGBClassifier

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "n_jobs": -1,
        }
        return XGBClassifier(**params), params
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "verbose": -1,
            "n_jobs": -1,
        }
        return LGBMClassifier(**params), params
    raise ValueError(f"Unknown candidate name: {name}")


def run_tuning(cfg: Config) -> dict:
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    best_name = _load_best_candidate(cfg)
    logger.info("Tuning candidate: %s", best_name)

    splits = load_splits(cfg)
    X_train = splits["X_train"]
    y_train = splits["y_train"][cfg.data.target]

    cv = StratifiedKFold(n_splits=cfg.training.cv_folds, shuffle=True, random_state=cfg.seed)

    def objective(trial: optuna.Trial) -> float:
        estimator, _ = _suggest_params(trial, best_name)
        pipe = build_pipeline(estimator, cfg)
        scores = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring=cfg.training.scoring, n_jobs=1
        )
        return float(np.mean(scores))

    study = optuna.create_study(
        direction=cfg.tuning.direction,
        sampler=TPESampler(seed=cfg.seed),
        study_name=f"tune_{best_name}",
    )
    with mlflow.start_run(run_name=f"tune_{best_name}"):
        study.optimize(
            objective,
            n_trials=cfg.tuning.n_trials,
            timeout=cfg.tuning.timeout_seconds,
            show_progress_bar=False,
        )
        mlflow.log_params({"tuned_candidate": best_name, **study.best_params})
        mlflow.log_metric(f"tuned_{cfg.training.scoring}", study.best_value)

    logger.info(
        "Best trial for %s: value=%.4f params=%s",
        best_name,
        study.best_value,
        study.best_params,
    )

    # Refit best model on full training set and persist as the final model.
    best_estimator, _ = _suggest_params(optuna.trial.FixedTrial(study.best_params), best_name)
    final_pipe = build_pipeline(best_estimator, cfg)
    final_pipe.fit(X_train, y_train)

    model_path = cfg.paths.models / "final_model.joblib"
    joblib.dump(final_pipe, model_path)
    logger.info("Persisted tuned model to %s", model_path)

    summary = {
        "candidate": best_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials_run": len(study.trials),
        "model_path": str(model_path),
    }
    _write_summary(summary, cfg.paths.metrics / "tuning.json")
    return summary


def _write_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(summary, f, indent=2)
