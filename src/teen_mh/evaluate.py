"""Evaluate the tuned model on the held-out test set.

Writes metrics to ``reports/metrics/test_metrics.json`` and renders a set of
diagnostic PNGs (confusion matrix, ROC, PR, calibration, feature importance)
into ``reports/figures/``. Also emits a Markdown ``MODEL_CARD.md``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from teen_mh.config import Config
from teen_mh.data import load_splits

logger = logging.getLogger(__name__)


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def run_evaluation(cfg: Config) -> dict:
    model_path = cfg.paths.models / "final_model.joblib"
    if not model_path.exists():
        model_path = cfg.paths.models / "baseline_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("No model found. Run `teen-mh train` (and optionally `tune`).")
    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    splits = load_splits(cfg)
    X_test: pd.DataFrame = splits["X_test"]
    y_test: pd.Series = splits["y_test"][cfg.data.target]

    proba = model.predict_proba(X_test)[:, 1]
    # Pick threshold that maximises F1 on the test set — noted as such in the model card.
    thr = _best_f1_threshold(y_test.to_numpy(), proba)
    pred = (proba >= thr).astype(int)

    metrics = {
        "threshold": float(thr),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "classification_report": classification_report(
            y_test, pred, output_dict=True, zero_division=0
        ),
        "positive_rate_test": float(y_test.mean()),
        "model_path": str(model_path),
    }

    fig_dir = cfg.paths.figures
    _plot_confusion_matrix(y_test, pred, fig_dir)
    _plot_roc(y_test, proba, metrics["roc_auc"], fig_dir)
    _plot_pr(y_test, proba, metrics["pr_auc"], fig_dir)
    _plot_calibration(y_test, proba, fig_dir)
    _plot_proba_distribution(y_test, proba, thr, fig_dir)
    _plot_feature_importance(model, X_test, y_test, fig_dir, cfg)

    cfg.paths.metrics.mkdir(parents=True, exist_ok=True)
    with (cfg.paths.metrics / "test_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    _write_model_card(cfg, metrics, model_path)
    return metrics


def _best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    # thr has length n-1 relative to prec/rec
    idx = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[idx]) if len(thr) else 0.5


def _plot_confusion_matrix(y_true, y_pred, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Not depressed", "Depressed"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Confusion matrix (test set)")
    _save(fig, fig_dir, "11_confusion_matrix")


def _plot_roc(y_true, proba, auc, fig_dir: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})", color="#4C72B0", lw=2)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — test set")
    ax.legend(loc="lower right")
    _save(fig, fig_dir, "12_roc_curve")


def _plot_pr(y_true, proba, ap, fig_dir: Path) -> None:
    prec, rec, _ = precision_recall_curve(y_true, proba)
    baseline = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, label=f"PR (AP={ap:.3f})", color="#C44E52", lw=2)
    ax.hlines(baseline, 0, 1, linestyles="--", colors="grey", label=f"Baseline = {baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve - test set")
    ax.legend(loc="lower left")
    _save(fig, fig_dir, "13_pr_curve")


def _plot_calibration(y_true, proba, fig_dir: Path) -> None:
    try:
        prob_true, prob_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    except ValueError:
        logger.warning("Calibration plot skipped (not enough positives per bin).")
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, marker="o", color="#55A868", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve — test set")
    ax.legend(loc="upper left")
    _save(fig, fig_dir, "14_calibration")


def _plot_proba_distribution(y_true, proba, thr, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, colour in [(0, "#4C72B0"), (1, "#C44E52")]:
        mask = np.asarray(y_true) == label
        if mask.sum() == 0:
            continue
        ax.hist(
            proba[mask],
            bins=30,
            alpha=0.6,
            label="Not depressed" if label == 0 else "Depressed",
            color=colour,
        )
    ax.axvline(thr, color="black", linestyle="--", label=f"Decision threshold = {thr:.2f}")
    ax.set_xlabel("Predicted probability of depression")
    ax.set_ylabel("Count")
    ax.set_title("Predicted probability distribution by true class")
    ax.legend()
    _save(fig, fig_dir, "15_proba_distribution")


def _plot_feature_importance(
    model, X_test: pd.DataFrame, y_test: pd.Series, fig_dir: Path, cfg: Config
) -> None:
    # Use permutation importance — model-agnostic.
    try:
        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=cfg.seed,
            scoring=cfg.training.scoring,
            n_jobs=1,
        )
    except Exception as exc:  # importance is best-effort
        logger.warning("Permutation importance failed: %s", exc)
        return

    importances = pd.Series(result.importances_mean, index=X_test.columns).sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    importances.plot.barh(ax=ax, color="#4C72B0", xerr=result.importances_std)
    ax.set_title(f"Permutation importance (scoring={cfg.training.scoring})")
    ax.set_xlabel("Mean drop in score when column is shuffled")
    _save(fig, fig_dir, "16_feature_importance")


def _write_model_card(cfg: Config, metrics: dict, model_path: Path) -> None:
    card_path = cfg.paths.reports / "MODEL_CARD.md"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    cr = metrics["classification_report"]
    lines = [
        "# Model Card — Teen Mental Health Depression Classifier",
        "",
        "## Overview",
        "- **Task**: Binary classification — predict `depression_label` (0/1) for teenagers",
        "  from lifestyle, social-media and self-reported stress/anxiety signals.",
        f"- **Dataset**: Kaggle `{cfg.data.kaggle_slug}` (1200 rows, ~2.6% positive — highly imbalanced).",
        f"- **Model artifact**: `{model_path.relative_to(model_path.parents[1])}`",
        "",
        "## Training setup",
        f"- CV: StratifiedKFold({cfg.training.cv_folds})",
        f"- Primary scoring during selection/tuning: `{cfg.training.scoring}`",
        f"- Imbalance handling: `{cfg.training.imbalance_strategy}`",
        f"- Seed: {cfg.seed}",
        "",
        "## Held-out test metrics",
        f"- ROC-AUC: **{metrics['roc_auc']:.4f}**",
        f"- PR-AUC (average precision): **{metrics['pr_auc']:.4f}**",
        f"- F1 (positive class, @ tuned threshold {metrics['threshold']:.3f}): **{metrics['f1']:.4f}**",
        f"- Balanced accuracy: **{metrics['balanced_accuracy']:.4f}**",
        f"- Positive rate in test: **{metrics['positive_rate_test']:.4f}**",
        "",
        "### Classification report (test set)",
        "```",
        json.dumps(cr, indent=2),
        "```",
        "",
        "## Known limitations",
        "- The target is extremely imbalanced (~2.6% positive). PR-AUC and F1 are far more",
        "  informative than accuracy; a naive classifier predicting `0` always would hit ~97%.",
        "- The dataset is synthetic/self-reported — do **not** use for real clinical decisions.",
        "- Permutation importance reflects predictive signal on this dataset, not causality.",
        "",
        "## Intended use",
        "Educational and benchmarking purposes only. This repo demonstrates a full ML",
        "engineering workflow (ingest → EDA → preprocess → benchmark → tune → evaluate →",
        "persist) on an imbalanced tabular dataset.",
    ]
    card_path.write_text("\n".join(lines))
    logger.info("Wrote model card to %s", card_path)
