"""Deep-dive EDA.

Every chart is rendered and saved as an individual PNG under
``reports/figures/``. A summary JSON with descriptive stats, missingness, class
balance and correlations is also emitted so downstream consumers (e.g. the
README figure gallery) don't have to re-derive them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from teen_mh.config import Config
from teen_mh.data import load_raw

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="talk")
PALETTE = sns.color_palette("viridis", as_cmap=False)
POS_NEG_PALETTE = {0: "#4C72B0", 1: "#C44E52"}


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def run_eda(cfg: Config) -> dict:
    """Execute the full EDA sweep. Returns a dict of summary stats for logging."""
    df = load_raw(cfg)
    target = cfg.data.target
    fig_dir = cfg.paths.figures
    metrics_dir = cfg.paths.metrics
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": df.isna().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "target_distribution": df[target].value_counts().to_dict(),
        "positive_rate": float(df[target].mean()),
        "numeric_describe": df.describe().to_dict(),
    }

    _plot_target_balance(df, target, fig_dir)
    _plot_missingness(df, fig_dir)
    _plot_numeric_distributions(df, cfg, fig_dir)
    _plot_categorical_counts(df, cfg, fig_dir)
    _plot_numeric_boxplots_by_target(df, cfg, target, fig_dir)
    _plot_categorical_rates_by_target(df, cfg, target, fig_dir)
    corr = _plot_correlation_heatmap(df, cfg, fig_dir)
    summary["correlation_with_target"] = corr
    _plot_pairwise_top_features(df, cfg, target, corr, fig_dir)
    _plot_age_vs_screen_time(df, target, fig_dir)
    _plot_sleep_vs_stress(df, target, fig_dir)

    with (metrics_dir / "eda_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    logger.info("EDA summary written to %s", metrics_dir / "eda_summary.json")
    return summary


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    raise TypeError(f"Not serializable: {type(o)}")


def _plot_target_balance(df: pd.DataFrame, target: str, fig_dir: Path) -> None:
    counts = df[target].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [f"Not depressed ({counts.index[0]})", f"Depressed ({counts.index[-1]})"],
        counts.values,
        color=[POS_NEG_PALETTE[0], POS_NEG_PALETTE[1]],
    )
    for bar, value in zip(bars, counts.values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value}\n({value / counts.sum():.1%})",
            ha="center",
            va="bottom",
            fontsize=12,
        )
    ax.set_title("Target class balance — severe imbalance")
    ax.set_ylabel("Count")
    ax.set_ylim(0, counts.max() * 1.15)
    _save(fig, fig_dir, "01_target_balance")


def _plot_missingness(df: pd.DataFrame, fig_dir: Path) -> None:
    missing = df.isna().mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(missing.index, missing.values * 100, color="#55A868")
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing values per column")
    ax.set_xlim(0, max(5, missing.max() * 100 + 1))
    _save(fig, fig_dir, "02_missingness")


def _plot_numeric_distributions(df: pd.DataFrame, cfg: Config, fig_dir: Path) -> None:
    numeric_cols = cfg.features.numeric
    for col in numeric_cols:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df[col], kde=True, ax=axes[0], color="#4C72B0")
        axes[0].set_title(f"Distribution — {col}")
        sns.boxplot(x=df[col], ax=axes[1], color="#4C72B0")
        axes[1].set_title(f"Boxplot — {col}")
        _save(fig, fig_dir, f"03_dist_{col}")


def _plot_categorical_counts(df: pd.DataFrame, cfg: Config, fig_dir: Path) -> None:
    for col in cfg.features.categorical:
        fig, ax = plt.subplots(figsize=(8, 5))
        order = df[col].value_counts().index
        sns.countplot(data=df, x=col, order=order, ax=ax, palette="viridis", hue=col, legend=False)
        ax.set_title(f"Counts — {col}")
        ax.set_ylabel("Count")
        _save(fig, fig_dir, f"04_cat_{col}")


def _plot_numeric_boxplots_by_target(
    df: pd.DataFrame, cfg: Config, target: str, fig_dir: Path
) -> None:
    for col in cfg.features.numeric:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            data=df,
            x=target,
            y=col,
            ax=ax,
            palette=POS_NEG_PALETTE,
            hue=target,
            legend=False,
        )
        ax.set_title(f"{col} by {target}")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Not depressed", "Depressed"])
        _save(fig, fig_dir, f"05_box_{col}_by_target")


def _plot_categorical_rates_by_target(
    df: pd.DataFrame, cfg: Config, target: str, fig_dir: Path
) -> None:
    for col in cfg.features.categorical:
        rates = df.groupby(col)[target].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=rates.index, y=rates.values * 100, ax=ax, color="#C44E52")
        ax.set_title(f"Depression rate (%) by {col}")
        ax.set_ylabel("Depression rate (%)")
        for i, v in enumerate(rates.values):
            ax.text(i, v * 100, f"{v:.1%}", ha="center", va="bottom", fontsize=11)
        _save(fig, fig_dir, f"06_rate_{col}_by_target")


def _plot_correlation_heatmap(df: pd.DataFrame, cfg: Config, fig_dir: Path) -> dict:
    numeric = df[[*cfg.features.numeric, cfg.data.target]]
    corr = numeric.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax, square=True)
    ax.set_title("Pearson correlation — numeric features + target")
    _save(fig, fig_dir, "07_correlation_heatmap")
    return corr[cfg.data.target].drop(cfg.data.target).to_dict()


def _plot_pairwise_top_features(
    df: pd.DataFrame, cfg: Config, target: str, corr_with_target: dict, fig_dir: Path
) -> None:
    top = sorted(corr_with_target.items(), key=lambda kv: abs(kv[1]), reverse=True)[:4]
    cols = [name for name, _ in top]
    if len(cols) < 2:
        return
    pair_df = df[[*cols, target]].copy()
    pair_df[target] = pair_df[target].map({0: "Not depressed", 1: "Depressed"})
    grid = sns.pairplot(
        pair_df,
        hue=target,
        palette={"Not depressed": POS_NEG_PALETTE[0], "Depressed": POS_NEG_PALETTE[1]},
        diag_kind="kde",
        corner=True,
        plot_kws={"alpha": 0.55, "s": 30},
        height=2.4,
    )
    grid.figure.suptitle(f"Pair plot — top {len(cols)} features by |corr with {target}|", y=1.02)
    path = fig_dir / "08_pairplot_top_features.png"
    grid.figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(grid.figure)
    logger.info("Saved %s", path)


def _plot_age_vs_screen_time(df: pd.DataFrame, target: str, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="age",
        y="screen_time_before_sleep",
        hue=target,
        palette=POS_NEG_PALETTE,
        alpha=0.7,
        ax=ax,
    )
    ax.set_title("Age vs. screen time before sleep, colored by depression")
    _save(fig, fig_dir, "09_age_vs_screen_time")


def _plot_sleep_vs_stress(df: pd.DataFrame, target: str, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="sleep_hours",
        y="stress_level",
        hue=target,
        palette=POS_NEG_PALETTE,
        alpha=0.7,
        ax=ax,
    )
    ax.set_title("Sleep hours vs. stress level, colored by depression")
    _save(fig, fig_dir, "10_sleep_vs_stress")
