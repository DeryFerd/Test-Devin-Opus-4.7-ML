"""Data ingestion: pull dataset from Kaggle and materialise train/test splits.

The Kaggle dataset endpoint is public, so we don't need credentials for this
specific slug — a plain GET against ``/api/v1/datasets/download/<slug>`` is
enough. If the endpoint ever starts requiring auth, set ``KAGGLE_USERNAME`` and
``KAGGLE_KEY`` env vars and we'll fall back to the kaggle CLI.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from teen_mh.config import Config

logger = logging.getLogger(__name__)

KAGGLE_DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/{slug}"


def download_dataset(cfg: Config, force: bool = False) -> Path:
    """Download the raw CSV into ``data/raw/`` and return its path."""
    raw_dir = cfg.paths.data_raw
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / cfg.data.csv_filename

    if csv_path.exists() and not force:
        logger.info("Raw dataset already present at %s — skipping download.", csv_path)
        return csv_path

    slug = cfg.data.kaggle_slug
    url = KAGGLE_DOWNLOAD_URL.format(slug=slug)

    # Try anonymous HTTP first — the dataset publisher made this endpoint public.
    try:
        import urllib.request

        logger.info("Downloading %s via anonymous HTTPS …", slug)
        with urllib.request.urlopen(url, timeout=60) as resp:
            blob = resp.read()
        _extract_zip(blob, raw_dir)
    except Exception as exc:  # fall back to kaggle CLI
        logger.warning("Anonymous download failed (%s). Falling back to kaggle CLI.", exc)
        _download_via_kaggle_cli(slug, raw_dir)

    if not csv_path.exists():
        # Some Kaggle dumps unzip into nested folders; surface the real file.
        matches = list(raw_dir.rglob(cfg.data.csv_filename))
        if not matches:
            raise FileNotFoundError(
                f"Expected {cfg.data.csv_filename} under {raw_dir} after download."
            )
        shutil.move(str(matches[0]), csv_path)

    logger.info("Dataset ready at %s", csv_path)
    return csv_path


def _extract_zip(blob: bytes, dest: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(dest)


def _download_via_kaggle_cli(slug: str, dest: Path) -> None:
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        raise RuntimeError(
            "Anonymous download failed and KAGGLE_USERNAME/KAGGLE_KEY are not set. "
            "Export them or place kaggle.json in ~/.kaggle/ and retry."
        )
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"],
        check=True,
    )


def load_raw(cfg: Config) -> pd.DataFrame:
    csv_path = cfg.paths.data_raw / cfg.data.csv_filename
    if not csv_path.exists():
        csv_path = download_dataset(cfg)
    return pd.read_csv(csv_path)


def split_dataset(df: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    """Stratified train/test split.

    The target is highly imbalanced (~2.6% positive), so stratification is
    critical to keep both folds representative.
    """
    target = cfg.data.target
    y = df[target]
    X = df.drop(columns=[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.data.test_size,
        random_state=cfg.seed,
        stratify=y,
    )
    # Store y as single-column DataFrames so we can round-trip through Parquet.
    return {
        "X_train": X_train.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True).to_frame(name=target),
        "y_test": y_test.reset_index(drop=True).to_frame(name=target),
    }


def persist_splits(splits: dict[str, pd.DataFrame], cfg: Config) -> None:
    out = cfg.paths.data_processed
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_parquet(out / f"{name}.parquet", index=False)
    logger.info("Persisted splits to %s", out)


def load_splits(cfg: Config) -> dict[str, pd.DataFrame]:
    out = cfg.paths.data_processed
    required = ["X_train", "X_test", "y_train", "y_test"]
    if not all((out / f"{n}.parquet").exists() for n in required):
        df = load_raw(cfg)
        splits = split_dataset(df, cfg)
        persist_splits(splits, cfg)
        return splits
    return {n: pd.read_parquet(out / f"{n}.parquet") for n in required}
