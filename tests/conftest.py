from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from teen_mh.config import Config, load_config


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load_config()


@pytest.fixture
def tmp_cfg(tmp_path: Path) -> Config:
    cfg = load_config()
    cfg.paths.data_raw = tmp_path / "data" / "raw"
    cfg.paths.data_processed = tmp_path / "data" / "processed"
    cfg.paths.models = tmp_path / "models"
    cfg.paths.reports = tmp_path / "reports"
    cfg.paths.figures = tmp_path / "reports" / "figures"
    cfg.paths.metrics = tmp_path / "reports" / "metrics"
    cfg.paths.mlruns = tmp_path / "mlruns"
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def toy_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "age": rng.integers(12, 20, size=n),
            "gender": rng.choice(["male", "female"], size=n),
            "daily_social_media_hours": rng.uniform(0, 8, size=n),
            "platform_usage": rng.choice(["Instagram", "TikTok", "Both"], size=n),
            "sleep_hours": rng.uniform(4, 10, size=n),
            "screen_time_before_sleep": rng.uniform(0, 3, size=n),
            "academic_performance": rng.uniform(1, 4, size=n),
            "physical_activity": rng.uniform(0, 3, size=n),
            "social_interaction_level": rng.choice(["low", "medium", "high"], size=n),
            "stress_level": rng.integers(0, 11, size=n),
            "anxiety_level": rng.integers(0, 11, size=n),
            "addiction_level": rng.integers(0, 11, size=n),
        }
    )
    # Inject a weak signal so stratified split and SMOTE have work to do.
    score = df["stress_level"] + df["anxiety_level"] + df["addiction_level"] - df["sleep_hours"]
    df["depression_label"] = (score > np.quantile(score, 0.92)).astype(int)
    return df
