"""Typed config loader.

All tunable knobs live in ``conf/config.yaml``. This module validates them with
pydantic so downstream code can rely on correct types instead of dict lookups.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "conf" / "config.yaml"


class Paths(BaseModel):
    data_raw: Path
    data_processed: Path
    models: Path
    reports: Path
    figures: Path
    metrics: Path
    mlruns: Path


class DataCfg(BaseModel):
    kaggle_slug: str
    csv_filename: str
    target: str
    test_size: float = Field(gt=0, lt=1)
    val_size: float = Field(gt=0, lt=1)


class FeaturesCfg(BaseModel):
    numeric: list[str]
    categorical: list[str]


class TrainingCfg(BaseModel):
    cv_folds: int = Field(ge=2)
    scoring: str
    imbalance_strategy: str
    candidates: list[str]


class TuningCfg(BaseModel):
    n_trials: int = Field(ge=1)
    timeout_seconds: int = Field(ge=1)
    direction: str


class MLflowCfg(BaseModel):
    experiment_name: str
    tracking_uri: str


class Config(BaseModel):
    seed: int
    paths: Paths
    data: DataCfg
    features: FeaturesCfg
    training: TrainingCfg
    tuning: TuningCfg
    mlflow: MLflowCfg

    def resolve_paths(self, root: Path | None = None) -> None:
        root = root or REPO_ROOT
        for name in Paths.model_fields:
            value = getattr(self.paths, name)
            if not value.is_absolute():
                setattr(self.paths, name, (root / value).resolve())

    def ensure_dirs(self) -> None:
        for name in Paths.model_fields:
            getattr(self.paths, name).mkdir(parents=True, exist_ok=True)


def load_config(path: Path | str | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with path.open("r") as f:
        raw = yaml.safe_load(f)
    cfg = Config.model_validate(raw)
    cfg.resolve_paths()
    return cfg
