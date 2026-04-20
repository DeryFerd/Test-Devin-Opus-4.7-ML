"""Feature preprocessing pipeline.

Built as a ``ColumnTransformer`` so the whole fit/transform logic ships with the
model via joblib — no leakage, no manual re-implementation at inference time.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from teen_mh.config import Config


def build_preprocessor(cfg: Config) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, cfg.features.numeric),
            ("cat", categorical_pipeline, cfg.features.categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def feature_names_after_transform(preprocessor: ColumnTransformer) -> list[str]:
    """Return expanded feature names (post one-hot)."""
    return list(preprocessor.get_feature_names_out())
