import numpy as np

from teen_mh.features import build_preprocessor, feature_names_after_transform


def test_preprocessor_transforms_shape(tmp_cfg, toy_df):
    pre = build_preprocessor(tmp_cfg)
    X = toy_df.drop(columns=[tmp_cfg.data.target])
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == len(X)
    # One-hot expands categorical columns.
    assert Xt.shape[1] > len(tmp_cfg.features.numeric) + len(tmp_cfg.features.categorical)
    assert not np.isnan(Xt).any()


def test_feature_names_match_output(tmp_cfg, toy_df):
    pre = build_preprocessor(tmp_cfg)
    X = toy_df.drop(columns=[tmp_cfg.data.target])
    Xt = pre.fit_transform(X)
    names = feature_names_after_transform(pre)
    assert len(names) == Xt.shape[1]
