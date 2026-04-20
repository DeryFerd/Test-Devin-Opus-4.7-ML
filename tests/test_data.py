from teen_mh.data import persist_splits, split_dataset


def test_split_is_stratified(tmp_cfg, toy_df):
    splits = split_dataset(toy_df, tmp_cfg)
    target = tmp_cfg.data.target
    train_rate = splits["y_train"][target].mean()
    test_rate = splits["y_test"][target].mean()
    # Stratification should keep positive rate within ~30% of each other.
    assert abs(train_rate - test_rate) / max(train_rate, 1e-9) < 0.5
    assert len(splits["X_train"]) + len(splits["X_test"]) == len(toy_df)
    assert target not in splits["X_train"].columns


def test_persist_splits_round_trip(tmp_cfg, toy_df):
    splits = split_dataset(toy_df, tmp_cfg)
    persist_splits(splits, tmp_cfg)
    for name in ("X_train", "X_test", "y_train", "y_test"):
        path = tmp_cfg.paths.data_processed / f"{name}.parquet"
        assert path.exists(), f"{path} not written"
