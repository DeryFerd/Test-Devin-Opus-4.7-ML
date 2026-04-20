from teen_mh.config import Paths, load_config


def test_config_loads() -> None:
    cfg = load_config()
    assert cfg.data.target == "depression_label"
    assert cfg.training.cv_folds >= 2
    assert 0 < cfg.data.test_size < 1
    assert "gender" in cfg.features.categorical
    assert "age" in cfg.features.numeric


def test_config_paths_are_absolute() -> None:
    cfg = load_config()
    for name in Paths.model_fields:
        path = getattr(cfg.paths, name)
        assert path.is_absolute(), f"{name} should be absolute, got {path}"
