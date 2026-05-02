from src.config import load_config


def test_load_config_returns_expected_keys():
    cfg = load_config()
    assert cfg.data.target == "cnt"
    assert cfg.training.n_trials > 0
    assert cfg.serving.api_port == 8000
    assert 0 < cfg.validation.min_test_r2 < 1
    assert cfg.validation.rmse_threshold > 0
