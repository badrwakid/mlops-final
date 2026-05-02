import pytest
from src.evaluation.metrics import compute_metrics


def test_compute_metrics_returns_zero_errors_and_unit_r2_for_perfect_predictions():
    metrics = compute_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)


def test_compute_metrics_returns_expected_keys():
    metrics = compute_metrics([1.0, 2.0, 3.0], [1.0, 2.5, 2.0])

    assert set(metrics) == {"rmse", "mae", "r2"}
    assert all(isinstance(value, float) for value in metrics.values())
