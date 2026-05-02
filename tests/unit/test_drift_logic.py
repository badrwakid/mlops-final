from monitoring.drift_logic import drift_alert


def test_no_alert_when_below_threshold():
    result = drift_alert(
        drift_per_feature={"a": False, "b": False, "c": True, "d": False},
        threshold_share=0.50,
    )
    assert result.alert is False
    assert result.drift_share == 0.25
    assert result.drifted_features == ["c"]


def test_alert_when_above_threshold():
    result = drift_alert(
        drift_per_feature={"a": True, "b": True, "c": True, "d": False},
        threshold_share=0.50,
    )
    assert result.alert is True
    assert result.drift_share == 0.75
    assert set(result.drifted_features) == {"a", "b", "c"}


def test_handles_empty_dict():
    result = drift_alert(drift_per_feature={}, threshold_share=0.20)
    assert result.alert is False
    assert result.drift_share == 0.0
