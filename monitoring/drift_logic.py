from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DriftResult:
    alert: bool
    drift_share: float
    drifted_features: list[str]


def drift_alert(drift_per_feature: dict[str, bool], threshold_share: float) -> DriftResult:
    if not drift_per_feature:
        return DriftResult(alert=False, drift_share=0.0, drifted_features=[])
    drifted = [f for f, d in drift_per_feature.items() if d]
    share = len(drifted) / len(drift_per_feature)
    return DriftResult(alert=share > threshold_share, drift_share=share, drifted_features=drifted)
