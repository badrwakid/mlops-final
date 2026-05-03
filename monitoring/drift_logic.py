from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DriftResult:
    alert: bool
    drift_share: float
    drifted_features: list[str]


def drift_alert(
    drift_per_feature: dict[str, bool],
    threshold_share: float,
    exclude_cols: set[str] | None = None,
    feature_weights: dict[str, float] | None = None,
) -> DriftResult:
    if not drift_per_feature:
        return DriftResult(alert=False, drift_share=0.0, drifted_features=[])
    excluded = exclude_cols or set()
    scoped = {k: v for k, v in drift_per_feature.items() if k not in excluded}
    if not scoped:
        return DriftResult(alert=False, drift_share=0.0, drifted_features=[])
    drifted = [f for f, d in scoped.items() if d]
    if feature_weights:
        total_weight = sum(feature_weights.get(feature, 1.0) for feature in scoped)
        drift_weight = sum(feature_weights.get(feature, 1.0) for feature in drifted)
        share = (drift_weight / total_weight) if total_weight else 0.0
    else:
        share = len(drifted) / len(scoped)
    return DriftResult(alert=share > threshold_share, drift_share=share, drifted_features=drifted)
