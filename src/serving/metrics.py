from prometheus_client import Counter, Gauge, Histogram

PREDICTION_CONFIDENCE = Histogram(
    "bike_prediction_confidence",
    "Histogram of per-prediction confidence scores in [0,1]",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
FEATURE_TEMP = Histogram(
    "bike_feature_temp",
    "Histogram of normalized temperature input feature",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
FEATURE_HR = Histogram(
    "bike_feature_hr",
    "Histogram of hour-of-day input feature",
    buckets=tuple(range(0, 25, 3)),
)
MODEL_VERSION = Gauge(
    "bike_model_version_info",
    "Currently loaded model version label",
    ["version"],
)
INFERENCE_COUNT = Counter(
    "bike_inference_total",
    "Inference count by endpoint, model version, and predicted output class "
    "(regression: binned demand level; maps rubric 'inference count by class')",
    ["endpoint", "model_version", "output_class"],
)
PREDICTION_LATENCY = Histogram(
    "bike_prediction_latency_seconds",
    "End-to-end prediction latency in seconds",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
PREDICTION_VALUE = Histogram(
    "bike_prediction_value",
    "Histogram of predicted rental counts",
    buckets=(0, 25, 50, 100, 200, 400, 800, 1600),
)
DRIFT_PSI = Gauge(
    "bike_feature_drift_psi",
    "Per-feature drift proxy from latest batch Evidently run (1.0 drift, 0.0 no drift)",
    ["feature"],
)
