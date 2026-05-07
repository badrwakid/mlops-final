from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "params.yaml"


class PathsCfg(BaseModel):
    raw_csv: str
    processed: str
    train: str
    test: str
    reference: str
    production: str
    preprocessor: str
    model: str
    metrics: str


class DataCfg(BaseModel):
    target: str
    drop_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    split_column: str
    test_size: float
    reference_holdout: float
    random_state: int


class PreprocessingCfg(BaseModel):
    numeric_imputer_strategy: str
    categorical_imputer_strategy: str
    feature_selection_k: int
    cyclical_hr_mnth: bool = True
    feature_scores_json: str | None = None


class TrainingCfg(BaseModel):
    model_type: str
    n_trials: int
    cv_folds: int
    hpo_search_space: dict


class DriftCfg(BaseModel):
    perturb_temp_factor: float
    perturb_hum_factor: float
    perturb_windspeed_noise_std: float
    drift_threshold_share: float
    perturbed_features: list[str]
    generate_synthetic_demo_report: bool = True
    synthetic_demo_report_name: str = "drift_synthetic_demo.html"


class ServingCfg(BaseModel):
    model_name: str
    model_stage: str
    api_host: str
    api_port: int


class MLflowCfg(BaseModel):
    tracking_uri: str
    experiment_name: str
    registered_model_name: str


class ValidationCfg(BaseModel):
    min_test_r2: float
    rmse_threshold: float


class Config(BaseModel):
    paths: PathsCfg
    data: DataCfg
    preprocessing: PreprocessingCfg
    training: TrainingCfg
    drift: DriftCfg
    serving: ServingCfg
    mlflow: MLflowCfg
    validation: ValidationCfg


def load_config(path: Path | str = CONFIG_PATH) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
