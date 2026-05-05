import pandas as pd
import pytest

import scripts.export_runs as export_runs_module
from scripts.compute_subgroup_metrics import build_subgroup_payload
from scripts.export_runs import export_runs, write_experiment_log


def test_build_subgroup_payload_includes_overall_and_filters_small_groups():
    df = pd.DataFrame({
        "season": [1] * 35 + [2] * 25,
        "weathersit": [1] * 60,
        "workingday": [0] * 30 + [1] * 30,
    })
    y_true = [10.0] * 60
    y_pred = [12.0] * 35 + [8.0] * 25

    payload = build_subgroup_payload(
        df,
        y_true,
        y_pred,
        group_fields=["season"],
        min_n=30,
    )

    assert set(payload) == {"overall", "subgroups"}
    assert set(payload["overall"]) == {"rmse", "mae", "r2"}
    assert list(payload["subgroups"]) == ["season"]
    assert len(payload["subgroups"]["season"]) == 1
    season_group = payload["subgroups"]["season"][0]
    assert season_group["value"] == 1
    assert season_group["n"] == 35
    assert set(season_group) == {"value", "n", "rmse", "mae", "r2"}


def test_write_experiment_log_writes_dataframe_to_csv(tmp_path):
    runs = pd.DataFrame({
        "run_id": ["abc"],
        "metrics.test_r2": [0.75],
        "params.model_type": ["random_forest"],
    })
    output_path = tmp_path / "experiment_log.csv"

    row_count = write_experiment_log(runs, output_path)

    assert row_count == 1
    written = pd.read_csv(output_path)
    assert written.to_dict(orient="records") == runs.to_dict(orient="records")


def test_write_experiment_log_rejects_empty_dataframe(tmp_path):
    with pytest.raises(ValueError, match="No runs"):
        write_experiment_log(pd.DataFrame(), tmp_path / "experiment_log.csv")


def test_write_experiment_log_strips_machine_specific_artifact_uri_prefix(tmp_path):
    runs = pd.DataFrame({
        "run_id": ["abc"],
        "artifact_uri": [r"file:D:/Users/dev/mlruns/123/run-id/artifacts/model"],
    })
    out_path = tmp_path / "experiment_log.csv"
    write_experiment_log(runs, out_path)
    written = pd.read_csv(out_path)
    assert written["artifact_uri"].iloc[0] == "mlruns/123/run-id/artifacts/model"


def test_export_runs_is_read_only_and_uses_mlflow_api(monkeypatch, tmp_path):
    class _MlflowCfg:
        tracking_uri = "file:./mlruns"
        experiment_name = "bike_sharing"

    class _Cfg:
        mlflow = _MlflowCfg()

    class _Experiment:
        experiment_id = "1"

    def fail_if_repair_called(*_args, **_kwargs):
        raise AssertionError("export should not mutate MLflow metadata")

    runs = pd.DataFrame({"run_id": ["abc"], "metrics.test_r2": [0.75]})
    output_path = tmp_path / "experiment_log.csv"

    monkeypatch.setattr(export_runs_module, "load_config", lambda: _Cfg())
    monkeypatch.setattr(
        export_runs_module,
        "ensure_run_uuid_compatibility",
        fail_if_repair_called,
        raising=False,
    )
    monkeypatch.setattr(export_runs_module.mlflow, "set_tracking_uri", lambda _uri: None)
    monkeypatch.setattr(
        export_runs_module.mlflow,
        "get_experiment_by_name",
        lambda _name: _Experiment(),
    )
    monkeypatch.setattr(
        export_runs_module.mlflow,
        "search_runs",
        lambda experiment_ids, output_format: runs,
    )

    row_count = export_runs(output_path)

    assert row_count == 1
    assert pd.read_csv(output_path).to_dict(orient="records") == runs.to_dict(
        orient="records"
    )
