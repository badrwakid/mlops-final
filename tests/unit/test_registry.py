from __future__ import annotations

from types import SimpleNamespace

import mlflow.tracking
import pytest
import src.config
import src.training.registry as registry
from src.training.registry import (
    ModelRunCandidate,
    _resolve_tracking_uri,
    build_promotion_decision,
    evaluate_candidate_promotion,
    find_best_model_run,
    prepare_candidate_registration,
    promote_version_to_production,
    register_candidate_model,
    select_best_run,
)


def test_resolve_tracking_uri_prefers_environment_override(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:./mlruns")

    assert _resolve_tracking_uri("http://localhost:5000") == "file:./mlruns"


def test_select_best_run_filters_invalid_rows_and_orders_by_r2_then_rmse():
    runs = [
        {
            "run_id": "missing-rmse",
            "metrics.test_r2": 0.99,
        },
        {
            "run_id": "low-r2",
            "metrics.test_r2": 0.72,
            "metrics.test_rmse": 50.0,
            "params.model_type": "random_forest",
        },
        {
            "run_id": "best-rmse",
            "metrics.test_r2": 0.81,
            "metrics.test_rmse": 60.0,
        },
        {
            "run_id": "best-r2",
            "metrics.test_r2": 0.82,
            "metrics.test_rmse": 80.0,
        },
        {
            "run_id": "invalid-r2",
            "metrics.test_r2": "not-a-number",
            "metrics.test_rmse": 1.0,
        },
    ]

    selected = select_best_run(runs)

    assert selected == ModelRunCandidate(
        run_id="best-r2",
        test_r2=0.82,
        test_rmse=80.0,
        metrics={"test_r2": 0.82, "test_rmse": 80.0},
        params={},
    )


def test_select_best_run_uses_lowest_rmse_then_run_id_for_deterministic_ties():
    runs = [
        ModelRunCandidate(
            run_id="run-c",
            test_r2=0.8,
            test_rmse=42.0,
            metrics={"test_r2": 0.8, "test_rmse": 42.0},
            params={},
        ),
        ModelRunCandidate(
            run_id="run-a",
            test_r2=0.8,
            test_rmse=42.0,
            metrics={"test_r2": 0.8, "test_rmse": 42.0},
            params={},
        ),
        ModelRunCandidate(
            run_id="run-b",
            test_r2=0.8,
            test_rmse=40.0,
            metrics={"test_r2": 0.8, "test_rmse": 40.0},
            params={},
        ),
    ]

    assert select_best_run(runs).run_id == "run-b"
    assert select_best_run(runs[:2]).run_id == "run-a"


def test_select_best_run_excludes_gate_failed_candidates_when_validation_state_exists():
    runs = [
        {
            "run_id": "gate-failed",
            "metrics.test_r2": 0.95,
            "metrics.test_rmse": 10.0,
            "params.validation_passed": "False",
        },
        {
            "run_id": "gate-passed",
            "metrics.test_r2": 0.80,
            "metrics.test_rmse": 20.0,
            "tags.validation_passed": "true",
        },
    ]

    assert select_best_run(runs).run_id == "gate-passed"


def test_select_best_run_excludes_non_finished_candidates_when_status_exists():
    runs = [
        {
            "run_id": "running",
            "status": "RUNNING",
            "metrics.test_r2": 0.95,
            "metrics.test_rmse": 10.0,
            "params.validation_passed": True,
        },
        {
            "run_id": "finished",
            "status": "FINISHED",
            "metrics.test_r2": 0.80,
            "metrics.test_rmse": 20.0,
            "params.validation_passed": True,
        },
    ]

    assert select_best_run(runs).run_id == "finished"


def test_find_best_model_run_enforces_min_test_r2_gate():
    class FakeClient:
        def get_experiment_by_name(self, experiment_name):
            assert experiment_name == "bike_sharing"
            return SimpleNamespace(experiment_id="1")

        def search_runs(self, experiment_ids, max_results):
            assert experiment_ids == ["1"]
            return [
                SimpleNamespace(
                    info=SimpleNamespace(run_id="below-gate", status="FINISHED"),
                    data=SimpleNamespace(
                        metrics={"test_r2": 0.69, "test_rmse": 66.7},
                        params={"validation_passed": "True"},
                        tags={},
                    ),
                )
            ]

    with pytest.raises(RuntimeError, match="No eligible model runs"):
        find_best_model_run(FakeClient(), "bike_sharing", min_test_r2=0.70)


def test_register_candidate_model_builds_runs_uri_and_sets_version_metadata():
    calls = []

    class FakeClient:
        def create_registered_model(self, name):
            calls.append(("create_registered_model", name))

        def create_model_version(self, name, source, run_id):
            calls.append(("create_model_version", name, source, run_id))
            return SimpleNamespace(version="7")

        def set_model_version_tag(self, name, version, key, value):
            calls.append(("set_model_version_tag", name, version, key, value))

    candidate = ModelRunCandidate(
        run_id="abc123",
        test_r2=0.757,
        test_rmse=66.71,
        metrics={"test_r2": 0.757, "test_rmse": 66.71, "best_cv_rmse": 64.39},
        params={"model_type": "random_forest"},
    )

    version = register_candidate_model(FakeClient(), candidate, "bike_share_regressor")

    assert version.version == "7"
    assert ("create_model_version", "bike_share_regressor", "runs:/abc123/model", "abc123") in calls
    assert (
        "set_model_version_tag",
        "bike_share_regressor",
        "7",
        "source_run_id",
        "abc123",
    ) in calls
    assert (
        "set_model_version_tag",
        "bike_share_regressor",
        "7",
        "test_r2",
        "0.757",
    ) in calls
    assert (
        "set_model_version_tag",
        "bike_share_regressor",
        "7",
        "test_rmse",
        "66.71",
    ) in calls
    assert (
        "set_model_version_tag",
        "bike_share_regressor",
        "7",
        "param.model_type",
        "random_forest",
    ) in calls
    assert (
        "set_model_version_tag",
        "bike_share_regressor",
        "7",
        "metric.best_cv_rmse",
        "64.39",
    ) in calls


def test_find_best_model_run_keeps_search_page_size_within_mlflow_file_store_limit():
    class FakeClient:
        def get_experiment_by_name(self, experiment_name):
            assert experiment_name == "bike_sharing"
            return SimpleNamespace(experiment_id="1")

        def search_runs(self, experiment_ids, max_results):
            assert experiment_ids == ["1"]
            assert max_results <= 50_000
            return [
                SimpleNamespace(
                    info=SimpleNamespace(run_id="candidate"),
                    data=SimpleNamespace(
                        metrics={"test_r2": 0.75, "test_rmse": 66.7},
                        params={},
                    ),
                )
            ]

    candidate = find_best_model_run(FakeClient(), "bike_sharing")

    assert candidate.run_id == "candidate"


def _registry_config(min_test_r2=0.70, rmse_threshold=80.0):
    return SimpleNamespace(
        mlflow=SimpleNamespace(
            tracking_uri="http://localhost:5000",
            experiment_name="bike_sharing",
            registered_model_name="bike_share_regressor",
        ),
        validation=SimpleNamespace(min_test_r2=min_test_r2, rmse_threshold=rmse_threshold),
    )


def _finished_candidate_run(run_id="candidate", test_r2=0.80):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, status="FINISHED"),
        data=SimpleNamespace(
            metrics={"test_r2": test_r2, "test_rmse": 66.7},
            params={"validation_passed": "True"},
            tags={},
        ),
    )


def test_prepare_candidate_registration_reuses_existing_version_without_promotion(monkeypatch):
    calls = []

    class FakeClient:
        def get_experiment_by_name(self, experiment_name):
            return SimpleNamespace(experiment_id="1")

        def search_runs(self, experiment_ids, max_results):
            return [_finished_candidate_run(run_id="existing-run")]

        def search_model_versions(self, filter_string):
            calls.append(("search_model_versions", filter_string))
            return [
                SimpleNamespace(
                    version="3",
                    run_id="existing-run",
                    source="runs:/existing-run/model",
                    tags={"run_id": "existing-run"},
                )
            ]

        def create_model_version(self, name, source, run_id):
            raise AssertionError("prepare should reuse existing version")

        def transition_model_version_stage(self, *args, **kwargs):
            raise AssertionError("prepare must not promote stages")

        def set_registered_model_alias(self, *args, **kwargs):
            raise AssertionError("prepare must not set aliases")

    fake_client = FakeClient()
    monkeypatch.setattr(src.config, "load_config", lambda: _registry_config())
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda tracking_uri: fake_client)

    result = prepare_candidate_registration()

    assert result.created is False
    assert result.version.version == "3"
    assert calls == [("search_model_versions", "name = 'bike_share_regressor'")]


def test_prepare_candidate_registration_checks_model_artifact_before_registering(monkeypatch):
    class FakeClient:
        def get_experiment_by_name(self, experiment_name):
            return SimpleNamespace(experiment_id="1")

        def search_runs(self, experiment_ids, max_results):
            return [_finished_candidate_run(run_id="missing-artifact")]

        def search_model_versions(self, filter_string):
            return []

        def list_artifacts(self, run_id, path):
            assert (run_id, path) == ("missing-artifact", "model")
            return []

        def create_model_version(self, name, source, run_id):
            raise AssertionError("missing artifact should block registration")

    fake_client = FakeClient()
    monkeypatch.setattr(src.config, "load_config", lambda: _registry_config())
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda tracking_uri: fake_client)

    with pytest.raises(RuntimeError, match="Model artifact"):
        prepare_candidate_registration()


def _candidate(run_id="candidate", test_r2=0.80, test_rmse=66.7):
    return ModelRunCandidate(
        run_id=run_id,
        test_r2=test_r2,
        test_rmse=test_rmse,
        metrics={"test_r2": test_r2, "test_rmse": test_rmse},
        params={},
    )


def _version(version="7", run_id="candidate", stage="None", tags=None):
    return SimpleNamespace(
        version=version,
        run_id=run_id,
        current_stage=stage,
        tags=tags or {},
    )


def test_promotion_decision_allows_candidate_without_current_production_when_gate_passes():
    decision = build_promotion_decision(
        candidate=_candidate(test_r2=0.81, test_rmse=61.0),
        candidate_version=_version(version="5"),
        min_test_r2=0.70,
        current_production=None,
    )

    assert decision.should_promote is True
    assert decision.candidate.run_id == "candidate"
    assert decision.candidate.version == "5"
    assert decision.current_production is None
    assert decision.compare_status == "no_current_production"
    assert "No current Production model found." in decision.safety_notes


def test_promotion_decision_rejects_candidate_below_minimum_gate():
    decision = build_promotion_decision(
        candidate=_candidate(test_r2=0.69, test_rmse=61.0),
        candidate_version=_version(version="5"),
        min_test_r2=0.70,
        current_production=None,
    )

    assert decision.should_promote is False
    assert decision.compare_status == "below_min_test_r2"
    assert "Candidate test_r2=0.690000 is below promotion gate 0.700000." in decision.risk_notes


def test_promotion_decision_rejects_degradation_vs_current_production():
    decision = build_promotion_decision(
        candidate=_candidate(test_r2=0.80, test_rmse=61.0),
        candidate_version=_version(version="5"),
        min_test_r2=0.70,
        current_production=_version(
            version="4",
            run_id="prod-run",
            stage="Production",
            tags={"test_r2": "0.82", "test_rmse": "64.0"},
        ),
    )

    assert decision.should_promote is False
    assert decision.compare_status == "candidate_degrades_r2"
    assert decision.current_production.version == "4"
    assert decision.current_production.metrics == {"test_r2": 0.82, "test_rmse": 64.0}


def test_promotion_decision_allows_equal_r2_with_lower_rmse_tie_break():
    decision = build_promotion_decision(
        candidate=_candidate(test_r2=0.82, test_rmse=60.0),
        candidate_version=_version(version="5"),
        min_test_r2=0.70,
        current_production=_version(
            version="4",
            run_id="prod-run",
            stage="Production",
            tags={"test_r2": "0.82", "test_rmse": "64.0"},
        ),
    )

    assert decision.should_promote is True
    assert decision.compare_status == "candidate_ties_r2_and_improves_rmse"
    assert "Candidate matches Production test_r2 and improves or matches test_rmse." in decision.safety_notes


def test_promotion_decision_rejects_equal_r2_with_higher_rmse_tie_break():
    decision = build_promotion_decision(
        candidate=_candidate(test_r2=0.82, test_rmse=70.0),
        candidate_version=_version(version="5"),
        min_test_r2=0.70,
        current_production=_version(
            version="4",
            run_id="prod-run",
            stage="Production",
            tags={"test_r2": "0.82", "test_rmse": "64.0"},
        ),
    )

    assert decision.should_promote is False
    assert decision.compare_status == "candidate_ties_r2_but_degrades_rmse"
    assert "Candidate matches Production test_r2 but has higher test_rmse." in decision.risk_notes


def test_promotion_decision_rejects_when_current_production_metrics_unknown():
    decision = build_promotion_decision(
        candidate=_candidate(test_r2=0.82, test_rmse=60.0),
        candidate_version=_version(version="5"),
        min_test_r2=0.70,
        current_production=_version(version="4", run_id="prod-run", stage="Production"),
    )

    assert decision.should_promote is False
    assert decision.compare_status == "current_production_metrics_unknown"
    assert decision.current_production.metrics == {}
    assert "Current Production metrics unavailable; manual metric comparison required." in decision.risk_notes


def test_promotion_decision_resolves_current_production_metrics_from_run_when_tags_missing():
    class FakeClient:
        def get_run(self, run_id):
            assert run_id == "prod-run"
            return SimpleNamespace(
                data=SimpleNamespace(metrics={"test_r2": 0.81, "test_rmse": 64.0})
            )

    decision = build_promotion_decision(
        candidate=_candidate(test_r2=0.82, test_rmse=60.0),
        candidate_version=_version(version="5"),
        min_test_r2=0.70,
        current_production=_version(version="4", run_id="prod-run", stage="Production"),
        client=FakeClient(),
    )

    assert decision.should_promote is True
    assert decision.compare_status == "candidate_improves_r2"
    assert decision.current_production.metrics == {"test_r2": 0.81, "test_rmse": 64.0}


def test_main_default_path_prints_help_without_registration_or_promotion(monkeypatch, capsys):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("default CLI path must not prepare, evaluate, or promote")

    monkeypatch.setattr(registry, "prepare_candidate_registration", fail_if_called)
    monkeypatch.setattr(registry, "evaluate_candidate_promotion", fail_if_called)
    monkeypatch.setattr(registry, "promote_version_to_production", fail_if_called)

    registry.main([])

    captured = capsys.readouterr()
    assert "usage:" in captured.out
    assert "prepare" in captured.out
    assert "evaluate" in captured.out
    assert "promote" in captured.out
    assert captured.err == ""


def test_promote_version_to_production_transitions_staging_then_production_and_archives():
    calls = []

    class FakeClient:
        def transition_model_version_stage(self, **kwargs):
            calls.append(kwargs)

        def set_model_version_tag(self, name, version, key, value):
            calls.append(
                {
                    "name": name,
                    "version": version,
                    "key": key,
                    "value": value,
                }
            )

    decision = build_promotion_decision(
        candidate=_candidate(test_r2=0.82, test_rmse=60.0),
        candidate_version=_version(version="7"),
        min_test_r2=0.70,
        current_production=_version(
            version="4",
            run_id="prod-run",
            stage="Production",
            tags={"test_r2": "0.81", "test_rmse": "64.0"},
        ),
    )

    promote_version_to_production(
        FakeClient(),
        "bike_share_regressor",
        "7",
        decision=decision,
        approval_mode="test",
    )

    transition_calls = [call for call in calls if "stage" in call]
    tag_calls = [call for call in calls if "key" in call]
    assert transition_calls == [
        {
            "name": "bike_share_regressor",
            "version": "7",
            "stage": "Staging",
            "archive_existing_versions": False,
        },
        {
            "name": "bike_share_regressor",
            "version": "7",
            "stage": "Production",
            "archive_existing_versions": True,
        },
    ]
    tags = {call["key"]: call["value"] for call in tag_calls}
    assert tags["promotion.approved"] == "true"
    assert tags["promotion.approval_mode"] == "test"
    assert tags["promotion.previous_production_version"] == "4"
    assert tags["promotion.compare_status"] == "candidate_improves_r2"
    assert tags["promotion.candidate.test_r2"] == "0.82"
    assert tags["promotion.candidate.test_rmse"] == "60.0"
    assert tags["promotion.production.test_r2"] == "0.81"
    assert tags["promotion.production.test_rmse"] == "64.0"
    assert "promotion.decision_timestamp_utc" in tags


def test_evaluate_candidate_promotion_prepares_and_decides_without_stage_transition(monkeypatch):
    class FakeClient:
        def get_experiment_by_name(self, experiment_name):
            return SimpleNamespace(experiment_id="1")

        def search_runs(self, experiment_ids, max_results):
            return [_finished_candidate_run(run_id="existing-run")]

        def search_model_versions(self, filter_string):
            return [
                SimpleNamespace(
                    version="3",
                    run_id="existing-run",
                    source="runs:/existing-run/model",
                    current_stage="None",
                    tags={"run_id": "existing-run", "test_r2": "0.80", "test_rmse": "66.7"},
                )
            ]

        def transition_model_version_stage(self, *args, **kwargs):
            raise AssertionError("evaluate must not promote stages")

    fake_client = FakeClient()
    monkeypatch.setattr(src.config, "load_config", lambda: _registry_config())
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda tracking_uri: fake_client)

    decision = evaluate_candidate_promotion()

    assert decision.should_promote is True
    assert decision.candidate.version == "3"
    assert decision.current_production is None
