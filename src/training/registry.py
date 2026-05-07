from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MLFLOW_SEARCH_MAX_RESULTS = 50_000


@dataclass(frozen=True)
class ModelRunCandidate:
    run_id: str
    test_r2: float
    test_rmse: float
    metrics: dict[str, Any]
    params: dict[str, Any]
    status: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegistrationResult:
    candidate: ModelRunCandidate
    model_name: str
    version: Any
    created: bool


@dataclass(frozen=True)
class ModelVersionSummary:
    version: str
    run_id: str | None
    metrics: dict[str, float]
    current_stage: str | None = None


@dataclass(frozen=True)
class PromotionDecision:
    candidate: ModelVersionSummary
    current_production: ModelVersionSummary | None
    should_promote: bool
    compare_status: str
    safety_notes: list[str]
    risk_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_tracking_uri(configured_uri: str) -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or configured_uri


def _as_finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _strip_prefixed_values(row: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    return {
        key.removeprefix(prefix): value
        for key, value in row.items()
        if isinstance(key, str) and key.startswith(prefix)
    }


def _mapping_candidate(row: Mapping[str, Any]) -> ModelRunCandidate | None:
    run_id = row.get("run_id") or row.get("info.run_id") or row.get("run.info.run_id")
    if run_id is None:
        return None

    metrics = dict(row.get("metrics") or {})
    metrics.update(_strip_prefixed_values(row, "metrics."))
    for key in ("test_r2", "test_rmse"):
        if key in row and key not in metrics:
            metrics[key] = row[key]

    params = dict(row.get("params") or {})
    params.update(_strip_prefixed_values(row, "params."))
    if "validation_passed" in row and "validation_passed" not in params:
        params["validation_passed"] = row["validation_passed"]

    tags = dict(row.get("tags") or {})
    tags.update(_strip_prefixed_values(row, "tags."))
    status = row.get("status") or row.get("info.status") or row.get("run.info.status")

    return _build_candidate(str(run_id), metrics, params, status=status, tags=tags)


def _mlflow_run_candidate(run: Any) -> ModelRunCandidate | None:
    run_info = getattr(run, "info", None)
    run_data = getattr(run, "data", None)
    run_id = getattr(run_info, "run_id", None)
    if run_id is None:
        run_id = getattr(run, "run_id", None)
    if run_id is None:
        return None

    metrics = dict(getattr(run_data, "metrics", None) or getattr(run, "metrics", None) or {})
    params = dict(getattr(run_data, "params", None) or getattr(run, "params", None) or {})
    tags = dict(getattr(run_data, "tags", None) or getattr(run, "tags", None) or {})
    status = getattr(run_info, "status", None) or getattr(run, "status", None)
    return _build_candidate(str(run_id), metrics, params, status=status, tags=tags)


def _build_candidate(
    run_id: str,
    metrics: Mapping[str, Any],
    params: Mapping[str, Any],
    status: Any = None,
    tags: Mapping[str, Any] | None = None,
) -> ModelRunCandidate | None:
    test_r2 = _as_finite_float(metrics.get("test_r2"))
    test_rmse = _as_finite_float(metrics.get("test_rmse"))
    if test_r2 is None or test_rmse is None:
        return None
    return ModelRunCandidate(
        run_id=run_id,
        test_r2=test_r2,
        test_rmse=test_rmse,
        metrics=dict(metrics),
        params=dict(params),
        status=str(status) if status is not None else None,
        tags=dict(tags or {}),
    )


def _to_candidate(run: Any) -> ModelRunCandidate | None:
    if isinstance(run, ModelRunCandidate):
        return _build_candidate(run.run_id, run.metrics, run.params, status=run.status, tags=run.tags)
    if isinstance(run, Mapping):
        return _mapping_candidate(run)
    return _mlflow_run_candidate(run)


def _iter_run_records(runs: Any) -> Iterable[Any]:
    if hasattr(runs, "to_dict"):
        return runs.to_dict(orient="records")
    return runs


def _normalized_validation_key(key: Any) -> str:
    return str(key).lower().replace(".", "_").replace("-", "_")


def _validation_passed_value(candidate: ModelRunCandidate) -> Any | None:
    for values in (candidate.params, candidate.tags):
        for key, value in values.items():
            normalized_key = _normalized_validation_key(key)
            if normalized_key in {"validation_passed", "gate_passed"}:
                return value
    return None


def _is_validation_passed(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.lower() == "true")


def _is_candidate_eligible(
    candidate: ModelRunCandidate,
    min_test_r2: float | None = None,
) -> bool:
    if candidate.status is not None and candidate.status.upper() != "FINISHED":
        return False

    validation_passed = _validation_passed_value(candidate)
    if validation_passed is not None and not _is_validation_passed(validation_passed):
        return False

    if min_test_r2 is not None and candidate.test_r2 < min_test_r2:
        return False

    return True


def select_best_run(
    runs: Iterable[Any],
    min_test_r2: float | None = None,
) -> ModelRunCandidate | None:
    candidates = [
        candidate
        for candidate in (_to_candidate(run) for run in _iter_run_records(runs))
        if candidate is not None and _is_candidate_eligible(candidate, min_test_r2)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: (-candidate.test_r2, candidate.test_rmse, candidate.run_id))


def find_best_model_run(
    client: Any,
    experiment_name: str,
    max_results: int = MLFLOW_SEARCH_MAX_RESULTS,
    *,
    min_test_r2: float | None = None,
) -> ModelRunCandidate:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=min(max_results, MLFLOW_SEARCH_MAX_RESULTS),
    )
    candidate = select_best_run(runs, min_test_r2=min_test_r2)
    if candidate is None:
        raise RuntimeError(f"No eligible model runs found for experiment: {experiment_name}")
    return candidate


def _is_already_exists_error(exc: Exception) -> bool:
    error_code = getattr(exc, "error_code", None)
    return error_code == "RESOURCE_ALREADY_EXISTS" or "already exists" in str(exc).lower()


def _create_registered_model_if_needed(client: Any, model_name: str) -> None:
    from mlflow.exceptions import MlflowException

    try:
        client.create_registered_model(model_name)
    except MlflowException as exc:
        if not _is_already_exists_error(exc):
            raise


def register_candidate_model(client: Any, candidate: ModelRunCandidate, model_name: str) -> Any:
    _create_registered_model_if_needed(client, model_name)
    source = f"runs:/{candidate.run_id}/model"
    version = client.create_model_version(
        name=model_name,
        source=source,
        run_id=candidate.run_id,
    )
    tags = {
        "run_id": candidate.run_id,
        "source_run_id": candidate.run_id,
        "test_r2": str(candidate.test_r2),
        "test_rmse": str(candidate.test_rmse),
    }
    tags.update(
        {f"metric.{key}": str(candidate.metrics[key]) for key in sorted(candidate.metrics)}
    )
    tags.update(
        {f"param.{key}": str(candidate.params[key]) for key in sorted(candidate.params)}
    )
    for key, value in tags.items():
        client.set_model_version_tag(model_name, version.version, key, value)
    return version


def _find_existing_model_version_for_run(client: Any, model_name: str, run_id: str) -> Any | None:
    source = f"runs:/{run_id}/model"
    versions = client.search_model_versions(f"name = '{model_name}'")
    for version in sorted(versions, key=lambda item: str(getattr(item, "version", ""))):
        tags = getattr(version, "tags", None) or {}
        if getattr(version, "run_id", None) == run_id:
            return version
        if getattr(version, "source", None) == source:
            return version
        if tags.get("run_id") == run_id or tags.get("source_run_id") == run_id:
            return version
    return None


def _version_number(version: Any) -> int:
    try:
        return int(getattr(version, "version", version))
    except (TypeError, ValueError):
        return 0


def _version_str(version: Any) -> str:
    return str(getattr(version, "version", version))


def _model_version_metric(version: Any, key: str) -> float | None:
    tags = getattr(version, "tags", None) or {}
    for candidate_key in (key, f"metric.{key}", f"metrics.{key}"):
        metric = _as_finite_float(tags.get(candidate_key))
        if metric is not None:
            return metric
    return None


def _run_metrics(client: Any | None, run_id: str | None) -> dict[str, float]:
    if client is None or run_id is None:
        return {}
    try:
        run = client.get_run(run_id)
    except Exception:
        return {}
    run_data = getattr(run, "data", None)
    metrics = getattr(run_data, "metrics", None) or getattr(run, "metrics", None) or {}
    return {
        key: metric
        for key in ("test_r2", "test_rmse")
        if (metric := _as_finite_float(metrics.get(key))) is not None
    }


def _model_version_summary(version: Any, client: Any | None = None) -> ModelVersionSummary:
    metrics = {
        key: metric
        for key in ("test_r2", "test_rmse")
        if (metric := _model_version_metric(version, key)) is not None
    }
    if len(metrics) < 2:
        metrics.update(_run_metrics(client, getattr(version, "run_id", None)))
    return ModelVersionSummary(
        version=_version_str(version),
        run_id=getattr(version, "run_id", None),
        metrics=metrics,
        current_stage=getattr(version, "current_stage", None),
    )


def find_current_production_version(client: Any, model_name: str) -> Any | None:
    versions = client.search_model_versions(f"name = '{model_name}'")
    production_versions = [
        version
        for version in versions
        if getattr(version, "current_stage", None) == "Production"
    ]
    if not production_versions:
        return None
    return max(production_versions, key=_version_number)


def build_promotion_decision(
    *,
    candidate: ModelRunCandidate,
    candidate_version: Any,
    min_test_r2: float,
    current_production: Any | None,
    client: Any | None = None,
) -> PromotionDecision:
    candidate_summary = ModelVersionSummary(
        version=_version_str(candidate_version),
        run_id=candidate.run_id,
        metrics={"test_r2": candidate.test_r2, "test_rmse": candidate.test_rmse},
        current_stage=getattr(candidate_version, "current_stage", None),
    )
    safety_notes: list[str] = []
    risk_notes: list[str] = []

    if candidate.test_r2 < min_test_r2:
        risk_notes.append(
            f"Candidate test_r2={candidate.test_r2:.6f} is below promotion gate {min_test_r2:.6f}."
        )
        return PromotionDecision(
            candidate=candidate_summary,
            current_production=_model_version_summary(current_production, client)
            if current_production is not None
            else None,
            should_promote=False,
            compare_status="below_min_test_r2",
            safety_notes=safety_notes,
            risk_notes=risk_notes,
        )

    safety_notes.append(
        f"Candidate passes min_test_r2 gate: {candidate.test_r2:.6f} >= {min_test_r2:.6f}."
    )

    if current_production is None:
        safety_notes.append("No current Production model found.")
        return PromotionDecision(
            candidate=candidate_summary,
            current_production=None,
            should_promote=True,
            compare_status="no_current_production",
            safety_notes=safety_notes,
            risk_notes=risk_notes,
        )

    production_summary = _model_version_summary(current_production, client)
    production_r2 = production_summary.metrics.get("test_r2")
    production_rmse = production_summary.metrics.get("test_rmse")

    if production_r2 is None or production_rmse is None:
        risk_notes.append("Current Production metrics unavailable; manual metric comparison required.")
        return PromotionDecision(
            candidate=candidate_summary,
            current_production=production_summary,
            should_promote=False,
            compare_status="current_production_metrics_unknown",
            safety_notes=safety_notes,
            risk_notes=risk_notes,
        )

    if candidate.test_r2 > production_r2:
        safety_notes.append("Candidate improves Production test_r2.")
        return PromotionDecision(
            candidate=candidate_summary,
            current_production=production_summary,
            should_promote=True,
            compare_status="candidate_improves_r2",
            safety_notes=safety_notes,
            risk_notes=risk_notes,
        )

    if candidate.test_r2 < production_r2:
        risk_notes.append("Candidate test_r2 is lower than current Production.")
        return PromotionDecision(
            candidate=candidate_summary,
            current_production=production_summary,
            should_promote=False,
            compare_status="candidate_degrades_r2",
            safety_notes=safety_notes,
            risk_notes=risk_notes,
        )

    if candidate.test_rmse <= production_rmse:
        safety_notes.append("Candidate matches Production test_r2 and improves or matches test_rmse.")
        return PromotionDecision(
            candidate=candidate_summary,
            current_production=production_summary,
            should_promote=True,
            compare_status="candidate_ties_r2_and_improves_rmse",
            safety_notes=safety_notes,
            risk_notes=risk_notes,
        )

    risk_notes.append("Candidate matches Production test_r2 but has higher test_rmse.")
    return PromotionDecision(
        candidate=candidate_summary,
        current_production=production_summary,
        should_promote=False,
        compare_status="candidate_ties_r2_but_degrades_rmse",
        safety_notes=safety_notes,
        risk_notes=risk_notes,
    )


def _promotion_audit_tags(
    decision: PromotionDecision,
    *,
    approval_mode: str,
    approved: bool,
) -> dict[str, str]:
    tags = {
        "promotion.decision_timestamp_utc": datetime.now(UTC).isoformat(),
        "promotion.approved": str(approved).lower(),
        "promotion.approval_mode": approval_mode,
        "promotion.compare_status": decision.compare_status,
        "promotion.should_promote": str(decision.should_promote).lower(),
        "promotion.candidate.test_r2": str(decision.candidate.metrics["test_r2"]),
        "promotion.candidate.test_rmse": str(decision.candidate.metrics["test_rmse"]),
    }
    if decision.current_production is not None:
        tags["promotion.previous_production_version"] = decision.current_production.version
        for metric_name, metric_value in decision.current_production.metrics.items():
            tags[f"promotion.production.{metric_name}"] = str(metric_value)
    else:
        tags["promotion.previous_production_version"] = ""
    return tags


def _set_promotion_audit_tags(
    client: Any,
    model_name: str,
    version: str,
    decision: PromotionDecision,
    *,
    approval_mode: str,
    approved: bool,
) -> None:
    for key, value in _promotion_audit_tags(
        decision,
        approval_mode=approval_mode,
        approved=approved,
    ).items():
        client.set_model_version_tag(model_name, version, key, value)


def promote_version_to_production(
    client: Any,
    model_name: str,
    version: Any,
    *,
    decision: PromotionDecision,
    approval_mode: str = "explicit_cli_yes",
    approved: bool = True,
) -> str:
    if not decision.should_promote:
        raise ValueError(
            "Promotion rejected by policy decision: "
            f"compare_status={decision.compare_status} should_promote={decision.should_promote}"
        )
    if not approved:
        raise ValueError(
            "Promotion blocked: explicit approval is required "
            f"(approval_mode={approval_mode}, compare_status={decision.compare_status})."
        )
    version_str = _version_str(version)
    _set_promotion_audit_tags(
        client,
        model_name,
        version_str,
        decision,
        approval_mode=approval_mode,
        approved=approved,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*(transition_model_version_stage|model registry stages|deprecated).*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*(transition_model_version_stage|model registry stages|deprecated).*",
            category=DeprecationWarning,
        )
        client.transition_model_version_stage(
            name=model_name,
            version=version_str,
            stage="Staging",
            archive_existing_versions=False,
        )
        client.transition_model_version_stage(
            name=model_name,
            version=version_str,
            stage="Production",
            archive_existing_versions=True,
        )
    return version_str


def _verify_model_artifact_exists(
    client: Any,
    candidate: ModelRunCandidate,
    artifact_path: str = "model",
) -> None:
    source = f"runs:/{candidate.run_id}/{artifact_path}"
    try:
        artifacts = client.list_artifacts(candidate.run_id, artifact_path)
    except Exception as exc:
        raise RuntimeError(f"Model artifact is not readable for run {candidate.run_id}: {source}") from exc

    if not artifacts:
        raise RuntimeError(f"Model artifact not found for run {candidate.run_id}: {source}")


def prepare_candidate_registration(config_path: str | Path | None = None) -> RegistrationResult:
    import mlflow
    from mlflow.tracking import MlflowClient

    from src.config import load_config

    cfg = load_config(config_path) if config_path is not None else load_config()
    tracking_uri = _resolve_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    candidate = find_best_model_run(
        client,
        cfg.mlflow.experiment_name,
        min_test_r2=cfg.validation.min_test_r2,
    )
    existing_version = _find_existing_model_version_for_run(
        client,
        cfg.mlflow.registered_model_name,
        candidate.run_id,
    )
    if existing_version is None:
        _verify_model_artifact_exists(client, candidate)
        version = register_candidate_model(client, candidate, cfg.mlflow.registered_model_name)
        created = True
    else:
        version = existing_version
        created = False

    action = "created" if created else "reused"
    print(
        "registry: "
        f"{action} "
        f"model={cfg.mlflow.registered_model_name} "
        f"version={version.version} "
        f"run_id={candidate.run_id} "
        f"test_r2={candidate.test_r2:.6f} "
        f"test_rmse={candidate.test_rmse:.6f}"
    )
    return RegistrationResult(
        candidate=candidate,
        model_name=cfg.mlflow.registered_model_name,
        version=version,
        created=created,
    )


def evaluate_candidate_promotion(config_path: str | Path | None = None) -> PromotionDecision:
    import mlflow
    from mlflow.tracking import MlflowClient

    from src.config import load_config

    registration = prepare_candidate_registration(config_path)
    cfg = load_config(config_path) if config_path is not None else load_config()
    tracking_uri = _resolve_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    current_production = find_current_production_version(
        client,
        cfg.mlflow.registered_model_name,
    )
    decision = build_promotion_decision(
        candidate=registration.candidate,
        candidate_version=registration.version,
        min_test_r2=cfg.validation.min_test_r2,
        current_production=current_production,
        client=client,
    )
    print(json.dumps(decision.to_dict(), indent=2, sort_keys=True))
    return decision


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MLflow registry utilities")
    subparsers = parser.add_subparsers(dest="command")
    prepare_parser = subparsers.add_parser("prepare", help="register the best eligible candidate")
    prepare_parser.add_argument("--config", type=Path, default=None)
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="register/reuse the best eligible candidate and print a non-promoting decision",
    )
    evaluate_parser.add_argument("--config", type=Path, default=None)
    promote_parser = subparsers.add_parser(
        "promote",
        help="explicitly promote the evaluated candidate to Production",
    )
    promote_parser.add_argument("--config", type=Path, default=None)
    promote_parser.add_argument(
        "--yes",
        action="store_true",
        help="confirm the human gate and execute the Production transition",
    )

    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare_candidate_registration(args.config)
        return
    if args.command == "evaluate":
        evaluate_candidate_promotion(args.config)
        return
    if args.command == "promote":
        if not args.yes:
            decision = evaluate_candidate_promotion(args.config)
            raise SystemExit(
                "Promotion requires explicit human confirmation. "
                "Review the decision above, then rerun with --yes."
            )
        import mlflow
        from mlflow.tracking import MlflowClient

        from src.config import load_config

        decision = evaluate_candidate_promotion(args.config)
        if not decision.should_promote:
            raise SystemExit(f"Promotion rejected: {decision.compare_status}")
        cfg = load_config(args.config) if args.config is not None else load_config()
        tracking_uri = _resolve_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        version = promote_version_to_production(
            client,
            cfg.mlflow.registered_model_name,
            decision.candidate.version,
            decision=decision,
        )
        print(f"registry: promoted model={cfg.mlflow.registered_model_name} version={version}")
        return
    parser.print_help()


if __name__ == "__main__":
    main()
