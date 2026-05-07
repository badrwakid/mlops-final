from pathlib import Path


def test_streamlit_app_exists() -> None:
    assert Path("src/dashboard/app.py").exists()


def test_streamlit_references_required_endpoints() -> None:
    content = Path("src/dashboard/app.py").read_text(encoding="utf-8")
    for endpoint in ["/health", "/ready", "/predict", "/predict/batch", "/metrics"]:
        assert endpoint in content


def test_required_pages_exist() -> None:
    content = Path("src/dashboard/app.py").read_text(encoding="utf-8")
    for page in [
        "Overview",
        "Live Prediction",
        "Batch Prediction",
        "MLflow Tracking",
        "Model Registry",
        "Monitoring & Drift",
        "Documentation Evidence",
    ]:
        assert page in content


def test_forbidden_pages_do_not_exist() -> None:
    content = Path("src/dashboard/app.py").read_text(encoding="utf-8").lower()
    assert "professor crash test" not in content
    assert "rubric checklist" not in content

