from pathlib import Path


def test_streamlit_app_exists() -> None:
    assert Path("src/dashboard/app.py").exists()


def test_streamlit_references_required_endpoints() -> None:
    content = Path("src/dashboard/app.py").read_text(encoding="utf-8")
    for endpoint in ["/health", "/ready", "/metrics"]:
        assert endpoint in content


def test_required_monitoring_pages_exist() -> None:
    content = Path("src/dashboard/app.py").read_text(encoding="utf-8")
    for page in [
        "Drift Overview",
        "Feature Drift Analysis",
        "Evidently Reports",
        "Prometheus Monitoring",
        "Monitoring Runbook",
        "Model & Data Context",
        "Documentation Evidence",
    ]:
        assert page in content


def test_monitoring_helpers_and_paths_exist() -> None:
    content = Path("src/dashboard/app.py").read_text(encoding="utf-8")
    required_tokens = [
        "drift_summary.json",
        "def load_json_file(",
        "def load_text_file(",
        "def parse_prometheus_metrics(",
        "def percent(",
    ]
    for token in required_tokens:
        assert token in content


def test_forbidden_pages_do_not_exist() -> None:
    content = Path("src/dashboard/app.py").read_text(encoding="utf-8").lower()
    assert "professor crash test" not in content
    assert "rubric checklist" not in content


def test_streamlit_requirement_present() -> None:
    content = Path("requirements-streamlit.txt").read_text(encoding="utf-8").lower()
    assert "streamlit==" in content


def test_readme_mentions_monitoring_dashboard() -> None:
    content = Path("README.md").read_text(encoding="utf-8")
    assert "Data Monitoring & Drift Detection Dashboard" in content

