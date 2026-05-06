from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"
RUNTIME_SOURCE_DIRS = (
    REPO_ROOT / "src",
    REPO_ROOT / "monitoring",
    REPO_ROOT / "scripts",
)

EXACT_PIN_RE = re.compile(r"^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?==[^=\s]+$")

KEYWORD_FAMILY_RE = re.compile(
    r"\b(?:threshold|limit|gate|min_test_r2|rmse_threshold|drift_threshold_share)\w*\b",
    flags=re.IGNORECASE,
)
NUMERIC_ASSIGNMENT_RE = re.compile(r"(?:=|:=)\s*-?\d+(?:\.\d+)?\b")
NUMERIC_COMPARISON_RE = re.compile(r"(?:<=|>=|<|>|==|!=)\s*-?\d+(?:\.\d+)?\b")


def _iter_dependency_specs() -> list[str]:
    specs: list[str] = []
    for raw_line in REQUIREMENTS_PATH.read_text(encoding="utf-8-sig").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        dep_spec = stripped.split("#", 1)[0].strip()
        if dep_spec and not dep_spec.startswith("-"):
            specs.append(dep_spec)
    return specs


def test_requirements_entries_use_exact_pins() -> None:
    specs = _iter_dependency_specs()
    assert specs, "requirements.txt must contain dependency entries"
    offenders = [spec for spec in specs if not EXACT_PIN_RE.fullmatch(spec)]
    assert not offenders, f"Dependencies must use exact pins (package==x.y.z): {offenders}"


def _iter_runtime_python_files() -> list[Path]:
    files: list[Path] = []
    for directory in RUNTIME_SOURCE_DIRS:
        files.extend(sorted(directory.rglob("*.py")))
    return files


def _looks_like_hardcoded_threshold_line(code_line: str) -> bool:
    # Keep this practical: only flag lines that combine threshold-ish words with
    # direct numeric assignment/comparison literals.
    if not KEYWORD_FAMILY_RE.search(code_line):
        return False
    has_numeric_gate = bool(NUMERIC_ASSIGNMENT_RE.search(code_line) or NUMERIC_COMPARISON_RE.search(code_line))
    if not has_numeric_gate:
        return False

    return bool(KEYWORD_FAMILY_RE.search(code_line))


def test_obvious_runtime_thresholds_are_not_hardcoded() -> None:
    offenders: list[str] = []
    for source_path in _iter_runtime_python_files():
        lines = source_path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            code = line.split("#", 1)[0]
            if not code.strip():
                continue
            if _looks_like_hardcoded_threshold_line(code):
                offenders.append(f"{source_path.relative_to(REPO_ROOT)}:{line_number}:{code.strip()}")

    assert not offenders, (
        "Avoid obvious hardcoded threshold literals in runtime code; use config values instead. "
        f"Found: {offenders}"
    )
