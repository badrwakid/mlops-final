from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement

REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_FILES = [
    "README.md",
    "docs/model_card.md",
    "docs/data_card.md",
    "docs/experiment_log.csv",
    "requirements.txt",
    "configs/params.yaml",
    ".gitignore",
]


def _read_text(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8-sig")


def _markdown_headings(markdown_text: str) -> list[str]:
    headings: list[str] = []
    for line in markdown_text.splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
        if not match:
            continue
        heading = re.sub(r"\s+", " ", match.group(1)).strip().lower()
        if heading:
            headings.append(heading)
    return headings


def _has_heading_match(headings: list[str], aliases: tuple[str, ...]) -> bool:
    return any(any(alias in heading for alias in aliases) for heading in headings)


def _active_gitignore_rules(gitignore_text: str) -> list[str]:
    rules: list[str] = []
    for line in gitignore_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rules.append(stripped)
    return rules


def _resolve_case_insensitive_path(relative_path: str) -> Path | None:
    current = REPO_ROOT
    for part in Path(relative_path).parts:
        if not current.exists() or not current.is_dir():
            return None
        children = {child.name.casefold(): child for child in current.iterdir()}
        match = children.get(part.casefold())
        if match is None:
            return None
        current = match
    return current


@pytest.mark.parametrize("relative_path", REQUIRED_FILES)
def test_required_docs_and_repro_files_exist(relative_path: str) -> None:
    assert (REPO_ROOT / relative_path).exists(), f"Missing required file: {relative_path}"


def test_collaboration_artifacts_exist() -> None:
    required_paths = [
        "docs/team_collaboration.md",
        ".github/ISSUE_TEMPLATE/task.yml",
        "scripts/export_team_contribution.py",
    ]
    for relative_path in required_paths:
        resolved = _resolve_case_insensitive_path(relative_path)
        assert resolved is not None and resolved.is_file(), f"Missing required file: {relative_path}"

    pr_template_candidates = (
        ".github/PULL_REQUEST_TEMPLATE.md",
        ".github/pull_request_template.md",
    )
    has_pr_template = any(
        (resolved := _resolve_case_insensitive_path(candidate)) is not None and resolved.is_file()
        for candidate in pr_template_candidates
    )
    assert has_pr_template, "Missing PR template: expected uppercase or lowercase filename variant"

    collab_doc = _read_text("docs/team_collaboration.md").lower()
    assert "reviewer approval" in collab_doc or "approval before merge" in collab_doc, (
        "docs/team_collaboration.md must define PR approval expectations"
    )
    assert "direct pushes to `main` are not allowed" in collab_doc or "protected main" in collab_doc, (
        "docs/team_collaboration.md must describe main-branch protection policy"
    )
    assert "issue workflow" in collab_doc and "in review" in collab_doc and "done" in collab_doc, (
        "docs/team_collaboration.md must describe the issue workflow states"
    )

    pr_template_path = None
    for candidate in pr_template_candidates:
        resolved = _resolve_case_insensitive_path(candidate)
        if resolved is not None:
            pr_template_path = resolved
            break
    assert pr_template_path is not None, "PR template path resolution failed"
    pr_template_text = pr_template_path.read_text(encoding="utf-8-sig").lower()
    assert "checklist" in pr_template_text and "approved" in pr_template_text, (
        "PR template must include review approval checklist guidance"
    )
    assert "ci" in pr_template_text and "required checks" in pr_template_text, (
        "PR template must include CI/required-checks checklist guidance"
    )


def test_readme_has_title_and_quickstart_section() -> None:
    readme = _read_text("README.md")

    has_title = re.search(r"^\ufeff?\s*#\s+.+", readme) is not None
    has_quickstart = (
        re.search(r"^\s*##+\s+quick\s*start\b", readme, flags=re.IGNORECASE | re.MULTILINE)
        is not None
    )

    assert has_title, "README.md must contain a top-level markdown title"
    assert has_quickstart, "README.md must contain a quickstart section"


def test_model_card_contains_required_core_headings() -> None:
    model_card = _read_text("docs/model_card.md")
    headings = _markdown_headings(model_card)

    required_heading_aliases = {
        "description": ("description", "model details", "overview"),
        "intended use": ("intended use", "intended-use", "use case"),
        "training data": ("training data",),
        "metrics": ("metrics", "evaluation"),
        "subgroup": ("subgroup", "fairness"),
        "limitations": ("limitations",),
        "ethical": ("ethical", "ethics"),
    }

    missing = [
        key for key, aliases in required_heading_aliases.items() if not _has_heading_match(headings, aliases)
    ]
    assert not missing, f"docs/model_card.md is missing required heading concepts: {missing}"


def test_data_card_contains_required_core_headings() -> None:
    data_card = _read_text("docs/data_card.md")
    headings = _markdown_headings(data_card)

    required_heading_aliases = {
        "source": ("source", "dataset"),
        "schema": ("schema",),
        "preprocessing": ("preprocessing", "processing pipeline", "processing"),
        "bias": ("bias",),
        "privacy": ("privacy",),
        "license": ("license", "licensing"),
    }

    missing = [
        key for key, aliases in required_heading_aliases.items() if not _has_heading_match(headings, aliases)
    ]
    assert not missing, f"docs/data_card.md is missing required heading concepts: {missing}"


def test_experiment_log_has_required_contract_columns() -> None:
    csv_path = REPO_ROOT / "docs/experiment_log.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []

    assert "run_id" in fieldnames, "docs/experiment_log.csv must include 'run_id'"
    assert "status" in fieldnames, "docs/experiment_log.csv must include 'status'"
    assert any(
        col.startswith("metrics.") for col in fieldnames
    ), "docs/experiment_log.csv must include at least one 'metrics.*' column"
    assert any(
        col.startswith("params.") for col in fieldnames
    ), "docs/experiment_log.csv must include at least one 'params.*' column"


def test_requirements_are_exact_pinned_with_double_equals() -> None:
    requirements_lines = _read_text("requirements.txt").splitlines()

    dependency_specs: list[str] = []
    for raw_line in requirements_lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        dep_spec = stripped.split("#", 1)[0].strip()
        if dep_spec.startswith("-"):
            # Ignore pip options/include directives such as -r/--requirement/--index-url.
            continue
        if dep_spec:
            dependency_specs.append(dep_spec)

    assert dependency_specs, "requirements.txt must contain dependency entries"
    not_exact_pinned: list[str] = []
    for dep_spec in dependency_specs:
        try:
            requirement = Requirement(dep_spec)
        except InvalidRequirement:
            not_exact_pinned.append(dep_spec)
            continue

        specifiers = list(requirement.specifier)
        if len(specifiers) != 1 or specifiers[0].operator != "==":
            not_exact_pinned.append(dep_spec)

    assert (
        not not_exact_pinned
    ), f"All dependencies must be exact pinned with '=='. Offending entries: {not_exact_pinned}"


def test_gitignore_includes_venv_ide_and_data_exclusions() -> None:
    active_rules = _active_gitignore_rules(_read_text(".gitignore"))
    exclusion_rules = [rule for rule in active_rules if not rule.startswith("!")]

    has_venv_exclusion = any(rule in {".venv/", "venv/", "env/"} for rule in exclusion_rules)
    has_ide_exclusion = any(rule in {".vscode/", ".idea/"} for rule in exclusion_rules)
    has_data_exclusion = any(
        rule.startswith(("data/raw/", "data/raw/*", "data/processed/", "data/processed/*", "data/splits/", "data/splits/*"))
        for rule in exclusion_rules
    )

    assert has_venv_exclusion, ".gitignore must exclude virtual environment directories"
    assert has_ide_exclusion, ".gitignore must exclude IDE directories"
    assert has_data_exclusion, ".gitignore must exclude data directories"
