from __future__ import annotations

import argparse
from pathlib import Path

# Directory *components* to skip anywhere in the relative path (POSIX parts).
EXCLUDED_DIR_PARTS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        ".venv_strict",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "node_modules",
        "mlruns",
    }
)

DEFAULT_EXCLUDE_PREFIXES: tuple[str, ...] = (
    "mlops-final/",
    ".cursor/",
    "mcps/",
)

# For audit docs, focus on source/config/docs (skip bulky binaries under data/processed).
DEFAULT_CODE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".yml",
        ".yaml",
        ".md",
        ".txt",
        ".ini",
        ".toml",
        ".json",
        ".dvc",
    }
)


def _should_skip_relative(rel: str, exclude_prefixes: tuple[str, ...]) -> bool:
    if any(rel == p.rstrip("/") or rel.startswith(p) for p in exclude_prefixes):
        return True
    parts = rel.split("/")
    if any(part in EXCLUDED_DIR_PARTS for part in parts):
        return True
    for idx, part in enumerate(parts):
        if part == ".dvc" and idx + 1 < len(parts) and parts[idx + 1] in {"cache", "tmp"}:
            return True
    return False


def collect_files(
    root: Path,
    *,
    exclude_prefixes: tuple[str, ...] = DEFAULT_EXCLUDE_PREFIXES,
    extensions: frozenset[str] | None = None,
) -> list[str]:
    root = root.resolve()
    items: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if _should_skip_relative(rel, exclude_prefixes):
            continue
        if extensions is not None:
            suf = path.suffix.lower()
            special = path.name in {"Dockerfile", ".dockerignore", ".gitignore"} or path.name.endswith(
                ".Dockerfile"
            )
            if not special and suf not in extensions:
                continue
        items.append(rel)
    return sorted(set(items))


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit sorted repository-relative file paths.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--exclude-prefix",
        action="append",
        default=[],
        help="Additional path prefixes to skip (POSIX, trailing slash recommended). Repeatable.",
    )
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="Only include editable config/source extensions (.py,.yaml,.md,…).",
    )
    parser.add_argument(
        "--no-default-excludes",
        action="store_true",
        help="Do not apply DEFAULT_EXCLUDE_PREFIXES (still skips cache dirs).",
    )
    args = parser.parse_args()
    exclude = tuple(DEFAULT_EXCLUDE_PREFIXES) if not args.no_default_excludes else ()
    exclude = exclude + tuple(args.exclude_prefix or ())
    extensions = DEFAULT_CODE_EXTENSIONS if args.code_only else None
    files = collect_files(args.root.resolve(), exclude_prefixes=exclude, extensions=extensions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(files) + "\n", encoding="utf-8")
    print(f"Wrote {len(files)} paths to {args.out}")


if __name__ == "__main__":
    main()
