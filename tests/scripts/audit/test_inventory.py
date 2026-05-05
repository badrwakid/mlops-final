from pathlib import Path

from scripts.audit.build_inventory import collect_files


def test_collect_files_is_sorted_and_unique(tmp_path: Path) -> None:
    (tmp_path / "b.py").write_text("print('b')", encoding="utf-8")
    (tmp_path / "a.py").write_text("print('a')", encoding="utf-8")
    files = collect_files(tmp_path, exclude_prefixes=(), extensions=None)
    assert files == sorted(set(files))


def test_skips_excluded_nested_dir(tmp_path: Path) -> None:
    nested = tmp_path / ".venv" / "lib"
    nested.mkdir(parents=True)
    (nested / "x.py").write_text("x", encoding="utf-8")
    (tmp_path / "keep.py").write_text("k", encoding="utf-8")
    files = collect_files(tmp_path, exclude_prefixes=(), extensions=None)
    assert "keep.py" in files
    assert not any(f.startswith(".venv/") for f in files)
