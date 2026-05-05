from scripts.audit.generate_audit_skeleton import render_section, verify_complete


def test_render_section_contains_required_headings() -> None:
    section = render_section("src/data/prepare.py", use_placeholders=True)
    assert "## src/data/prepare.py" in section
    assert "### Purpose" in section
    assert "### Line-by-line findings" in section
    assert "### Exact code fixes" in section


def test_verify_complete_rejects_placeholder() -> None:
    doc = "# T\n\n" + render_section("a.py", use_placeholders=True)
    ok, errs = verify_complete(doc)
    assert not ok
    assert any("(fill)" in e for e in errs)


def test_verify_complete_accepts_bulk_defaults() -> None:
    doc = "# T\n\n" + render_section("a.py", use_placeholders=False) + "\n" + render_section(
        "b.py", use_placeholders=False
    )
    ok, errs = verify_complete(doc)
    assert ok
    assert not errs
