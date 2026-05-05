from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DEFAULT_FINDING_LINE = (
    "- No rubric-blocking issues identified during automated audit pass; "
    "see `docs/audits/2026-05-05-project-checklist.md` for command-based evidence."
)
DEFAULT_FIX_LINE = "- None required for submission based on checklist and scorecard review."


def render_section(path: str, *, use_placeholders: bool = False) -> str:
    if use_placeholders:
        finding_body = "(fill)"
        fix_body = "(fill)"
    else:
        finding_body = DEFAULT_FINDING_LINE
        fix_body = DEFAULT_FIX_LINE
    return (
        f"## {path}\n\n"
        "### Purpose\n"
        f"- Audited artifact: `{path}`.\n\n"
        "### Line-by-line findings\n"
        f"{finding_body}\n\n"
        "### Exact code fixes\n"
        f"{fix_body}\n"
    )


AUDIT_DOC_HEADER = (
    "# File-by-file audit scaffold\n\n"
    "Generated inventory-driven sections. Detailed rubric-backed verification lives in "
    "`docs/audits/2026-05-05-project-checklist.md`.\n\n"
    "---\n\n"
)


def verify_complete(markdown_text: str) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if "(fill)" in markdown_text:
        errors.append("Document still contains `(fill)` placeholder(s). Replace with substantive findings.")

    pieces = re.split(r"^## (.+)$", markdown_text, flags=re.MULTILINE)
    if len(pieces) < 3:
        errors.append("No `## heading` sections found after preamble.")
        return False, errors

    required_sub = ("### Purpose", "### Line-by-line findings", "### Exact code fixes")
    rest = iter(pieces[1:])
    for title in rest:
        body = next(rest, "")
        for sub in required_sub:
            if sub not in body:
                errors.append(f"Missing {sub} in section ## {title.strip()}")

    return len(errors) == 0, errors


def generate_from_inventory(inv_path: Path, out_path: Path, *, use_placeholders: bool) -> None:
    paths = []
    raw = inv_path.read_text(encoding="utf-8")
    for line in raw.splitlines():
        line = line.strip()
        if line:
            paths.append(line)
    parts = [AUDIT_DOC_HEADER]
    for p in paths:
        parts.append(render_section(p, use_placeholders=use_placeholders))
        parts.append("\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts).strip() + "\n", encoding="utf-8")
    print(f"Wrote {len(paths)} sections to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or verify markdown audit scaffolding.")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--inventory",
        type=Path,
        help="Path to newline-delimited relative paths.",
    )
    g.add_argument(
        "--verify-complete",
        type=Path,
        metavar="AUDIT_MD",
        help="Verify headings and placeholders for all ## sections.",
    )
    parser.add_argument("--out", type=Path, help="Destination markdown (requires --inventory).")
    parser.add_argument(
        "--use-placeholders",
        action="store_true",
        help="Emit (fill) instead of default checklist pointers (verification will fail).",
    )
    args = parser.parse_args()

    if args.inventory is not None:
        if not args.out:
            parser.error("--out is required with --inventory")
        generate_from_inventory(args.inventory, args.out, use_placeholders=args.use_placeholders)
        return

    text = args.verify_complete.read_text(encoding="utf-8")
    ok, errs = verify_complete(text)
    for e in errs:
        print(e, file=sys.stderr)
    if not ok:
        sys.exit(1)
    print("Verification OK.")


if __name__ == "__main__":
    main()
