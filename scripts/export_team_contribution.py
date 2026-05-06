from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

OUTPUT_PATH = Path("docs") / "team_contribution.csv"


def collect_commit_counts() -> list[tuple[str, str, int]]:
    result = subprocess.run(
        ["git", "shortlog", "-sne", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    rows: list[tuple[str, str, int]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        count_part, author_part = line.split("\t", maxsplit=1)
        count = int(count_part.strip())
        name, email = author_part.rsplit(" <", maxsplit=1)
        rows.append((name.strip(), email.rstrip(">").strip(), count))
    return rows


def export_team_contribution(output_path: Path = OUTPUT_PATH) -> int:
    rows = collect_commit_counts()
    rows.sort(key=lambda row: (-row[2], row[0].casefold(), row[1].casefold()))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["author", "email", "commit_count"])
        writer.writerows(rows)
    print(f"exported {len(rows)} authors to {output_path}")
    return len(rows)


def main() -> int:
    try:
        export_team_contribution()
    except (subprocess.CalledProcessError, ValueError) as exc:
        print(f"export_team_contribution: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
