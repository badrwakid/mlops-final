#!/usr/bin/env python3
"""Fetch UCI Bike Sharing hour.csv into data/raw/ (POSIX-friendly; mirrors fetch_uci_hour_csv.ps1)."""

from __future__ import annotations

import hashlib
import pathlib
import tempfile
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEST_DIR = ROOT / "data" / "raw"
DEST = DEST_DIR / "hour.csv"
URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
EXPECTED_MD5 = "d50bd5a6f55131e72a7bedc334e2fce1"


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        zpath = tmp / "bike-sharing-dataset.zip"
        urllib.request.urlretrieve(URL, zpath)
        with zipfile.ZipFile(zpath) as zf:
            member = next(
                (n for n in zf.namelist() if not n.endswith("/") and n.lower().endswith("hour.csv")),
                None,
            )
            if member is None:
                raise RuntimeError("hour.csv not found inside UCI zip.")
            data = zf.read(member)

    md5_hex = hashlib.md5(data).hexdigest()
    if md5_hex.lower() != EXPECTED_MD5.lower():
        raise RuntimeError(f"MD5 mismatch (got {md5_hex}, expected {EXPECTED_MD5}).")

    DEST.write_bytes(data)
    print(f"OK: {DEST} ({len(data)} bytes, md5 matches hour.csv.dvc)")


if __name__ == "__main__":
    main()
