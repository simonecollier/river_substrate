#!/usr/bin/env python3
"""
Verify Garmin .RSD -> rectified GeoTIFF export completeness.

Expected tile count per beam is derived from the meta CSV's `chunk_id` values:
  - expected_ids = {chunk_id values present}
  - expected_count = len(expected_ids)

Actual tiles are read from:
  <rsd_dir>/<RSD_STEM>/_pingmapper_project/<RSD_STEM>/<beam>/rect_<product>/*.tif

This checks that for each beam (ss_port / ss_star):
  - all expected chunk_ids have a corresponding GeoTIFF tile
  - reports missing chunk indices and missing directories/files
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BEAMS = ("ss_port", "ss_star")


@dataclass
class BeamCheck:
    beam: str
    meta_csv: Path | None
    expected_ids: set[int]
    actual_ids: set[int]
    missing_ids: set[int]
    extra_ids: set[int]
    actual_files: list[Path]


def _discover_rsds(rsd_dir: Path) -> list[Path]:
    rsd_dir = rsd_dir.expanduser().resolve()
    return sorted(rsd_dir.glob("*.RSD")) + sorted(rsd_dir.glob("*.rsd"))


def _project_dir(rsd_dir: Path, rsd: Path) -> Path:
    return rsd_dir / rsd.stem / "_pingmapper_project" / rsd.stem


def _meta_csv_for_beam(proj_dir: Path, beam: str) -> Path | None:
    # PINGMapper uses B002 for ss_port and B003 for ss_star
    meta_dir = proj_dir / "meta"
    if beam == "ss_port":
        p = meta_dir / "B002_ss_port_meta.csv"
    elif beam == "ss_star":
        p = meta_dir / "B003_ss_star_meta.csv"
    else:
        return None
    return p if p.exists() else None


def _read_expected_chunk_ids(meta_csv: Path) -> set[int]:
    ids: set[int] = set()
    with meta_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None or "chunk_id" not in r.fieldnames:
            return set()
        for row in r:
            v = row.get("chunk_id", "")
            if v is None or v == "":
                continue
            try:
                # chunk_id can be float-like in some exports; parse robustly
                ids.add(int(float(v)))
            except Exception:
                continue
    return ids


_CHUNK_SUFFIX_RE = re.compile(r".*_(\d{1,5})\.tif$", re.IGNORECASE)


def _actual_tile_ids(tifs: Iterable[Path]) -> set[int]:
    ids: set[int] = set()
    for p in tifs:
        m = _CHUNK_SUFFIX_RE.match(p.name)
        if not m:
            continue
        ids.add(int(m.group(1)))
    return ids


def _collect_actual_tiles(proj_dir: Path, beam: str, product: str) -> list[Path]:
    return sorted((proj_dir / beam / f"rect_{product.lower()}").glob("*.tif"))


def check_one_rsd(rsd_dir: Path, rsd: Path, product: str) -> tuple[Path, list[BeamCheck]]:
    proj_dir = _project_dir(rsd_dir, rsd)
    checks: list[BeamCheck] = []
    for beam in BEAMS:
        meta_csv = _meta_csv_for_beam(proj_dir, beam)
        expected = _read_expected_chunk_ids(meta_csv) if meta_csv else set()
        actual_files = _collect_actual_tiles(proj_dir, beam, product) if proj_dir.exists() else []
        actual = _actual_tile_ids(actual_files)
        missing = expected - actual
        extra = actual - expected
        checks.append(
            BeamCheck(
                beam=beam,
                meta_csv=meta_csv,
                expected_ids=expected,
                actual_ids=actual,
                missing_ids=missing,
                extra_ids=extra,
                actual_files=actual_files,
            )
        )
    return proj_dir, checks


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify rectified GeoTIFF outputs from RSD conversions.")
    ap.add_argument("--rsd-dir", required=True, help="Directory containing .RSD files.")
    ap.add_argument("--product", default="wcr", choices=["wcr", "wcp"], help="Which rectified product to verify.")
    ap.add_argument("--csv-out", default="", help="Optional CSV report path.")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit on number of RSDs checked.")
    args = ap.parse_args()

    rsd_dir = Path(args.rsd_dir).expanduser().resolve()
    rsds = _discover_rsds(rsd_dir)
    if args.limit and args.limit > 0:
        rsds = rsds[: args.limit]

    if not rsds:
        raise SystemExit(f"No .RSD files found in {rsd_dir}")

    report_rows: list[dict[str, object]] = []

    ok = 0
    incomplete = 0
    missing_projects = 0

    for rsd in rsds:
        proj_dir, checks = check_one_rsd(rsd_dir, rsd, args.product)
        proj_exists = proj_dir.exists()
        if not proj_exists:
            missing_projects += 1

        rsd_ok = True
        for c in checks:
            # If there is no meta csv, we can't establish expected; treat as not-ok.
            if c.meta_csv is None:
                rsd_ok = False
            # If expected is empty but meta exists, treat as not-ok (likely failed run)
            if c.meta_csv is not None and not c.expected_ids:
                rsd_ok = False
            if c.missing_ids:
                rsd_ok = False

            report_rows.append(
                {
                    "rsd": rsd.name,
                    "project_dir": str(proj_dir),
                    "project_exists": proj_exists,
                    "beam": c.beam,
                    "meta_csv": str(c.meta_csv) if c.meta_csv else "",
                    "expected_tiles": len(c.expected_ids),
                    "actual_tiles": len(c.actual_ids),
                    "missing_tiles": len(c.missing_ids),
                    "missing_ids": ",".join(str(i) for i in sorted(c.missing_ids)),
                }
            )

        if rsd_ok:
            ok += 1
        else:
            incomplete += 1

    print(f"Checked {len(rsds)} RSD file(s) under {rsd_dir}")
    print(f"OK: {ok}")
    print(f"Incomplete/problem: {incomplete}")
    print(f"Missing project dirs: {missing_projects}")

    if args.csv_out:
        out = Path(args.csv_out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
            w.writeheader()
            w.writerows(report_rows)
        print(f"Wrote CSV report: {out}")


if __name__ == "__main__":
    main()

