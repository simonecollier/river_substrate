#!/usr/bin/env python3
"""
Batch convert Garmin `.RSD` files to rectified GeoTIFFs, then flatten outputs.

Why
----
PINGMapper exports rectified tiles into per-project subfolders. GIS workflows
often prefer a single folder containing all tiles. This script:
  1) runs the existing `convert_rsd_to_geotiff.py` conversion per `.RSD`
  2) copies the resulting `*.tif` tiles into a single "flat" directory

It does NOT delete the PINGMapper projects by default.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import shutil
import sys
from glob import glob
from pathlib import Path


def _discover_rsds(input_dir: Path) -> list[Path]:
    d = input_dir.expanduser().resolve()
    if not d.is_dir():
        raise FileNotFoundError(d)
    return sorted(d.glob("*.RSD")) + sorted(d.glob("*.rsd"))


def _collect_tiles(proj_dir: Path, product: str) -> list[Path]:
    product = product.lower()
    tiles: list[Path] = []
    for beam in ("ss_port", "ss_star"):
        tiles.extend(Path(p) for p in glob(str(proj_dir / beam / f"rect_{product}" / "*.tif")))
    return sorted(tiles)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch convert .RSD to rectified GeoTIFFs and flatten into one folder."
    )
    ap.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing .RSD files.",
    )
    ap.add_argument(
        "--flat-dir",
        default="",
        help="Directory to receive all output .tif tiles (default: <input-dir>/geotiff_tiles_flat).",
    )
    ap.add_argument(
        "--product",
        default="wcr",
        choices=["wcr", "wcp", "both"],
        help="Which rectified product(s) to export/copy (default: wcr).",
    )
    ap.add_argument("--nchunk", type=int, default=500, help="PINGMapper nchunk (default: 500).")
    ap.add_argument("--mosaic", default="none", choices=["none", "vrt", "tif"], help="Also build mosaics.")
    ap.add_argument("--use-gpu", action="store_true", help="Ask PINGMapper to use GPU where supported.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite per-file project outputs if present.")
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip work for existing outputs. Specifically: "
            "(1) if a per-RSD PINGMapper project already exists, skip reconversion; "
            "(2) if a flat output tile already exists, skip copying it."
        ),
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of files processed (default: 0 means no limit).",
    )

    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    rsds = _discover_rsds(input_dir)
    if args.limit and args.limit > 0:
        rsds = rsds[: args.limit]
    if not rsds:
        raise FileNotFoundError(f"No .RSD files found in {input_dir}")

    flat_dir = Path(args.flat_dir).expanduser().resolve() if args.flat_dir else (input_dir / "geotiff_tiles_flat")
    flat_dir.mkdir(parents=True, exist_ok=True)

    # Run each file in a fresh subprocess to avoid memory buildup across files.
    converter_script = (Path(__file__).resolve().parent / "convert_rsd_to_geotiff.py").resolve()
    if not converter_script.exists():
        raise FileNotFoundError(f"Missing converter script: {converter_script}")

    print(f"Found {len(rsds)} .RSD file(s).")
    print(f"Flat output directory: {flat_dir}")

    copied = 0
    for rsd in rsds:
        print(f"\n=== {rsd.name} ===")
        proj_dir = input_dir / rsd.stem / "_pingmapper_project" / rsd.stem

        # If requested, skip conversion when the project already exists
        if args.skip_existing and proj_dir.exists() and not args.overwrite:
            print(f"Skipping conversion (project exists): {proj_dir}")
        else:
            # Use the same interpreter running this script (your ping env),
            # so dependencies (gdal, pingmapper, pingverter, etc.) match.
            python_exe = sys.executable
            cmd = [
                python_exe,
                os.fspath(converter_script),
                "--input",
                os.fspath(rsd),
                "--output",
                os.fspath(input_dir),
                "--product",
                args.product,
                "--nchunk",
                str(args.nchunk),
                "--mosaic",
                args.mosaic,
            ]
            if args.use_gpu:
                cmd.append("--use-gpu")
            if args.overwrite:
                cmd.append("--overwrite")

            # Keep the converter from trying to download models to a non-writable prefix
            env = dict(os.environ)
            env.setdefault("PYTHONUNBUFFERED", "1")

            # Execute conversion in a fresh process to release memory between files.
            subprocess.run(cmd, check=True, env=env)

        products = [args.product] if args.product in {"wcr", "wcp"} else ["wcr", "wcp"]
        for prod in products:
            for tif in _collect_tiles(Path(proj_dir), prod):
                # Ensure unique filenames in the flat directory
                out_name = tif.name
                out_path = flat_dir / out_name
                if out_path.exists() and args.skip_existing:
                    continue
                shutil.copy2(tif, out_path)
                copied += 1

        print(f"Project: {proj_dir}")

    print(f"\nDone. Copied {copied} GeoTIFF tile(s) into: {flat_dir}")


if __name__ == "__main__":
    main()

