#!/usr/bin/env python3
"""
Convert Garmin sonar `.RSD` recordings to *rectified* GeoTIFF tiles.

This repo's `.RSD` workflow is handled by PINGMapper + PINGVerter:
  1) Read the `.RSD` into a PINGMapper project (creates metadata + pickles)
  2) Rectify side-scan channels into georeferenced GeoTIFF tiles (WCR/WCP)

Outputs (per input file) are written under:
  <output>/<RSD_STEM>/_pingmapper_project/<RSD_STEM>/
    ss_port/rect_wcr/*.tif   (if --product wcr or both)
    ss_star/rect_wcr/*.tif
    ss_port/rect_wcp/*.tif   (if --product wcp or both)
    ss_star/rect_wcp/*.tif
    *_wcr_mosaic.(tif|vrt)   (if --mosaic tif|vrt and wcr requested)
    *_wcp_mosaic.(tif|vrt)   (if --mosaic tif|vrt and wcp requested)

Notes
-----
- This script uses the *vendored* PINGMapper copy in `river_substrate/PINGMapper/`.
- Some environments (notably with newer pandas) can break older PINGVerter.
  We apply a tiny runtime compatibility shim for `fillna(method=...)`.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import shutil
import sys
from glob import glob
from pathlib import Path


def _ensure_vendored_pingmapper_on_path() -> None:
    here = Path(__file__).resolve()
    vendored_pkg_dir = here.parent / "PINGMapper"
    if vendored_pkg_dir.exists():
        sys.path.insert(0, str(vendored_pkg_dir))


def _patch_pandas_fillna_method_compat() -> None:
    """
    PINGVerter (currently) calls `Series.fillna(method="ffill")`, which was removed
    in newer pandas versions. We provide a small shim so existing code keeps working.
    """
    try:
        import inspect
        import pandas as pd  # noqa: F401
        from pandas.core.generic import NDFrame  # type: ignore

        if "method" in inspect.signature(NDFrame.fillna).parameters:
            return  # pandas still supports method=

        _orig_fillna = NDFrame.fillna

        def _fillna_compat(self, value=None, *, method=None, **kwargs):  # type: ignore[no-redef]
            if method is not None and value is None:
                m = str(method).lower()
                axis = kwargs.get("axis", None)
                limit = kwargs.get("limit", None)
                if m in {"ffill", "pad"}:
                    return self.ffill(axis=axis, limit=limit)
                if m in {"bfill", "backfill"}:
                    return self.bfill(axis=axis, limit=limit)
                raise TypeError(f"fillna() got an unexpected method={method!r}")
            return _orig_fillna(self, value=value, **kwargs)

        NDFrame.fillna = _fillna_compat  # type: ignore[assignment]
    except Exception:
        # If pandas isn't installed yet, or internals differ, just skip.
        return


def _patch_pingverter_two_channel_side_scan() -> None:
    """
    PINGVerter currently assigns Garmin beams using a simple heuristic:
      - 4 channels => ds_hifreq + ds_vhifreq + ss_port + ss_star
      - 2 channels => ds_hifreq + ds_vhifreq

    Some Garmin recordings contain *only* side-scan (port+star) as 2 channels.
    In that case, PINGMapper will (incorrectly) report "No side-scan channels".

    We patch `_splitBeamsToCSV()` to auto-detect the 2-channel port/star case by
    looking at `port_star_elem_angle`:
      - angle < 90 => ss_port
      - angle > 90 => ss_star
    """
    try:
        import numpy as np  # noqa: F401

        import pingverter.garmin_class as gc  # type: ignore

        _orig = gc.gar._splitBeamsToCSV

        def _splitBeamsToCSV_patched(self):  # type: ignore[no-redef]
            df = self.header_dat

            # Determine how many channels exist
            try:
                dfBeams = df.drop_duplicates(subset=["channel_id", "F"])
            except Exception:
                dfBeams = df.drop_duplicates(subset=["channel_id"])

            if len(dfBeams) == 2 and "port_star_elem_angle" in df.columns:
                # Use median element angle to infer left/right
                med = (
                    df.groupby("channel_id")["port_star_elem_angle"]
                    .median()
                    .dropna()
                    .to_dict()
                )

                if len(med) == 2:
                    items = sorted(med.items(), key=lambda kv: kv[1])
                    (cid_a, ang_a), (cid_b, ang_b) = items

                    # Require one angle to be clearly on each side of 90 degrees.
                    if ang_a < 90 < ang_b:
                        # Force these two channels to be treated as side-scan
                        # by temporarily faking a 4-channel-like mapping inside
                        # the original method: we replace dfBeams length logic by
                        # directly setting beam_set here, reusing the rest of the
                        # original implementation below.
                        beam_set = {
                            int(cid_a): ("ss_port", 2),
                            int(cid_b): ("ss_star", 3),
                        }

                        # Re-implement the bottom half of the original method
                        # (writing per-beam CSVs + populating self.beamMeta)
                        self.beamMeta = beamMeta = {}

                        for beam, group in df.groupby("channel_id"):
                            meta = {}
                            humBeamName, humBeamint = beam_set[int(beam)]
                            humBeam = "B00" + str(humBeamint)
                            meta["beamName"] = humBeamName
                            meta["beam"] = humBeam
                            group = group.copy()
                            group["beam"] = humBeamint

                            meta["sonFile"] = self.sonFile

                            cols2Drop = ["magic_number"]
                            cols = group.columns
                            cols2Drop += [c for c in cols if "fp" in c]
                            cols2Drop += [c for c in cols if "SP" in c]
                            cols2Drop += [c for c in cols if "su" in c]
                            for c in cols2Drop:
                                if c in group.columns:
                                    group.drop(columns=[c], inplace=True)

                            group = self._getChunkID(group)

                            outCSV = os.path.join(self.metaDir, humBeam + "_" + humBeamName + "_meta.csv")
                            group.to_csv(outCSV, index=False, float_format="%.14f")
                            meta["metaCSV"] = outCSV
                            meta["metaMETA"] = outCSV.replace(".csv", ".meta")
                            beamMeta[humBeam] = meta

                        return

            # Fall back to upstream behavior
            return _orig(self)

        gc.gar._splitBeamsToCSV = _splitBeamsToCSV_patched  # type: ignore[assignment]
    except Exception:
        return


def _discover_rsds(input_path: Path) -> list[Path]:
    p = input_path.expanduser().resolve()
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"Input path not found: {p}")

    rsds = sorted(p.glob("*.RSD")) + sorted(p.glob("*.rsd"))
    return rsds


def _pingmapper_project_dir(out_dir: Path, rsd_path: Path) -> Path:
    # Keep PINGMapper internals separate from other outputs
    return out_dir / "_pingmapper_project" / rsd_path.stem


def _maybe_avoid_model_download(out_dir: Path, allow_model_download: bool) -> None:
    """
    PINGMapper's `read_master_func()` tries to download models if they aren't present.
    For *pure conversion to GeoTIFF*, we don't need those models, and downloads can
    fail on restricted networks. To avoid that, we run with a temporary CONDA_PREFIX
    containing an "existing" model directory.
    """
    if allow_model_download:
        return

    fake_prefix = (out_dir / "_pingmapper_conda_prefix").resolve()
    model_dir = fake_prefix / "pingmapper_config" / "models" / "PINGMapperv2.0_SegmentationModelsv1.0"
    model_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CONDA_PREFIX"] = str(fake_prefix)


def _iter_rectified_tiles(proj_dir: Path, product: str) -> list[Path]:
    product = product.lower()
    if product not in {"wcr", "wcp"}:
        raise ValueError("product must be 'wcr' or 'wcp'")

    pattern = f"rect_{product}"
    tiles: list[Path] = []
    for beam in ("ss_port", "ss_star"):
        tiles.extend(Path(p) for p in glob(str(proj_dir / beam / pattern / "*.tif")))
    return sorted(tiles)


def convert_one(
    rsd_path: Path,
    output_root: Path,
    *,
    product: str,
    nchunk: int,
    thread_cnt: int,
    mosaic: str,
    use_gpu: bool,
    overwrite: bool,
    allow_model_download: bool,
) -> Path:
    rsd_path = rsd_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    out_dir = output_root / rsd_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    proj_dir = _pingmapper_project_dir(out_dir, rsd_path)
    if proj_dir.exists():
        if overwrite:
            shutil.rmtree(proj_dir)
        else:
            raise FileExistsError(
                f"PINGMapper project already exists at {proj_dir}. Use --overwrite to replace it."
            )
    proj_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = proj_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    logfilename = str(logs_dir / f"convert_rsd_to_geotiff_{ts}.txt")

    # Ensure vendored PINGMapper is importable and patch pandas compatibility.
    _patch_pandas_fillna_method_compat()
    _ensure_vendored_pingmapper_on_path()
    _maybe_avoid_model_download(out_dir, allow_model_download=allow_model_download)
    _patch_pingverter_two_channel_side_scan()

    from pingmapper.main_readFiles import read_master_func  # noqa: E402
    from pingmapper.main_rectify import rectify_master_func  # noqa: E402

    # PINGMapper copies the running script into its meta folder; provide a stable "name".
    script_param = [str(Path(__file__).resolve()), f"convert_rsd_to_geotiff_{ts}.py"]

    # --- Step 1: read the RSD into a PINGMapper project ---
    read_ok = read_master_func(
        logfilename=logfilename,
        project_mode=1,
        script=script_param,
        inFile=str(rsd_path),
        projDir=str(proj_dir),
        nchunk=int(nchunk),
        threadCnt=int(thread_cnt),
        USE_GPU=bool(use_gpu),
        # Keep exports minimal (we only want rectified GeoTIFFs):
        wcp=False,
        wcr=False,
        pred_sub=0,
        map_sub=0,
        export_poly=False,
        pltSubClass=False,
        map_predict=0,
    )
    if not read_ok:
        raise RuntimeError("PINGMapper read step failed (no side-scan channels available).")

    # --- Step 2: rectify to GeoTIFF tiles ---
    product_l = product.lower()
    rect_wcr = product_l in {"wcr", "both"}
    rect_wcp = product_l in {"wcp", "both"}

    mosaic_l = mosaic.lower()
    mosaic_val = {"none": 0, "tif": 1, "vrt": 2}.get(mosaic_l)
    if mosaic_val is None:
        raise ValueError("mosaic must be one of: none, tif, vrt")

    rectify_master_func(
        logfilename=logfilename,
        project_mode=2,
        script=script_param,
        inFile=str(rsd_path),
        projDir=str(proj_dir),
        nchunk=int(nchunk),
        threadCnt=int(thread_cnt),
        USE_GPU=bool(use_gpu),
        rect_wcr=rect_wcr,
        rect_wcp=rect_wcp,
        mosaic=int(mosaic_val),
        pred_sub=0,
        map_sub=0,
        export_poly=False,
        pltSubClass=False,
        map_predict=0,
    )

    # Basic sanity check: did we get any GeoTIFF tiles?
    found_tiles: list[Path] = []
    if rect_wcr:
        found_tiles.extend(_iter_rectified_tiles(proj_dir, "wcr"))
    if rect_wcp:
        found_tiles.extend(_iter_rectified_tiles(proj_dir, "wcp"))
    if not found_tiles:
        raise RuntimeError(
            "Rectification completed, but no GeoTIFF tiles were found. "
            f"Expected outputs under {proj_dir}/ss_port/rect_* and {proj_dir}/ss_star/rect_*."
        )

    return proj_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Garmin .RSD sonar files into rectified GeoTIFF tiles (PINGMapper)."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a single .RSD file or a directory containing .RSD files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory root. Each input file writes to <output>/<RSD_STEM>/...",
    )
    parser.add_argument(
        "--product",
        default="wcr",
        choices=["wcr", "wcp", "both"],
        help="Which rectified product to export (default: wcr).",
    )
    parser.add_argument(
        "--nchunk",
        type=int,
        default=500,
        help="PINGMapper chunk size (pings per chunk). Controls tile sizes (default: 500).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="How many CPU workers PINGMapper can use (default: 1).",
    )
    parser.add_argument(
        "--mosaic",
        default="none",
        choices=["none", "vrt", "tif"],
        help="Also build a mosaic (default: none).",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Ask PINGMapper to use GPU where supported (default: off).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite any existing PINGMapper project output for an input file.",
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help=(
            "Allow PINGMapper to download its ML models if missing. "
            "By default, this script avoids downloads because conversion doesn't need them."
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_root = Path(args.output)
    rsds = _discover_rsds(input_path)
    if not rsds:
        raise FileNotFoundError(f"No .RSD files found under: {input_path.expanduser().resolve()}")

    print(f"Found {len(rsds)} .RSD file(s).")

    for rsd in rsds:
        print(f"\nConverting: {rsd}")
        proj_dir = convert_one(
            rsd,
            output_root,
            product=args.product,
            nchunk=args.nchunk,
            thread_cnt=args.threads,
            mosaic=args.mosaic,
            use_gpu=args.use_gpu,
            overwrite=args.overwrite,
            allow_model_download=args.allow_model_download,
        )
        print(f"Done: {proj_dir}")


if __name__ == "__main__":
    main()

