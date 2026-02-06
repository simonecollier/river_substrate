#!/usr/bin/env python3
"""
Export simple *visual* quicklooks (PNG tiles) from Garmin `.RSD` files.

This is meant for cases where the `.RSD` contains downscan beams (`ds_*`) and
there is no side-scan (`ss_port/ss_star`) to rectify into GeoTIFFs.

What it does
------------
- Creates a PINGMapper project for the `.RSD`
- Exports unrectified sonar tiles (`*.png`) so you can visually inspect the data

Outputs
-------
<output>/<RSD_STEM>/_pingmapper_project/<RSD_STEM>/
  ds_hifreq/wcp/*.png
  ds_vhifreq/wcp/*.png

If side-scan beams exist, those will also be exported (as PNG tiles), but
rectified GeoTIFFs are a separate step.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import inspect
import os
import shutil
import sys
from pathlib import Path


def _ensure_vendored_pingmapper_on_path() -> None:
    here = Path(__file__).resolve()
    vendored_pkg_dir = here.parent / "PINGMapper"
    if vendored_pkg_dir.exists():
        sys.path.insert(0, str(vendored_pkg_dir))


def _patch_pandas_fillna_method_compat() -> None:
    try:
        import pandas as pd  # noqa: F401
        from pandas.core.generic import NDFrame  # type: ignore

        if "method" in inspect.signature(NDFrame.fillna).parameters:
            return

        _orig = NDFrame.fillna

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
            return _orig(self, value=value, **kwargs)

        NDFrame.fillna = _fillna_compat  # type: ignore[assignment]
    except Exception:
        return


def _discover_rsds(input_path: Path) -> list[Path]:
    p = input_path.expanduser().resolve()
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"Input path not found: {p}")
    return sorted(p.glob("*.RSD")) + sorted(p.glob("*.rsd"))


def _pingmapper_project_dir(out_dir: Path, rsd_path: Path) -> Path:
    return out_dir / "_pingmapper_project" / rsd_path.stem


def _avoid_model_downloads(out_dir: Path) -> None:
    """
    PINGMapper tries to download models into `${CONDA_PREFIX}/pingmapper_config/...`
    even when we're only exporting *visual* tiles. In sandboxed runs, that path can
    be non-writable. Point CONDA_PREFIX to a writable folder and create the expected
    model directory so PINGMapper skips downloads.
    """
    fake_prefix = (out_dir / "_pingmapper_conda_prefix").resolve()
    model_dir = fake_prefix / "pingmapper_config" / "models" / "PINGMapperv2.0_SegmentationModelsv1.0"
    model_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CONDA_PREFIX"] = str(fake_prefix)


def export_quicklooks_one(
    rsd_path: Path,
    output_root: Path,
    *,
    nchunk: int,
    tile_ext: str,
    overwrite: bool,
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
            raise FileExistsError(f"Output exists: {proj_dir} (use --overwrite)")
    proj_dir.mkdir(parents=True, exist_ok=True)

    # Reduce matplotlib/font cache warnings by forcing writable cache dirs.
    os.environ.setdefault("MPLCONFIGDIR", str((out_dir / "_mplconfig").resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str((out_dir / "_xdg_cache").resolve()))

    _patch_pandas_fillna_method_compat()
    _ensure_vendored_pingmapper_on_path()
    _avoid_model_downloads(out_dir)

    from pingmapper.main_readFiles import read_master_func  # noqa: E402

    logs_dir = proj_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    logfilename = str(logs_dir / f"rsd_quicklook_{ts}.txt")
    script_param = [str(Path(__file__).resolve()), f"rsd_export_quicklooks_{ts}.py"]

    # We set wcp=True to export *unrectified* tiles for any beams that support it
    # (including ds_* beams). This is for visualization, so "no side-scan" is OK.
    _ = read_master_func(
        logfilename=logfilename,
        project_mode=1,
        script=script_param,
        inFile=str(rsd_path),
        projDir=str(proj_dir),
        nchunk=int(nchunk),
        tileFile=str(tile_ext),
        wcp=True,
        wcr=False,
        wco=False,
        wcm=False,
        pred_sub=0,
        map_sub=0,
        export_poly=False,
        pltSubClass=False,
        map_predict=0,
    )

    return proj_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Export PNG quicklooks from Garmin .RSD files.")
    ap.add_argument("--input", required=True, help="A .RSD file or directory containing .RSD files.")
    ap.add_argument(
        "--output",
        default="",
        help="Output folder root (default: same directory as the input file(s)).",
    )
    ap.add_argument("--nchunk", type=int, default=500, help="Pings per chunk / tile width (default: 500).")
    ap.add_argument("--tile-ext", default=".png", help="Image extension for tiles (default: .png).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit on number of files processed.")
    args = ap.parse_args()

    rsds = _discover_rsds(Path(args.input))
    if args.limit and args.limit > 0:
        rsds = rsds[: args.limit]
    if not rsds:
        raise FileNotFoundError(f"No .RSD files found under: {args.input}")

    if args.output:
        out_root = Path(args.output).expanduser().resolve()
    else:
        # default to same directory as inputs
        out_root = rsds[0].parent.resolve()

    print(f"Found {len(rsds)} .RSD file(s).")
    print(f"Output root: {out_root}")

    for rsd in rsds:
        print(f"\nExporting quicklooks for: {rsd.name}")
        proj_dir = export_quicklooks_one(
            rsd,
            out_root,
            nchunk=args.nchunk,
            tile_ext=args.tile_ext,
            overwrite=args.overwrite,
        )
        print(f"Done: {proj_dir}")


if __name__ == "__main__":
    main()

