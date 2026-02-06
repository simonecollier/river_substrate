#!/usr/bin/env python3
"""
Scan Garmin `.RSD` files and report which channels/beams they contain.

This uses `pingverter.converter.gar2pingmapper()` to generate metadata and then
reads the beam/channel info from the created PINGMapper-style project.

Why this exists
---------------
PINGMapper can only export *rectified GeoTIFFs* for side-scan beams
(`ss_port`, `ss_star`). Many Garmin `.RSD` recordings contain only downscan
channels (`ds_*`), which can be visualized but not rectified to side-scan GeoTIFFs.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path


def _patch_pandas_fillna_method_compat() -> None:
    """
    pingverter currently calls `fillna(method=...)`, which is removed in newer pandas.
    Provide a runtime shim so it works without pinning pandas.
    """
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


def _discover_rsds(path: Path) -> list[Path]:
    p = path.expanduser().resolve()
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(p)
    return sorted(p.glob("*.RSD")) + sorted(p.glob("*.rsd"))


def main() -> None:
    ap = argparse.ArgumentParser(description="List channels/beams contained in Garmin .RSD files.")
    ap.add_argument("--input", required=True, help="A .RSD file or a directory containing .RSD files.")
    ap.add_argument(
        "--probe-out",
        default=str(Path.cwd() / "_tmp_rsd_probe"),
        help="Where to write temporary probe projects (default: ./_tmp_rsd_probe).",
    )
    ap.add_argument(
        "--csv",
        default="",
        help="Optional path to write a CSV summary (default: not written).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of files to scan (default: 0 means no limit).",
    )

    args = ap.parse_args()

    _patch_pandas_fillna_method_compat()

    import pandas as pd  # noqa: E402
    from pingverter.converter import gar2pingmapper  # noqa: E402

    rsds = _discover_rsds(Path(args.input))
    if args.limit and args.limit > 0:
        rsds = rsds[: args.limit]

    probe_root = Path(args.probe_out).expanduser().resolve()
    probe_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for p in rsds:
        proj = probe_root / p.stem
        proj.mkdir(parents=True, exist_ok=True)

        try:
            sonar_obj = gar2pingmapper(str(p), str(proj), nchunk=500, tempC=10, exportUnknown=False)
            meta_csv = Path(sonar_obj.metaDir) / "All-Garmin-Sonar-MetaData.csv"
            df = pd.read_csv(meta_csv, low_memory=False)
            chans = sorted(df["channel_id"].dropna().unique().tolist()) if "channel_id" in df.columns else []

            beam_names = sorted({m.get("beamName", "") for m in getattr(sonar_obj, "beamMeta", {}).values()})
            has_side_scan = any(b in {"ss_port", "ss_star"} for b in beam_names)

            rows.append(
                {
                    "file": str(p),
                    "channels": ",".join(map(str, chans)),
                    "beams": ",".join([b for b in beam_names if b]),
                    "has_side_scan": bool(has_side_scan),
                }
            )
        except Exception as e:
            rows.append(
                {
                    "file": str(p),
                    "channels": "",
                    "beams": "",
                    "has_side_scan": False,
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    # Print a compact summary
    print(f"Scanned {len(rows)} file(s).")
    with_side_scan = [r for r in rows if r.get("has_side_scan")]
    print(f"Files with side-scan (ss_port/ss_star): {len(with_side_scan)}")
    for r in rows[:50]:
        print(f"- {Path(str(r['file'])).name}: beams=[{r.get('beams','')}] channels=[{r.get('channels','')}]")

    # Optional CSV output
    if args.csv:
        out_csv = Path(args.csv).expanduser().resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\nWrote CSV: {out_csv}")


if __name__ == "__main__":
    main()

