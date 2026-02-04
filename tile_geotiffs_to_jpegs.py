#!/usr/bin/env python3
"""
Tile GeoTIFFs into georeferenced JPEGs + a tile index shapefile.

What this does
--------------
- Finds GeoTIFFs (*.tif, *.tiff) in a folder (recursively), OR runs on a single
  GeoTIFF as a quick test.
- Creates an output folder named `tiled_tiffs` inside the input folder.
- For each GeoTIFF `NAME.tif`, creates:

    <input_folder>/tiled_tiffs/NAME/
      - NAME_<rowLetters><colNumber>.jpg
          JPEG tile image (either palette-colored or grayscale, depending on JPEG_MODE)
      - NAME_<rowLetters><colNumber>.jgw
          Worldfile for the JPEG (pixel -> map coordinate transform)
      - NAME_<rowLetters><colNumber>.prj
          CRS definition for the JPEG (WKT)
      - NAME_tile_index.shp (+ .dbf/.shx/.prj...)
          Tile footprint polygons with attributes:
            ORG_TIFF  = original GeoTIFF filename
            TILE_NAME = tile code only (e.g. a14)
            JPG_FILE  = JPEG filename for that tile

How to run
----------
1) Install dependencies (recommended on Windows/macOS):
     conda install -c conda-forge rasterio geopandas shapely imageio numpy

2) Edit the CONFIGURATION section below:
   - INPUT_FOLDER
   - (optional) SINGLE_TIF_PATH for a one-file test
   - MAX_TILE_PX
   - JPEG_MODE ("palette" + COLOR_TABLE_CLR, or "grayscale")
   - If using palette mode, set COLOR_TABLE_CLR to the `.clr` file you provide

3) Run:
   - macOS/Linux:
       python /path/to/tile_geotiffs_to_jpegs.py
   - Windows (example):
       python C:\\path\\to\\tile_geotiffs_to_jpegs.py



Important: NO RESAMPLING
------------------------
Tiles are created by cropping pixel windows from the GeoTIFF. We do not change
the pixel size / resolution. The JPEG is written at the same pixel grid as the
source window, and the `.jgw` + `.prj` allow mapping pixels back to the original
GeoTIFF coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from rasterio.windows import transform as window_transform

import geopandas as gpd
from shapely.geometry import box

# ============================================================================
# CONFIGURATION (edit these variables only)
# ============================================================================

# Folder to search for GeoTIFFs (recursively). Output will be created inside it.
INPUT_FOLDER = "/Users/simone/Documents/River_Substrate_Project/GeoTiffs"

# Optional: set this to a single .tif path to run as a quick test, otherwise
# set to None to process all GeoTIFFs under INPUT_FOLDER.
SINGLE_TIF_PATH = None

# Name of the output folder created inside INPUT_FOLDER
TILES_FOLDER_NAME = "tiled_tiffs"

# Maximum output tile size in pixels (width and height)
MAX_TILE_PX = 5000

# JPEG output mode:
# - "palette": apply a fixed 0-255 -> RGB palette from a .clr file (extracted from singleband pseudocolor).
# - "grayscale": write grayscale JPEGs using a SINGLE stretch computed per GeoTIFF.
JPEG_MODE = "palette"  # "palette" or "grayscale"

# Optional: QGIS/GDAL color table exported as a .clr file (0-255 -> RGB(A)).
# Used when JPEG_MODE == "palette".
COLOR_TABLE_CLR = "/Users/simone/Documents/River_Substrate_Project/sonar_colourmap.clr"

# If your .clr specifies transparency (alpha=0), JPEG can't store alpha.
# We'll fill those pixels with this background color.
PALETTE_BG_RGB = (0, 0, 0)

# Grayscale mode settings (used when JPEG_MODE == "grayscale")
GRAY_BAND = 1  # 1-based band index
STRETCH_LO_PCT = 2.0
STRETCH_HI_PCT = 98.0
# For speed, compute per-GeoTIFF stretch from a downsampled preview (no effect
# on georeferencing; it only affects display scaling).
STATS_SAMPLE_MAX_DIM = 2048

# Which bands to use for the JPEG (1-based). Default band 1.
RGB_BANDS = (1, 1, 1)

# JPEG quality (1-100)... does not affect georeferencing.
JPEG_QUALITY = 95

# Skip writing tiles that have *no* valid data.
# Keep tiles if they contain even a single valid pixel.
USE_MASK_TO_SKIP_EMPTY = True
MIN_VALID_PIXELS = 1

# ============================================================================


@dataclass(frozen=True)
class TileRecord:
    tile_name: str
    tile_code: str
    original_tiff: str
    row: int
    col: int
    row_letter: str
    width_px: int
    height_px: int
    window_col_off: int
    window_row_off: int
    geometry: object  # shapely geometry


def _row_index_to_letters(idx0: int) -> str:
    """
    0 -> a, 1 -> b, ..., 25 -> z, 26 -> aa, 27 -> ab, ...
    """
    if idx0 < 0:
        raise ValueError("row index must be >= 0")
    n = idx0
    out = []
    while True:
        n, r = divmod(n, 26)
        out.append(chr(ord("a") + r))
        if n == 0:
            break
        n -= 1  # Excel-style carry
    return "".join(reversed(out))


def _worldfile_text_from_affine(t: rasterio.Affine) -> str:
    """
    Worldfile uses:
      line1: A (pixel size in x)
      line2: D (rotation about y)
      line3: B (rotation about x)
      line4: E (pixel size in y; usually negative)
      line5: C' (x coordinate of center of upper-left pixel)
      line6: F' (y coordinate of center of upper-left pixel)
    where the affine is:
      x = A*col + B*row + C
      y = D*col + E*row + F
    """
    A, B, C, D, E, F = t.a, t.b, t.c, t.d, t.e, t.f
    Cc = C + (A / 2.0) + (B / 2.0)
    Fc = F + (D / 2.0) + (E / 2.0)
    return "\n".join([f"{A:.12f}", f"{D:.12f}", f"{B:.12f}", f"{E:.12f}", f"{Cc:.12f}", f"{Fc:.12f}"]) + "\n"


def _to_uint8(arr: np.ndarray, nodata: int | float | None = None) -> np.ndarray:
    """
    Convert to uint8 for JPEG display with a robust 2-98% stretch.

    This does NOT resample; it only scales values to 0..255 for display.
    """
    a = arr.astype(np.float32)
    if nodata is not None:
        a = np.where(a == float(nodata), np.nan, a)
    lo = np.nanpercentile(a, 2)
    hi = np.nanpercentile(a, 98)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanmin(a)
        hi = np.nanmax(a)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    a = (a - lo) / (hi - lo)
    a = np.clip(a, 0, 1)
    # If nodata created NaNs, ensure stable uint8 conversion.
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    return (a * 255).astype(np.uint8)


def _to_uint8_with_range(arr: np.ndarray, lo: float, hi: float, nodata: int | float | None = None) -> np.ndarray:
    """
    Convert to uint8 using a fixed [lo, hi] range (e.g., computed once per GeoTIFF).
    """
    a = arr.astype(np.float32)
    if nodata is not None:
        a = np.where(a == float(nodata), np.nan, a)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    a = (a - float(lo)) / float(hi - lo)
    a = np.clip(a, 0, 1)
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    return (a * 255).astype(np.uint8)


def _read_rgb(src: rasterio.DatasetReader, window: Window, rgb_bands: tuple[int, int, int]) -> np.ndarray:
    bands = []
    for b in rgb_bands:
        if b < 1 or b > src.count:
            raise ValueError(f"Requested band {b} but source has {src.count} band(s)")
        band = src.read(b, window=window, boundless=False)
        bands.append(band)
    return np.stack(bands, axis=-1)  # (H, W, 3)


def _write_jpeg(path: Path, rgb_uint8: np.ndarray, quality: int) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, rgb_uint8, quality=int(quality))


def _load_gdal_clr(clr_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a GDAL/QGIS .clr color table into lookup tables.

    QGIS commonly exports lines like:
      value R G B A [label]

    We'll accept:
    - 4 columns: value R G B  (alpha defaults to 255)
    - 5+ columns: value R G B A (ignore anything after A)
    """
    rgb = np.zeros((256, 3), dtype=np.uint8)
    a = np.full((256,), 255, dtype=np.uint8)

    for raw in clr_path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 4:
            continue
        try:
            v = int(float(parts[0]))
            r = int(float(parts[1]))
            g = int(float(parts[2]))
            b = int(float(parts[3]))
            alpha = int(float(parts[4])) if len(parts) >= 5 else 255
        except ValueError:
            continue
        if not (0 <= v <= 255):
            continue
        rgb[v] = (np.uint8(r), np.uint8(g), np.uint8(b))
        a[v] = np.uint8(alpha)

    return rgb, a


def _apply_palette_uint8(gray_uint8: np.ndarray, rgb_lut: np.ndarray, a_lut: np.ndarray) -> np.ndarray:
    """
    Convert a uint8 single-band image to RGB using a lookup table.
    """
    out = rgb_lut[gray_uint8]  # (H, W, 3)
    if a_lut is not None:
        alpha = a_lut[gray_uint8]  # (H, W)
        if np.any(alpha == 0):
            bg = np.array(PALETTE_BG_RGB, dtype=np.uint8)
            out = out.copy()
            out[alpha == 0] = bg
    return out


def _compute_global_stretch_for_geotiff(
    src: rasterio.DatasetReader,
    band: int,
    lo_pct: float,
    hi_pct: float,
    sample_max_dim: int,
) -> tuple[float, float]:
    """
    Compute a display stretch range (lo, hi) ONCE per GeoTIFF from a downsampled preview.
    This avoids per-tile contrast shifts while staying fast for large rasters.
    """
    if band < 1 or band > src.count:
        raise ValueError(f"Requested band {band} but source has {src.count} band(s)")

    # Determine preview size (preserve aspect ratio; nearest for speed).
    w = int(src.width)
    h = int(src.height)
    if w <= 0 or h <= 0:
        return 0.0, 1.0
    scale = min(1.0, float(sample_max_dim) / float(max(w, h)))
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))

    preview = src.read(
        band,
        out_shape=(out_h, out_w),
        resampling=Resampling.nearest,
        masked=False,
    ).astype(np.float32)

    # Prefer dataset mask to exclude nodata areas from stats if available.
    nodata = src.nodata
    if USE_MASK_TO_SKIP_EMPTY:
        try:
            m = src.read_masks(1, out_shape=(out_h, out_w), resampling=Resampling.nearest)
            preview = np.where(m == 0, np.nan, preview)
        except Exception:
            pass
    if nodata is not None:
        preview = np.where(preview == float(nodata), np.nan, preview)

    lo = float(np.nanpercentile(preview, lo_pct))
    hi = float(np.nanpercentile(preview, hi_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(preview))
        hi = float(np.nanmax(preview))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0, 1.0
    return lo, hi


def _window_has_any_valid_data(src: rasterio.DatasetReader, win: Window) -> bool:
    """
    Conservative rule: keep tile if it contains ANY valid pixel.
    Prefer using the dataset mask; fallback to nodata test if needed.
    """
    if USE_MASK_TO_SKIP_EMPTY:
        try:
            m = src.read_masks(1, window=win)
            # `read_masks` returns 0 where invalid, 255 where valid (typical)
            valid = int(np.count_nonzero(m))
            return valid >= int(MIN_VALID_PIXELS)
        except Exception:
            pass

    # Fallback: nodata test on the first requested band
    nodata = src.nodata
    if nodata is None:
        # If nodata isn't defined and no mask, be conservative: keep tile.
        return True
    b = RGB_BANDS[0]
    arr = src.read(b, window=win, boundless=False)
    valid = int(np.count_nonzero(arr != nodata))
    return valid >= int(MIN_VALID_PIXELS)


def _iter_geotiffs(input_folder: Path) -> list[Path]:
    exts = {".tif", ".tiff"}
    out: list[Path] = []
    for p in input_folder.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        # Avoid re-processing our own output
        if TILES_FOLDER_NAME in p.parts:
            continue
        out.append(p)
    return sorted(out)


def process_one_geotiff(tif_path: Path, tiles_root: Path) -> None:
    out_dir = tiles_root / tif_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    shp_path = out_dir / f"{tif_path.stem}_tile_index.shp"

    records: list[TileRecord] = []

    mode = str(JPEG_MODE).strip().lower()
    if mode not in ("palette", "grayscale"):
        raise ValueError('JPEG_MODE must be "palette" or "grayscale"')

    rgb_lut: np.ndarray | None = None
    a_lut: np.ndarray | None = None
    if mode == "palette":
        if not COLOR_TABLE_CLR:
            raise ValueError('JPEG_MODE="palette" requires COLOR_TABLE_CLR to be set')
        clr_path = Path(COLOR_TABLE_CLR).expanduser().resolve()
        if not clr_path.exists():
            raise FileNotFoundError(f"COLOR_TABLE_CLR not found: {clr_path}")
        rgb_lut, a_lut = _load_gdal_clr(clr_path)

    with rasterio.open(str(tif_path)) as src:
        if src.crs is None:
            raise ValueError(f"GeoTIFF has no CRS: {tif_path}")

        # For grayscale mode, compute a single stretch for this GeoTIFF (not per tile).
        gray_lo: float | None = None
        gray_hi: float | None = None
        if mode == "grayscale":
            gray_lo, gray_hi = _compute_global_stretch_for_geotiff(
                src=src,
                band=int(GRAY_BAND),
                lo_pct=float(STRETCH_LO_PCT),
                hi_pct=float(STRETCH_HI_PCT),
                sample_max_dim=int(STATS_SAMPLE_MAX_DIM),
            )

        # Decide how many splits we need so each tile is <= MAX_TILE_PX,
        # then split as evenly as possible (still pixel-aligned; no resampling).
        n_rows = int(math.ceil(src.height / float(MAX_TILE_PX)))
        n_cols = int(math.ceil(src.width / float(MAX_TILE_PX)))
        tile_h = int(math.ceil(src.height / float(n_rows)))
        tile_w = int(math.ceil(src.width / float(n_cols)))

        for r in range(n_rows):
            row_letters = _row_index_to_letters(r)
            row_off = r * tile_h
            h = min(tile_h, src.height - row_off)
            if h <= 0:
                continue

            for c in range(n_cols):
                col_off = c * tile_w
                w = min(tile_w, src.width - col_off)
                if w <= 0:
                    continue

                win = Window(col_off=col_off, row_off=row_off, width=w, height=h)

                if not _window_has_any_valid_data(src, win):
                    continue

                tile_code = f"{row_letters}{c + 1}"
                tile_name = f"{tif_path.stem}_{tile_code}"
                jpg_path = out_dir / f"{tile_name}.jpg"
                jgw_path = out_dir / f"{tile_name}.jgw"
                prj_path = out_dir / f"{tile_name}.prj"

                # Read and write JPEG (no resampling: window read only)
                if mode == "palette" and rgb_lut is not None and a_lut is not None:
                    # Palette mode: use band 1 values directly (assumed 0-255)
                    band = src.read(1, window=win, boundless=False)
                    if band.dtype != np.uint8:
                        band = np.clip(band, 0, 255).astype(np.uint8)
                    rgb_u8 = _apply_palette_uint8(band, rgb_lut, a_lut)
                    _write_jpeg(jpg_path, rgb_u8, quality=JPEG_QUALITY)
                else:
                    # Grayscale mode: fixed stretch per GeoTIFF (consistent across tiles).
                    band = src.read(int(GRAY_BAND), window=win, boundless=False)
                    if band.dtype == np.uint8:
                        # Even if already uint8, apply the same stretch to match QGIS-like display.
                        gray = band.astype(np.float32)
                    else:
                        gray = band.astype(np.float32)
                    gray_u8 = _to_uint8_with_range(gray, float(gray_lo), float(gray_hi), nodata=src.nodata)
                    _write_jpeg(jpg_path, gray_u8, quality=JPEG_QUALITY)

                # Sidecars for georeferencing
                t_win = window_transform(win, src.transform)
                jgw_path.write_text(_worldfile_text_from_affine(t_win), encoding="utf-8")
                prj_path.write_text(src.crs.to_wkt(), encoding="utf-8")

                # Footprint polygon
                left, bottom, right, top = window_bounds(win, src.transform)
                geom = box(left, bottom, right, top)

                records.append(
                    TileRecord(
                        tile_name=tile_name,
                        tile_code=tile_code,
                        original_tiff=tif_path.name,
                        row=r,
                        col=c,
                        row_letter=row_letters,
                        width_px=int(w),
                        height_px=int(h),
                        window_col_off=int(col_off),
                        window_row_off=int(row_off),
                        geometry=geom,
                    )
                )

        if not records:
            print(f"[SKIP] No non-empty tiles found for: {tif_path}")
            return

        gdf = gpd.GeoDataFrame(
            {
                # NOTE: ESRI Shapefile field names are limited to 10 characters.
                # Some writers may truncate/rename these on disk, but the VALUES
                # are short and will not be truncated.
                "ORG_TIFF": [t.original_tiff for t in records],
                "TILE_NAME": [t.tile_code for t in records],
                # Helpful extras (all <=10 chars)
                "JPG_FILE": [f"{t.tile_name}.jpg" for t in records],
                "ROW": [t.row for t in records],
                "COL": [t.col for t in records],
                "W_PX": [t.width_px for t in records],
                "H_PX": [t.height_px for t in records],
                "COL_OFF": [t.window_col_off for t in records],
                "ROW_OFF": [t.window_row_off for t in records],
            },
            geometry=[t.geometry for t in records],
            crs=src.crs,
        )

        # Write shapefile (tile footprint index)
        gdf.to_file(str(shp_path))

    print(f"[OK] {tif_path.name}: wrote {len(records)} tile(s) to {out_dir}")


def main() -> None:
    input_folder = Path(INPUT_FOLDER).expanduser().resolve()
    if not input_folder.exists():
        raise FileNotFoundError(f"INPUT_FOLDER not found: {input_folder}")

    tiles_root = input_folder / TILES_FOLDER_NAME
    tiles_root.mkdir(parents=True, exist_ok=True)

    if SINGLE_TIF_PATH:
        tif = Path(SINGLE_TIF_PATH).expanduser().resolve()
        if not tif.exists():
            raise FileNotFoundError(f"SINGLE_TIF_PATH not found: {tif}")
        process_one_geotiff(tif, tiles_root)
        return

    tifs = _iter_geotiffs(input_folder)
    if not tifs:
        print(f"No GeoTIFFs found under: {input_folder}")
        return

    for tif in tifs:
        try:
            process_one_geotiff(tif, tiles_root)
        except Exception as e:
            print(f"[ERROR] {tif}: {e}")


if __name__ == "__main__":
    main()

