"""
Measure the sky background brightness from a specified night.

For each image:
  1. Predict star positions using the fisheye correction model
  2. Centroid to get exact positions and measure star brightness
  3. Fit per-image zero_point: vmag = -2.5 * log10(adjusted_brightness) + zero_point
  4. Apply median filter to the sky region (ground masked out)
  5. Measure median pixel brightness per 25x25 pixel grid cell
  6. Convert to vmag_per_pixel using this image's zero_point

Outputs:
  {direction}_{date}_cells.csv  — source, az, el, vmag_per_pixel (one row per cell per image)
  {direction}_{date}_images.csv — source, zero_point, centroid_rate
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
from astropy.table import Table
from matplotlib.path import Path as MplPath
from scipy.ndimage import median_filter
from tqdm import tqdm

from config.settings import PROJECT_ROOT
from data_management.sky_image import SkyImage
from data_management.starlist import StarList
from models.centroid import centroid_starlist
from models.fisheye_correction import FisheyeCorrectionModel
from pipelines.calibrate import load_brightness_calibration
from pipelines.images_to_brightness import fit_vmag_regression

DIRECTIONS = ["North", "East", "South", "West"]

# Ground/horizon mask polygons per direction (pixel coordinates).
# Points define a polygon covering the ground; pixels inside are masked out.
GROUND_MASK_POLYGONS = {
    "West": [
        [0, 2822], [0, 130], [160, 146], [429, 420], [558, 764], [635, 1067],
        [632, 1366], [611, 1787], [747, 1791], [834, 1775],
        [933, 1779], [942, 1667], [966, 1667], [966, 1361], [1440, 1361], [1440, 1684],
        [1528, 1688], [1532, 1845], [2440, 1849], [3096, 1816], [3662, 1776], [4141, 1696], [4144, 2822],
    ],
    "East": [
        [0, 2822], [0, 700], [100, 760], [200, 1350], [100, 1840], [92, 1917],
        [404, 1976], [922, 2058], [1262, 2089], [1542, 2112], [1897, 1972],
        [1979, 1933], [2084, 1832], [2182, 1839], [2252, 1929], [2369, 1956],
        [2478, 1972], [2532, 1945], [2587, 1816], [2642, 1789], [2704, 1816],
        [2720, 1937], [2790, 1933], [2844, 1890], [2895, 1917], [3020, 1906],
        [3109, 1878], [3141, 1874], [3141, 1785], [3176, 1722], [3246, 1711],
        [3293, 1761], [3304, 1816], [3386, 1808], [3445, 1812], [3511, 1796],
        [3507, 1726], [4144, 1700], [4144, 2822],
    ],
    "North": [
        [0, 2822], [0, 1835], [162, 1878], [353, 1913], [583, 1956],
        [774, 1995], [989, 1991], [1141, 2015], [1277, 2046], [1546, 2058],
        [1761, 2065], [2057, 2073], [2079, 2069], [2603, 2061], [2981, 2038],
        [3300, 2011], [4144, 1972], [4144, 2822],
    ],
    "South": [
        [0, 2822], [0, 1656], [81, 1676], [123, 1695], [201, 1711],
        [209, 1633], [307, 1644], [314, 1699], [318, 1738], [427, 1754],
        [474, 1765], [564, 1777], [572, 1734], [638, 1746], [646, 1820],
        [747, 1839], [1125, 1913], [1347, 1968], [1410, 1995], [1562, 2085],
        [2712, 2046], [2872, 2011], [2988, 1991], [3074, 1980], [3144, 1964],
        [3347, 1976], [4144, 1906], [4144, 2822],
    ],
}

GRID_SIZE = 25  # pixels per grid cell (square)
MEDIAN_FILTER_SIZE = 5
TIME_RANGE = ((7, 0), (14, 20))  # UTC = 9 PM to 4:20 AM Hawaiian
MIN_CENTROIDS_FOR_FIT = 5  # minimum valid centroids required to fit per-image zero_point


def parse_args():
    parser = argparse.ArgumentParser(description="Measure sky background brightness")
    parser.add_argument("--direction", required=True, choices=DIRECTIONS,
                        help="Camera direction (North, East, South, West)")
    parser.add_argument("--date", required=True, help="Date in YYYYMMDD format (e.g. 20260118)")
    parser.add_argument("--data", choices=["local", "api"], default="local",
                        help="Data source (default: local)")
    parser.add_argument("--skip-median-filter", action="store_true",
                        help="Skip the median filter step (use raw pixel values)")
    return parser.parse_args()


def _build_ground_mask(direction: str, h: int, w: int) -> np.ndarray:
    """Build a boolean mask where True = ground (masked out), False = sky."""
    polygon = GROUND_MASK_POLYGONS[direction]
    poly_path = MplPath(polygon)
    yy, xx = np.mgrid[0:h, 0:w]
    ground_mask = poly_path.contains_points(
        np.column_stack([xx.ravel(), yy.ravel()])
    ).reshape(h, w)
    return ground_mask


def _compute_cell_azel(
    direction: str,
    model: FisheyeCorrectionModel,
    h: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Az/El for the center of each pixel grid cell via inverse lookup.

    Projects a dense Az/El grid to pixel space, then for each grid cell accumulates
    the Az/El of all projected points that fall within it and takes the circular mean
    (for Az) and arithmetic mean (for El). Cells with no projected points are NaN.

    Returns:
        cell_az: (n_rows, n_cols) array of azimuth values in degrees
        cell_el: (n_rows, n_cols) array of elevation values in degrees
    """
    az_range, el_range = SkyImage.get_azel_range(direction)
    az_min, az_max = az_range
    el_min, el_max = el_range

    # Build dense Az/El grid (step 0.1 deg)
    step = 0.1
    if az_min < 0:
        # Wrap-around case (e.g. North: -67 to 68 means 293..360 and 0..68)
        az_values = np.concatenate([
            np.arange(360 + az_min, 360, step),
            np.arange(0, az_max, step),
        ])
    else:
        az_values = np.arange(az_min, az_max, step)
    el_values = np.arange(el_min, el_max, step)

    Az_dense, El_dense = np.meshgrid(az_values, el_values)
    az_flat = Az_dense.ravel()
    el_flat = El_dense.ravel()

    X_test = Table()
    X_test["Az"] = az_flat
    X_test["El"] = el_flat
    x_px, y_px = model.predict(X_test)

    n_rows = h // GRID_SIZE
    n_cols = w // GRID_SIZE

    # Accumulators for circular mean of Az (sin/cos components) and arithmetic mean of El
    sin_az = np.zeros((n_rows, n_cols))
    cos_az = np.zeros((n_rows, n_cols))
    el_sum = np.zeros((n_rows, n_cols))
    counts = np.zeros((n_rows, n_cols), dtype=np.int32)

    for k in range(len(x_px)):
        xk, yk = x_px[k], y_px[k]
        if not (0 <= xk < w and 0 <= yk < h):
            continue
        r = int(yk) // GRID_SIZE
        c = int(xk) // GRID_SIZE
        az_rad = np.radians(az_flat[k])
        sin_az[r, c] += np.sin(az_rad)
        cos_az[r, c] += np.cos(az_rad)
        el_sum[r, c] += el_flat[k]
        counts[r, c] += 1

    valid = counts > 0
    cell_az = np.full((n_rows, n_cols), np.nan)
    cell_el = np.full((n_rows, n_cols), np.nan)
    cell_az[valid] = np.degrees(np.arctan2(sin_az[valid], cos_az[valid])) % 360
    cell_el[valid] = el_sum[valid] / counts[valid]

    return cell_az, cell_el


def _apply_median_filter(img: np.ndarray, ground_mask: np.ndarray, skip: bool = False) -> np.ndarray:
    """
    Extract the green channel and optionally apply a median filter to the sky region.

    Args:
        img: RGB image array (H, W, 3)
        ground_mask: Boolean mask where True = ground
        skip: If True, return the raw green channel without filtering

    Returns:
        Green channel (H, W), median-filtered in the sky region unless skip=True
    """
    green = img[:, :, 1].astype(np.float32)
    if skip:
        return green
    filtered = median_filter(green, size=MEDIAN_FILTER_SIZE)
    result = green.copy()
    result[~ground_mask] = filtered[~ground_mask]
    return result


def _measure_grid_medians(
    img_filtered: np.ndarray,
    ground_mask: np.ndarray,
) -> np.ndarray:
    """
    Measure median pixel brightness in each GRID_SIZE x GRID_SIZE pixel cell.

    Cells with fewer than 50% sky pixels (not masked by ground) are skipped (NaN).

    Returns:
        (n_rows, n_cols) array of median pixel values, NaN for invalid cells.
    """
    h, w = img_filtered.shape[:2]
    n_rows = h // GRID_SIZE
    n_cols = w // GRID_SIZE
    cell_medians = np.full((n_rows, n_cols), np.nan)

    min_sky_pixels = 0.5 * GRID_SIZE * GRID_SIZE

    for r in range(n_rows):
        for c in range(n_cols):
            y0, y1 = r * GRID_SIZE, (r + 1) * GRID_SIZE
            x0, x1 = c * GRID_SIZE, (c + 1) * GRID_SIZE
            sky_mask = ~ground_mask[y0:y1, x0:x1]
            if sky_mask.sum() < min_sky_pixels:
                continue
            pixels = img_filtered[y0:y1, x0:x1][sky_mask]
            cell_medians[r, c] = np.median(pixels)

    return cell_medians


def _time_in_range(t: datetime, start: tuple, end: tuple) -> bool:
    """Check if time falls within a range that may cross midnight."""
    t_minutes = t.hour * 60 + t.minute
    start_minutes = start[0] * 60 + start[1]
    end_minutes = end[0] * 60 + end[1]
    if start_minutes <= end_minutes:
        return start_minutes <= t_minutes <= end_minutes
    else:
        return t_minutes >= start_minutes or t_minutes <= end_minutes


def _parse_timestamp(filename: str) -> datetime:
    """Parse timestamp from FITS filename like CloudCamWest_20260118UTC070043.fits."""
    time_str = filename.split("UTC")[1][:6]
    time_str = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
    date_str = filename.split("UTC")[0][-8:]
    date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")


def measure_background(direction: str, date: str, data_source: str = "local", skip_median_filter: bool = False):
    """
    Measure sky background brightness for a given direction and date.

    Pipeline:
      1. Load fisheye model and brightness calibration
      2. Open first image to get image dimensions, build ground mask and cell Az/El lookup
      3. For each image:
           a. Centroid stars, measure brightness, fit per-image zero_point
           b. Apply median filter, measure 25x25 pixel grid, convert to vmag_per_pixel
           c. Emit one cell row (source, az, el, vmag_per_pixel) per valid grid cell
           d. Emit one image row (source, zero_point, centroid_rate)
      4. Save two CSVs to out/background/

    Args:
        direction: Camera direction (North, East, South, West)
        date: Date in YYYYMMDD format
        data_source: "local" or "api"
        skip_median_filter: If True, skip the median filter step
    """
    # Load model and calibration
    model = FisheyeCorrectionModel(direction=direction)
    brightness_cal = load_brightness_calibration(direction)
    print(f"Loaded model and calibration for {direction}")

    # Get file list
    if data_source == "api":
        all_files = SkyImage.get_file_list_api(direction, date)
        directory = None
    else:
        directory = PROJECT_ROOT / f"data/CloudCam{direction}/{date}"
        all_files = sorted(os.listdir(directory))

    # Filter by nighttime
    files_with_times = [(f, _parse_timestamp(f)) for f in all_files]
    filtered = [(f, t) for f, t in files_with_times
                if _time_in_range(t, TIME_RANGE[0], TIME_RANGE[1])]
    print(f"Found {len(filtered)} nighttime images (skipped {len(all_files) - len(filtered)})")

    if not filtered:
        print("No nighttime images found.")
        return None, None

    # Open first image to get dimensions, then build fixed-per-direction masks
    first_file, _ = filtered[0]
    if data_source == "api":
        ctx = SkyImage.open_api(direction, date, first_file)
    else:
        ctx = SkyImage.open(directory / first_file)
    with ctx as hdul:
        h, w = hdul[0].data.shape[1], hdul[0].data.shape[2]

    print(f"Image size: {h}x{w}, building ground mask and Az/El cell lookup...")
    ground_mask = _build_ground_mask(direction, h, w)
    cell_az, cell_el = _compute_cell_azel(direction, model, h, w)
    print(f"Az/El lookup complete. Grid: {cell_az.shape[0]}x{cell_az.shape[1]} cells")

    az_range, el_range = SkyImage.get_azel_range(direction)
    slist = StarList()

    cell_rows = []   # source, az, el, vmag_per_pixel
    image_rows = []  # source, zero_point, centroid_rate

    for filename, timestamp in tqdm(filtered, desc="Processing images"):
        if data_source == "api":
            ctx = SkyImage.open_api(direction, date, filename)
        else:
            ctx = SkyImage.open(directory / filename)

        with ctx as hdul:
            # --- Star brightness measurement ---
            catalog = slist.filter_catalog(timestamp, az_range, el_range)
            x_pred, y_pred = model.predict(catalog[["Az", "El"]])
            catalog["x_transform"] = x_pred
            catalog["y_transform"] = y_pred

            cx, cy, _, _, brightness = centroid_starlist(
                hdul, catalog, measure_brightness=True
            )
            gain = float(hdul[0].header["GAIN"])
            exp_time = float(hdul[0].header["EXP_TIME"])

            # Compute adjusted brightness for each successfully centroided star
            image_brightness_rows = []
            n_valid_centroids = 0
            for i in range(len(catalog)):
                if cx[i] is not None and cy[i] is not None:
                    n_valid_centroids += 1
                    adjusted = (
                        brightness[i]
                        - (gain * brightness_cal["gain_coef"]
                           + exp_time * brightness_cal["exp_time_coef"]
                           + brightness_cal["intercept"])
                        + brightness_cal["mean_brightness"]
                    )
                    if not (adjusted >= 0):
                        continue
                    vmag = catalog["vmag"][i]
                    if not (3.0 < vmag < 7.0):
                        continue
                    image_brightness_rows.append({
                        "adjusted_brightness": adjusted,
                        "vmag": vmag,
                    })

            # --- Background measurement ---
            img = np.flip(np.transpose(hdul[0].data, (1, 2, 0)), axis=1)

        # Fit per-image zero_point; skip image if too few valid centroids
        n_catalog = len(catalog)
        centroid_rate = n_valid_centroids / n_catalog if n_catalog > 0 else 0.0

        if len(image_brightness_rows) < MIN_CENTROIDS_FOR_FIT:
            continue

        try:
            img_df = pd.DataFrame(image_brightness_rows)
            regression = fit_vmag_regression(img_df)
        except ValueError:
            continue

        zero_point = regression["vmag_intercept"]

        # Apply median filter and measure grid
        img_filtered = _apply_median_filter(img, ground_mask, skip=skip_median_filter)
        cell_medians = _measure_grid_medians(img_filtered, ground_mask)

        # Convert each valid cell to vmag_per_pixel and emit a row
        n_rows_grid, n_cols_grid = cell_medians.shape
        for r in range(n_rows_grid):
            for c in range(n_cols_grid):
                if np.isnan(cell_medians[r, c]) or cell_medians[r, c] <= 0:
                    continue
                if np.isnan(cell_az[r, c]) or np.isnan(cell_el[r, c]):
                    continue
                vmag_per_pixel = -2.5 * np.log10(cell_medians[r, c]) + zero_point
                cell_rows.append({
                    "source": filename,
                    "az": cell_az[r, c],
                    "el": cell_el[r, c],
                    "vmag_per_pixel": vmag_per_pixel,
                })

        image_rows.append({
            "source": filename,
            "zero_point": zero_point,
            "centroid_rate": centroid_rate,
        })

    if not image_rows:
        print("No images had enough valid centroids to fit a zero_point.")
        return None, None

    cells_df = pd.DataFrame(cell_rows)
    images_df = pd.DataFrame(image_rows)

    out_dir = PROJECT_ROOT / "out" / "background"
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_path = out_dir / f"{direction.lower()}_{date}_cells.csv"
    images_path = out_dir / f"{direction.lower()}_{date}_images.csv"

    cells_df.to_csv(cells_path, index=False)
    images_df.to_csv(images_path, index=False)

    print(f"\nSaved {len(cells_df)} cell measurements to {cells_path}")
    print(f"Saved {len(images_df)} image summaries to {images_path}")
    print(f"zero_point range: {images_df['zero_point'].min():.2f} - {images_df['zero_point'].max():.2f}")
    print(f"centroid_rate range: {images_df['centroid_rate'].min():.2f} - {images_df['centroid_rate'].max():.2f}")

    return cells_df, images_df


if __name__ == "__main__":
    args = parse_args()
    measure_background(
        direction=args.direction,
        date=args.date,
        data_source=args.data,
        skip_median_filter=args.skip_median_filter,
    )
