"""
Extract star centroids from FITS images and fit brightness regression.

Given a calibrated fisheye distortion model, this pipeline:
  1. Loads the distortion matrix for a given direction
  2. Iterates FITS images, predicts star positions, and centroids them
  3. Fits brightness calibration (gain/exp_time adjustment + Pogson's law)
  4. Appends regression summary to out/evals/summaries/{direction}_brightness_summaries.csv
  5. Saves per-centroid brightness measurements to out/evals/measurements/{direction}_brightness_measurements.csv

Usage:
    python extract_centroids.py --direction South --date 20260118
    python extract_centroids.py --direction all --date 20260118 --save-centroids
    python extract_centroids.py --direction South --date 20260118 --save-plot
    python extract_centroids.py --direction South --date 20260118 --update-cam-cal

Arguments:
    --direction: Camera direction (North, South, East, West, or "all")
    --date: Date of FITS images (YYYYMMDD format)
    --save-centroids: Save centroid positions to data/Calibration/{date}/
    --save-plot: Save brightness regression scatter plot to out/evals/figs/
    --update-cam-cal: Fit gain/exp_time calibration and write to out/cals/{direction}_brightness_cal.json
    --data-source: Data source - "local" or "api" (default: "local")
"""

import argparse
import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from astropy.coordinates import IllegalSecondWarning
from tqdm import tqdm

from config.settings import PROJECT_ROOT
from data_management import SkyImage, StarList
from models.centroid import centroid_starlist
from models.fisheye_correction import FisheyeCorrectionModel

warnings.filterwarnings("ignore", category=IllegalSecondWarning)

DIRECTIONS = ["North", "East", "South", "West"]
ADJACENT_DISTANCE_THRESHOLD = 16  # pixels
VMAG_MIN = 3.0
VMAG_MAX = 7.0

# Default time range: 9 PM - 4:20 AM Hawaiian = 7:00 - 14:20 UTC
DEFAULT_TIME_RANGE = ((7, 0), (14, 20))


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

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
    """Parse timestamp from FITS filename."""
    time_str = filename.split("UTC")[1][:6]
    time_str = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
    date_str = filename.split("UTC")[0][-8:]
    date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")


def remove_adjacent_stars(catalog, x_col="x_transform", y_col="y_transform", threshold=ADJACENT_DISTANCE_THRESHOLD):
    """
    Remove stars whose transformed positions are within `threshold` pixels of any other star.

    Both members of a close pair are removed.

    Args:
        catalog: StarList catalog with x/y transform columns
        x_col: Column name for x coordinates
        y_col: Column name for y coordinates
        threshold: Euclidean distance threshold in pixels

    Returns:
        Filtered catalog with adjacent stars removed
    """
    x = np.array(catalog[x_col], dtype=float)
    y = np.array(catalog[y_col], dtype=float)
    n = len(x)
    if n <= 1:
        return catalog

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist, np.inf)

    too_close = np.any(dist < threshold, axis=1)
    keep_mask = ~too_close

    removed = int(too_close.sum())
    if removed > 0:
        catalog = catalog[keep_mask]
    return catalog


# ---------------------------------------------------------------------------
# Centroid extraction
# ---------------------------------------------------------------------------

def extract_centroids_for_direction(
    model: FisheyeCorrectionModel,
    direction: str,
    date: str,
    file_range: Tuple[int, int] = (0, 99999),
    time_range: Tuple[Tuple[int, int], Tuple[int, int]] = DEFAULT_TIME_RANGE,
    data_source: Literal["local", "api"] = "local",
    remove_adjacent: bool = True,
) -> Tuple[pd.DataFrame, int, int]:
    """
    Extract star centroids with brightness from FITS images for a single direction.

    For each image:
      1. Get full catalog for the az/el range
      2. Predict pixel positions for all stars
      3. Remove adjacent stars (uses full catalog for accurate distance checks)
      4. Filter to vmag range VMAG_MIN–VMAG_MAX
      5. Centroid the remaining stars and measure brightness

    Args:
        model: Trained fisheye correction model for predictions
        direction: Camera direction
        date: Date of FITS images (YYYYMMDD format)
        file_range: Range of file indices to process
        time_range: ((start_hour, start_min), (end_hour, end_min)) in UTC
        data_source: "local" for filesystem, "api" for remote API
        remove_adjacent: Remove stars that are too close to neighbors (default: True)

    Returns:
        Tuple of (centroids_df, attempted_centroids, successful_centroids)
        centroids_df columns: source, TargetName, Az, El, x_pixel, y_pixel, brightness, vmag, gain, exp_time
    """
    if data_source == "api":
        all_files = SkyImage.get_file_list_api(direction, date)
        data_dir = None
    else:
        data_dir = PROJECT_ROOT / "data" / f"CloudCam{direction}" / date
        all_files = sorted(os.listdir(data_dir))
    az_range, el_range = SkyImage.get_azel_range(direction)

    files_in_range = all_files[file_range[0]:file_range[1]]

    files_with_times = [(f, _parse_timestamp(f)) for f in files_in_range]
    filtered = [(f, t) for f, t in files_with_times
                if _time_in_range(t, time_range[0], time_range[1])]

    print(f"  Found {len(filtered)} files in time range (skipped {len(files_in_range) - len(filtered)})")

    slist = StarList()
    results = []
    attempted_centroids = 0
    successful_centroids = 0

    for filename, time in tqdm(filtered, desc=f"Centroiding {direction}"):
        if data_source == "api":
            ctx = SkyImage.open_api(direction, date, filename)
        else:
            ctx = SkyImage.open(data_dir / filename)

        with ctx as hdul:
            # Full catalog for the az/el range
            catalog = slist.filter_catalog(time, az_range, el_range)

            # Predict pixel positions for all stars
            x_pred, y_pred = model.predict(catalog[["Az", "El"]])
            catalog["x_transform"] = x_pred
            catalog["y_transform"] = y_pred

            # Remove adjacent stars using the full catalog for accurate distance checks
            if remove_adjacent:
                catalog = remove_adjacent_stars(catalog)

            # Filter to vmag range
            catalog = catalog[(catalog["vmag"] > VMAG_MIN) & (catalog["vmag"] < VMAG_MAX)]

            # Centroid the filtered catalog
            cx, cy, _, _, brightness = centroid_starlist(hdul, catalog, measure_brightness=True)
            gain = float(hdul[0].header["GAIN"])
            exp_time = float(hdul[0].header["EXP_TIME"])

            attempted_centroids += len(catalog)

            for i, star in enumerate(catalog):
                if cx[i] is not None and cy[i] is not None:
                    successful_centroids += 1
                    results.append({
                        "source": filename,
                        "TargetName": star["TargetName"],
                        "Az": star["Az"],
                        "El": star["El"],
                        "x_pixel": cx[i],
                        "y_pixel": cy[i],
                        "brightness": brightness[i],
                        "vmag": star["vmag"],
                        "gain": gain,
                        "exp_time": exp_time,
                    })

    return pd.DataFrame(results), attempted_centroids, successful_centroids


# ---------------------------------------------------------------------------
# Brightness calibration fitting
# ---------------------------------------------------------------------------

def fit_brightness_calibration(df: pd.DataFrame) -> dict:
    """
    Fit brightness calibration coefficients from centroid data with brightness measurements.

    Fits two models:
    1. Gain/exp_time adjustment: brightness = coef[0]*gain + coef[1]*exp_time + intercept
       The adjusted brightness is: brightness - predicted + mean_brightness
    2. Vmag prediction (Pogson's law): vmag = -2.5 * log10(adjusted_brightness) + vmag_intercept

    Args:
        df: DataFrame with columns: brightness, vmag, gain, exp_time

    Returns:
        Dictionary with keys: coef, intercept, mean_brightness, vmag_intercept, vmag_r2, vmag_std, num_centroids
    """
    clean = df.dropna()

    X = clean[["gain", "exp_time"]].values
    y = clean["brightness"].values

    model = LinearRegression().fit(X, y)
    mean_brightness = y.mean()

    predicted = model.predict(X)
    adjusted_brightness = y - predicted + mean_brightness

    valid_mask = adjusted_brightness > 0
    adjusted_brightness = adjusted_brightness[valid_mask]
    clean = clean[valid_mask]

    log_brightness = np.log10(adjusted_brightness)
    vmag_intercept = np.mean(clean["vmag"].values - (-2.5 * log_brightness))

    predicted_vmag = -2.5 * log_brightness + vmag_intercept
    residuals = clean["vmag"].values - predicted_vmag
    std = residuals.std()
    inlier_mask = np.abs(residuals) <= 3 * std

    adjusted_brightness = adjusted_brightness[inlier_mask]
    clean = clean[inlier_mask]

    log_brightness = np.log10(adjusted_brightness)
    vmag_intercept = np.mean(clean["vmag"].values - (-2.5 * log_brightness))
    predicted_vmag = -2.5 * log_brightness + vmag_intercept
    residuals = clean["vmag"].values - predicted_vmag
    vmag_std = residuals.std()

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((clean["vmag"].values - clean["vmag"].values.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {
        "coef": model.coef_,
        "intercept": model.intercept_,
        "mean_brightness": mean_brightness,
        "vmag_intercept": vmag_intercept,
        "vmag_r2": r2,
        "vmag_std": vmag_std,
        "num_centroids": len(clean),
    }


# ---------------------------------------------------------------------------
# Brightness calibration loading (used by measure_background.py)
# ---------------------------------------------------------------------------

def load_brightness_calibration(direction: str, cals_dir: Path = None) -> dict:
    """
    Load brightness calibration coefficients from file.

    Args:
        direction: Camera direction
        cals_dir: Directory containing calibration files (default: PROJECT_ROOT/out/cals)

    Returns:
        Dictionary with keys: gain_coef, exp_time_coef, intercept, mean_brightness,
                              vmag_intercept, vmag_r2
    """
    if cals_dir is None:
        cals_dir = PROJECT_ROOT / "out" / "cals"
    cal_path = cals_dir / f"{direction.lower()}_brightness_cal.json"

    if not cal_path.exists():
        raise FileNotFoundError(
            f"No brightness calibration found at {cal_path}. "
            f"Run calibrate.py first to generate calibration files."
        )

    with open(cal_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Vmag regression (also used by measure_background.py)
# ---------------------------------------------------------------------------

def fit_vmag_regression(df: pd.DataFrame) -> dict:
    """
    Fit vmag vs adjusted_brightness using Pogson's law: vmag = -2.5 * log10(brightness) + intercept.

    Args:
        df: DataFrame with columns: vmag, adjusted_brightness

    Returns:
        Dictionary with keys: vmag_intercept, vmag_r2, vmag_std, num_centroids
    """
    clean_df = df[["adjusted_brightness", "vmag"]].dropna()
    clean_df = clean_df[clean_df["adjusted_brightness"] > 0]

    if len(clean_df) < 2:
        raise ValueError(f"Not enough valid data points for regression: {len(clean_df)}")

    log_brightness = np.log10(clean_df["adjusted_brightness"].values)

    vmag_intercept = np.mean(clean_df["vmag"].values - (-2.5 * log_brightness))

    predicted_vmag = -2.5 * log_brightness + vmag_intercept
    residuals = clean_df["vmag"].values - predicted_vmag
    std = residuals.std()

    if std > 0:
        inlier_mask = np.abs(residuals) <= 3 * std
        clean_df = clean_df[inlier_mask]

    if len(clean_df) < 2:
        raise ValueError(f"Not enough inliers after outlier removal: {len(clean_df)}")

    log_brightness = np.log10(clean_df["adjusted_brightness"].values)
    vmag_intercept = np.mean(clean_df["vmag"].values - (-2.5 * log_brightness))
    predicted_vmag = -2.5 * log_brightness + vmag_intercept
    residuals = clean_df["vmag"].values - predicted_vmag
    vmag_std = residuals.std()

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((clean_df["vmag"].values - clean_df["vmag"].values.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {
        "vmag_intercept": vmag_intercept,
        "vmag_r2": r2,
        "vmag_std": vmag_std,
        "num_centroids": len(clean_df),
    }


# ---------------------------------------------------------------------------
# Output: evals CSV
# ---------------------------------------------------------------------------

def save_brightness_calibration(
    calibration: dict,
    direction: str,
    date: str,
    output_dir: Path = None,
) -> Path:
    """
    Save brightness calibration coefficients to JSON file.

    Args:
        calibration: Dictionary from fit_brightness_calibration()
        direction: Camera direction
        output_dir: Output directory (default: PROJECT_ROOT/out)

    Returns:
        Path to saved file
    """
    output_dir = output_dir or PROJECT_ROOT / "out"
    cals_dir = output_dir / "cals"
    cals_dir.mkdir(parents=True, exist_ok=True)
    cal_path = cals_dir / f"{direction.lower()}_brightness_cal.json"

    serializable = {
        "calibrate_date": date,
        "gain_coef": float(calibration["coef"][0]),
        "exp_time_coef": float(calibration["coef"][1]),
        "intercept": float(calibration["intercept"]),
        "mean_brightness": float(calibration["mean_brightness"]),
        "vmag_intercept": float(calibration["vmag_intercept"]),
        "vmag_r2": float(calibration["vmag_r2"]),
        "vmag_std": float(calibration["vmag_std"]),
    }

    with open(cal_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"Saved brightness calibration to: {cal_path}")
    return cal_path


def save_brightness_regression(
    regression: dict,
    direction: str,
    date: str,
    attempted_centroids: int,
    successful_centroids: int,
    output_dir: Path = None,
) -> Path:
    """
    Append brightness regression results to CSV file.

    Args:
        regression: Dictionary with regression parameters (vmag_intercept, vmag_r2, vmag_std, num_centroids)
        direction: Camera direction
        date: Date of measurement (YYYYMMDD format)
        attempted_centroids: Total number of centroid attempts
        successful_centroids: Number of successful centroids
        output_dir: Output directory (default: PROJECT_ROOT/out)

    Returns:
        Path to saved file
    """
    output_dir = output_dir or PROJECT_ROOT / "out"
    summaries_dir = output_dir / "evals" / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    output_path = summaries_dir / f"{direction.lower()}_brightness_summaries.csv"

    row = {
        "date": date,
        "direction": direction,
        "attempted_centroids": attempted_centroids,
        "successful_centroids": successful_centroids,
        "vmag_intercept": regression["vmag_intercept"],
        "vmag_r2": regression["vmag_r2"],
        "vmag_std": regression["vmag_std"],
    }

    file_exists = output_path.exists()

    df = pd.DataFrame([row])
    df.to_csv(output_path, mode='a', header=not file_exists, index=False)

    print(f"Appended brightness summary to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Output: brightness regression plot
# ---------------------------------------------------------------------------

def _plot_brightness_regression(df: pd.DataFrame, regression: dict, out_path: Path):
    """Generate brightness vs vmag scatter plot with fitted regression line and outlier boundaries."""
    _, ax = plt.subplots(figsize=(10, 7))

    predicted_vmag = -2.5 * np.log10(df["adjusted_brightness"].values.clip(min=1e-10)) + regression["vmag_intercept"]
    residuals = df["vmag"].values - predicted_vmag
    outlier_mask = np.abs(residuals) > 3 * regression["vmag_std"]

    inliers = df[~outlier_mask]
    outliers = df[outlier_mask]

    ax.scatter(inliers["adjusted_brightness"], inliers["vmag"], alpha=0.3, s=0.1, c='blue', label=f"Inliers (n={len(inliers)})")
    ax.scatter(outliers["adjusted_brightness"], outliers["vmag"], alpha=0.3, s=0.1, c='red', marker='x', label=f"Outliers (n={len(outliers)})")

    x_range = np.linspace(max(df["adjusted_brightness"].min(), 1e-10), df["adjusted_brightness"].max(), 100)
    y_pred = -2.5 * np.log10(x_range) + regression["vmag_intercept"]
    y_upper = y_pred + 3 * regression["vmag_std"]
    y_lower = y_pred - 3 * regression["vmag_std"]

    ax.plot(x_range, y_pred, 'k-', linewidth=2,
            label=f'Fit: vmag = -2.5 * log10(brightness) + {regression["vmag_intercept"]:.2f}')
    ax.plot(x_range, y_upper, 'g--', linewidth=1, label=f'3 std boundary (std={regression["vmag_std"]:.3f})')
    ax.plot(x_range, y_lower, 'g--', linewidth=1)

    ax.set(xlabel='Adjusted Brightness', ylabel='Visual Magnitude (vmag)',
           title=f'Brightness Regression (R² = {regression["vmag_r2"]:.4f}, n = {regression["num_centroids"]})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

def run_extract_centroids(
    direction: str,
    date: str,
    file_range: Tuple[int, int] = (0, 99999),
    data_source: Literal["local", "api"] = "local",
    save_centroids: bool = False,
    save_plot: bool = False,
    update_cam_cal: bool = False,
    output_dir: Path = None,
) -> dict:
    """
    Extract centroids and fit brightness regression for a single direction.

    Args:
        direction: Camera direction
        date: Date of FITS images (YYYYMMDD format)
        file_range: Range of file indices to process
        data_source: "local" for filesystem, "api" for remote API
        save_centroids: Save centroid positions to data/Calibration/{date}/
        save_plot: Save brightness regression plot to out/evals/
        output_dir: Output directory (default: PROJECT_ROOT/out)

    Returns:
        Brightness calibration dictionary
    """
    output_dir = output_dir or PROJECT_ROOT / "out"

    # Step 1: Load distortion model
    model = FisheyeCorrectionModel(direction=direction)
    print(f"Loaded Distort matrix for {direction}")

    # Step 2: Extract centroids with brightness
    print(f"\nExtracting centroids from {date}...")
    centroids_df, attempted, successful = extract_centroids_for_direction(
        model, direction, date, file_range,
        data_source=data_source,
    )
    print(f"Extracted {successful} / {attempted} centroids ({successful/attempted*100:.1f}% success rate)" if attempted > 0 else "No centroids attempted")

    # Step 3: Fit brightness calibration
    print(f"\nFitting brightness calibration...")
    brightness_cal = fit_brightness_calibration(centroids_df)
    print(f"  Gain coef: {brightness_cal['coef'][0]:.4f}")
    print(f"  Exp_time coef: {brightness_cal['coef'][1]:.4f}")
    print(f"  Mean brightness: {brightness_cal['mean_brightness']:.4f}")
    print(f"  Vmag: vmag = -2.5 * log10(adjusted_brightness) + {brightness_cal['vmag_intercept']:.4f}")
    print(f"  Vmag R^2: {brightness_cal['vmag_r2']:.4f}")
    print(f"  Vmag Std: {brightness_cal['vmag_std']:.4f}")

    # Optional: Save brightness calibration to out/cals/
    if update_cam_cal:
        save_brightness_calibration(brightness_cal, direction, date, output_dir)

    # Step 4: Compute adjusted brightness for measurements
    clean = centroids_df.dropna()
    X = clean[["gain", "exp_time"]].values
    predicted = brightness_cal["coef"][0] * X[:, 0] + brightness_cal["coef"][1] * X[:, 1] + brightness_cal["intercept"]
    clean = clean.copy()
    clean["adjusted_brightness"] = clean["brightness"].values - predicted + brightness_cal["mean_brightness"]
    clean = clean[clean["adjusted_brightness"] > 0]

    # Step 5: Save regression summary to evals CSV
    save_brightness_regression(
        brightness_cal, direction, date,
        attempted, successful, output_dir,
    )

    # Step 6: Save brightness measurements CSV (overwrites each run)
    measurements_dir = output_dir / "evals" / "measurements"
    measurements_dir.mkdir(parents=True, exist_ok=True)
    measurements_path = measurements_dir / f"{direction.lower()}_brightness_measurements.csv"
    clean[["source", "TargetName", "vmag", "adjusted_brightness"]].to_csv(measurements_path, index=False)
    print(f"Saved brightness measurements to: {measurements_path}")

    # Optional: Save centroids to data/Calibration/{date}/
    if save_centroids:
        cal_dir = PROJECT_ROOT / "data" / "Calibration" / date
        cal_dir.mkdir(parents=True, exist_ok=True)
        centroids_path = cal_dir / f"{direction.lower()}_centroid_mapping.csv"
        centroid_cols = ["source", "TargetName", "Az", "El", "x_pixel", "y_pixel"]
        centroids_df[centroid_cols].to_csv(centroids_path, index=False)
        print(f"Saved centroids to: {centroids_path}")

    # Optional: Save brightness regression plot
    if save_plot:
        figs_dir = output_dir / "evals" / "figs"
        figs_dir.mkdir(parents=True, exist_ok=True)
        plot_path = figs_dir / f"{direction.lower()}_brightness_regression.png"
        _plot_brightness_regression(clean, brightness_cal, plot_path)
        print(f"Saved brightness regression plot to: {plot_path}")

    return brightness_cal


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Main entry point for terminal execution."""
    parser = argparse.ArgumentParser(
        description="Extract star centroids and fit brightness regression"
    )

    parser.add_argument(
        "--direction", type=str, default="all",
        help="Camera direction (North, South, East, West, or 'all'). Default: all",
    )
    parser.add_argument(
        "--date", type=str, required=True,
        help="Date of FITS images (YYYYMMDD format)",
    )
    parser.add_argument(
        "--file-range", type=int, nargs=2, default=[0, 99999],
        metavar=("START", "END"),
        help="Range of file indices to process. Default: 0 99999",
    )
    parser.add_argument(
        "--data-source", type=str, default="local", choices=["local", "api"],
        help="Data source. Default: local",
    )
    parser.add_argument(
        "--save-centroids", action="store_true",
        help="Save centroid positions to data/Calibration/{date}/. Default: False",
    )
    parser.add_argument(
        "--save-plot", action="store_true",
        help="Save brightness regression plot to out/evals/. Default: False",
    )
    parser.add_argument(
        "--update-cam-cal", action="store_true",
        help="Fit gain/exp_time calibration and write to out/cals/. Default: False",
    )
    parser.add_argument(
        "--output-dir", type=str,
        help="Custom output directory path",
    )

    args = parser.parse_args()

    if args.direction != "all" and args.direction not in DIRECTIONS:
        parser.error(f"direction must be one of {DIRECTIONS} or 'all'")

    directions = DIRECTIONS if args.direction == "all" else [args.direction]
    output_dir = Path(args.output_dir) if args.output_dir else None

    for direction in directions:
        print(f"\n{'='*60}")
        print(f"Processing {direction}")
        print(f"{'='*60}")

        run_extract_centroids(
            direction, args.date,
            file_range=tuple(args.file_range),
            data_source=args.data_source,
            save_centroids=args.save_centroids,
            save_plot=args.save_plot,
            update_cam_cal=args.update_cam_cal,
            output_dir=output_dir,
        )

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
