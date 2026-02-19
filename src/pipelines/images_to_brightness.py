#!/usr/bin/env python
"""Pipeline to fit brightness regression for measurement nights."""

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from config.settings import PROJECT_ROOT
from data_management.sky_image import SkyImage
from data_management.starlist import StarList
from models.centroid import centroid_starlist
from models.fisheye_correction import FisheyeCorrectionModel
from pipelines.calibrate import load_brightness_calibration, remove_adjacent_stars

DIRECTIONS = ["North", "East", "South", "West"]


def parse_args():
    parser = argparse.ArgumentParser(description="Fit brightness regression for measurement nights")
    parser.add_argument("--date", required=True, help="Date in YYYYMMDD format")
    parser.add_argument("--direction", default="all", help="Camera direction (North, South, East, West, or 'all'). Default: all")
    parser.add_argument("--output", "-o", type=Path, default=PROJECT_ROOT / "out", help="Output directory")
    parser.add_argument("--sample-index", type=int, default=600, help="Index of sample image for visualization")
    parser.add_argument("--skip-sample", action="store_true", help="Skip sample image visualization")
    parser.add_argument("--data-source", choices=["local", "api"], default="local", help="Data source (default: local)")
    parser.add_argument("--remove-adjacent", action="store_true", help="Remove stars within 13 pixels of another star (both removed). Default: False")
    return parser.parse_args()

def _time_in_range(t: datetime, start: tuple, end: tuple) -> bool:
    """
    Check if time falls within a range that may cross midnight.

    Args:
        t: Time to check
        start: (hour, minute) tuple for start of range
        end: (hour, minute) tuple for end of range

    Returns:
        True if time is within range
    """
    t_minutes = t.hour * 60 + t.minute
    start_minutes = start[0] * 60 + start[1]
    end_minutes = end[0] * 60 + end[1]

    if start_minutes <= end_minutes:
        # Range doesn't cross midnight (e.g., 10:00 to 14:00)
        return start_minutes <= t_minutes <= end_minutes
    else:
        # Range crosses midnight (e.g., 21:00 to 04:20)
        return t_minutes >= start_minutes or t_minutes <= end_minutes

def _visualize_sample(model, direction: str, date: str, sample_index: int, out_dir: Path, data_source: str = "local"):
    """Generate visualization comparing centroid vs transformed positions."""
    if data_source == "api":
        files = SkyImage.get_file_list_api(direction, date)
    else:
        directory = PROJECT_ROOT / f"data/CloudCam{direction}/{date}"
        files = sorted(os.listdir(directory))
    if len(files) == 0:
        print("No files found for visualization, skipping.")
        return
    if sample_index >= len(files):
        sample_index = len(files) // 2
    filename = files[sample_index]

    # Open file from API or local filesystem
    if data_source == "api":
        ctx = SkyImage.open_api(direction, date, filename)
    else:
        ctx = SkyImage.open(directory / filename)

    with ctx as hdul:
        # Parse timestamp from filename
        time_str = filename.split('UTC')[1][:6]
        time_str = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
        date_str = filename.split('UTC')[0][-8:]
        date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        time = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")

        # Get filtered catalog and predict positions
        az_range, el_range = SkyImage.get_azel_range(direction)
        slist = StarList()
        catalog = slist.filter_catalog(time, az_range, el_range)

        x_trans, y_trans = model.predict(catalog[["Az", "El"]])
        catalog["x_transform"], catalog["y_transform"] = x_trans, y_trans
        catalog["centroid_x"], catalog["centroid_y"], _, _ = centroid_starlist(hdul, catalog)

        # Plot
        h, w = hdul[0].data.shape[1], hdul[0].data.shape[2]
        plt.figure(figsize=(10, 7))
        plt.imshow(np.flip(np.transpose(hdul[0].data, (1, 2, 0)), axis=1))
        plt.ylim(h, 0)
        plt.xlim(0, w)
        plt.scatter(catalog["centroid_x"], catalog["centroid_y"], ec='pink', s=20, marker="s", fc="None", label="Centroided")
        plt.scatter(catalog["x_transform"], catalog["y_transform"], ec='red', s=30, marker="s", fc="None", label="Transformed")
        plt.legend()
        plt.savefig(out_dir / "centroid_vs_transform.png", dpi=150)
        plt.close()

def extract_centroids_with_brightness(
    model,
    direction: str,
    date: str,
    time_range: tuple = ((7, 0), (14, 20)),
    data_source: str = "local",
    brightness_cal: dict = None,
    remove_adjacent: bool = False,
) -> pd.DataFrame:
    """
    Process FITS images to get centroids with adjusted brightness.

    Args:
        model: Trained distortion model for coordinate transformation
        direction: Camera direction (North, South, East, West)
        date: Date in YYYYMMDD format
        time_range: ((start_hour, start_min), (end_hour, end_min)) tuple in UTC.
                    Default: ((7, 0), (14, 20)) UTC = 9:00 PM to 4:20 AM Hawaiian
        data_source: "local" for filesystem, "api" for remote API
        brightness_cal: Brightness calibration dict (required for gain/exp_time adjustment)

    Returns:
        DataFrame with columns: source, timestamp, TargetName, vmag, adjusted_brightness
    """
    if brightness_cal is None:
        raise ValueError("brightness_cal is required for brightness adjustment")

    if data_source == "api":
        all_files = SkyImage.get_file_list_api(direction, date)
        directory = None
    else:
        directory = PROJECT_ROOT / f"data/CloudCam{direction}/{date}"
        all_files = sorted(os.listdir(directory))
    az_range, el_range = SkyImage.get_azel_range(direction)

    def parse_timestamp(filename: str) -> datetime:
        time_str = filename.split("UTC")[1][:6]
        time_str = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
        date_str = filename.split("UTC")[0][-8:]
        date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")

    files_with_times = [(f, parse_timestamp(f)) for f in all_files]
    filtered = [(f, t) for f, t in files_with_times if _time_in_range(t, time_range[0], time_range[1])]

    print(f"Found {len(filtered)} images in time range, skipped {len(all_files) - len(filtered)}")

    all_rows = []
    slist = StarList()

    for filename, timestamp in tqdm(filtered, desc="Processing images"):
        if data_source == "api":
            ctx = SkyImage.open_api(direction, date, filename)
        else:
            ctx = SkyImage.open(directory / filename)

        with ctx as hdul:
            catalog = slist.filter_catalog(timestamp, az_range, el_range)
            x_pred, y_pred = model.predict(catalog[["Az", "El"]])
            catalog["x_transform"], catalog["y_transform"] = x_pred, y_pred

            if remove_adjacent:
                catalog = remove_adjacent_stars(catalog)

            # Measure brightness
            cx, cy, _, _, brightness = centroid_starlist(hdul, catalog, measure_brightness=True)
            gain = float(hdul[0].header["GAIN"])
            exp_time = float(hdul[0].header["EXP_TIME"])

            for i in range(len(catalog)):
                if cx[i] is not None and cy[i] is not None:
                    # Compute adjusted brightness using calibration's gain/exp_time model
                    adjusted = (
                        brightness[i]
                        - (gain * brightness_cal["gain_coef"] + exp_time * brightness_cal["exp_time_coef"] + brightness_cal["intercept"])
                        + brightness_cal["mean_brightness"]
                    )
                    # Skip invalid brightness values (NaN or negative)
                    if not (adjusted >= 0):
                        continue

                    vmag = catalog["vmag"][i]
                    # Filter to valid vmag range (same as calibration)
                    # Note: NaN comparisons return False, so explicitly check
                    if not (3.0 < vmag < 7.0):
                        continue

                    all_rows.append({
                        "source": filename,
                        "timestamp": timestamp,
                        "TargetName": catalog['TargetName'][i],
                        "vmag": vmag,
                        "adjusted_brightness": adjusted,
                    })

    return pd.DataFrame(all_rows)


def fit_vmag_regression(df: pd.DataFrame) -> dict:
    """
    Fit vmag vs adjusted_brightness using Pogson's law: vmag = -2.5 * log10(brightness) + intercept.

    Args:
        df: DataFrame with columns: vmag, adjusted_brightness

    Returns:
        Dictionary with keys: vmag_intercept, vmag_r2, vmag_std, num_centroids
    """
    # Drop any rows with NaN values and non-positive brightness
    clean_df = df[["adjusted_brightness", "vmag"]].dropna()
    clean_df = clean_df[clean_df["adjusted_brightness"] > 0]

    if len(clean_df) < 2:
        raise ValueError(f"Not enough valid data points for regression: {len(clean_df)}")

    log_brightness = np.log10(clean_df["adjusted_brightness"].values)

    # Initial fit: vmag = -2.5 * log10(brightness) + intercept
    vmag_intercept = np.mean(clean_df["vmag"].values - (-2.5 * log_brightness))

    # Remove outliers (>3 std from fit) and refit
    predicted_vmag = -2.5 * log_brightness + vmag_intercept
    residuals = clean_df["vmag"].values - predicted_vmag
    std = residuals.std()

    if std > 0:
        inlier_mask = np.abs(residuals) <= 3 * std
        clean_df = clean_df[inlier_mask]

    if len(clean_df) < 2:
        raise ValueError(f"Not enough inliers after outlier removal: {len(clean_df)}")

    # Refit on inliers
    log_brightness = np.log10(clean_df["adjusted_brightness"].values)
    vmag_intercept = np.mean(clean_df["vmag"].values - (-2.5 * log_brightness))
    predicted_vmag = -2.5 * log_brightness + vmag_intercept
    residuals = clean_df["vmag"].values - predicted_vmag
    vmag_std = residuals.std()

    # Compute R² (SS_res / SS_tot)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((clean_df["vmag"].values - clean_df["vmag"].values.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {
        "vmag_intercept": vmag_intercept,
        "vmag_r2": r2,
        "vmag_std": vmag_std,
        "num_centroids": len(clean_df),
    }

def save_brightness_regression(
    regression: dict,
    direction: str,
    date: str,
    measurement_type: str,
    output_dir: Path,
) -> Path:
    """
    Append brightness regression results to CSV file.

    Args:
        regression: Dictionary with regression parameters (vmag_intercept, vmag_r2, vmag_std, num_centroids)
        direction: Camera direction
        date: Date of measurement (YYYYMMDD format)
        measurement_type: "calibration" or "measurement"
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    evals_dir = output_dir / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)
    output_path = evals_dir / f"{direction.lower()}_brightness_regression.csv"

    row = {
        "date": date,
        "direction": direction,
        "type": measurement_type,
        "num_centroids": regression["num_centroids"],
        "vmag_intercept": regression["vmag_intercept"],
        "vmag_r2": regression["vmag_r2"],
        "vmag_std": regression["vmag_std"],
    }

    # Check if file exists to determine if we need headers
    file_exists = output_path.exists()

    df = pd.DataFrame([row])
    df.to_csv(output_path, mode='a', header=not file_exists, index=False)

    print(f"Appended brightness regression to: {output_path}")
    return output_path


def _plot_brightness_regression(df: pd.DataFrame, regression: dict, out_path: Path):
    """Generate brightness vs vmag scatter plot with fitted regression line and outlier boundaries."""
    _, ax = plt.subplots(figsize=(10, 7))

    # Compute predicted values and residuals to identify outliers
    predicted_vmag = -2.5 * np.log10(df["adjusted_brightness"].values.clip(min=1e-10)) + regression["vmag_intercept"]
    residuals = df["vmag"].values - predicted_vmag
    outlier_mask = np.abs(residuals) > 3 * regression["vmag_std"]

    # Save plot data to tmp
    tmp_dir = PROJECT_ROOT / "data" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    plot_data = df.copy()
    plot_data["predicted_vmag"] = predicted_vmag
    plot_data["residual"] = residuals
    plot_data["is_outlier"] = outlier_mask
    plot_data.to_csv(tmp_dir / "brightness_regression_data.csv", index=False)

    # Scatter plot with different colors for inliers and outliers
    inliers = df[~outlier_mask]
    outliers = df[outlier_mask]

    ax.scatter(inliers["adjusted_brightness"], inliers["vmag"], alpha=0.3, s=5, c='blue', label=f"Inliers (n={len(inliers)})")
    ax.scatter(outliers["adjusted_brightness"], outliers["vmag"], alpha=0.5, s=10, c='red', marker='x', label=f"Outliers (n={len(outliers)})")

    # Regression curve and 3-sigma boundaries
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


def evaluate_model(
    direction: str,
    date: str,
    output: Path = None,
    sample_index: int = 300,
    skip_sample: bool = False,
    data_source: str = "local",
    remove_adjacent: bool = False,
):
    """
    Fit brightness regression for a measurement night.

    Requires a pre-calibrated model (run calibrate.py first). Uses the calibration's
    gain/exp_time model to compute adjusted brightness, then fits a new vmag regression
    for this measurement night.

    Args:
        direction: Camera direction (North, South, East, West)
        date: Date in YYYYMMDD format
        output: Output directory (default: PROJECT_ROOT/out)
        sample_index: Index of sample image for visualization
        skip_sample: Skip sample image visualization
        data_source: "local" for filesystem, "api" for remote API

    Returns:
        dict: Regression parameters (vmag_intercept, vmag_r2, vmag_std, num_centroids)
    """
    output = output or PROJECT_ROOT / "out"
    output = Path(output)
    output.mkdir(exist_ok=True)

    # Load pre-calibrated model
    model = FisheyeCorrectionModel(direction=direction)
    print(f"Loaded Distort matrix for {direction}")

    # Load brightness calibration for gain/exp_time adjustment
    brightness_cal = load_brightness_calibration(direction)
    calibrate_date = brightness_cal.get("calibrate_date", "unknown")
    print(f"Using gain/exp_time calibration from: {calibrate_date}")

    # Sample visualization
    if not skip_sample:
        _visualize_sample(model, direction, date, sample_index, output, data_source)
        print(f"Sample visualization saved to {output / 'centroid_vs_transform.png'}")

    # Process images: extract centroids with adjusted brightness
    centroid_df = extract_centroids_with_brightness(
        model, direction, date, data_source=data_source, brightness_cal=brightness_cal,
        remove_adjacent=remove_adjacent,
    )
    print(f"Extracted {len(centroid_df)} centroids with brightness")

    # Fit vmag regression for this night
    regression = fit_vmag_regression(centroid_df)

    # Save regression to CSV
    save_brightness_regression(regression, direction, date, "measurement", output)

    # Generate regression plot
    _plot_brightness_regression(centroid_df, regression, output / "brightness_regression.png")
    print(f"Regression plot saved to {output / 'brightness_regression.png'}")

    # Print summary
    print(f"\n=== Brightness Regression Results for {direction} ===")
    print(f"  Date: {date}")
    print(f"  Centroids used: {regression['num_centroids']}")
    print(f"  vmag = -2.5 * log10(adjusted_brightness) + {regression['vmag_intercept']:.2f}")
    print(f"  R²: {regression['vmag_r2']:.4f}")
    print(f"  Std: {regression['vmag_std']:.4f}")

    return regression


def main():
    args = parse_args()

    # Validate direction
    if args.direction != "all" and args.direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS} or 'all'")

    # Get list of directions to process
    directions = DIRECTIONS if args.direction == "all" else [args.direction]

    for direction in directions:
        evaluate_model(
            direction=direction,
            date=args.date,
            output=args.output,
            sample_index=args.sample_index,
            skip_sample=args.skip_sample,
            data_source=args.data_source,
            remove_adjacent=args.remove_adjacent,
        )

if __name__ == "__main__":
    main()
