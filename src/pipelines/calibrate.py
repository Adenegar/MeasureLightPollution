"""
Calibrate the fisheye distortion correction model.

Loads a FisheyeCorrectionModel from existing centroids, extracts new centroids
for brightness calibration, and saves the Distort matrix.

Usage:
    python calibrate.py --direction South --calibrate-date 20251207
    python calibrate.py --direction all --start-date all --calibrate-date 20251207

Arguments:
    --direction: Camera direction to calibrate (North, South, East, West, or "all")
    --start-date: Date(s) used to load the model (default: "all")
    --calibrate-date: Date(s) used for centroid extraction and brightness calibration
    --file-range: Range of file indices to process (default: 0 99999)
    --save-centroids: Save computed centroids to CSV (default: False)
    --data-source: Data source - "local" or "api" (default: "local")
"""

import argparse
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from astropy.coordinates import IllegalSecondWarning
from tqdm import tqdm

from config.settings import PROJECT_ROOT
from data_management import load_points, SkyImage, StarList
from models.centroid import centroid_starlist
from models.fisheye_correction import FisheyeCorrectionModel

warnings.filterwarnings("ignore", category=IllegalSecondWarning)

ADJACENT_DISTANCE_THRESHOLD = 16  # pixels


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

    # Compute pairwise distances using broadcasting
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)

    # Set diagonal to inf so a star doesn't match itself
    np.fill_diagonal(dist, np.inf)

    # A star is "adjacent" if its nearest neighbor is within threshold
    too_close = np.any(dist < threshold, axis=1)
    keep_mask = ~too_close

    removed = int(too_close.sum())
    if removed > 0:
        catalog = catalog[keep_mask]
    return catalog

DIRECTIONS = ["North", "East", "South", "West"]

# Default time range: 9 PM - 4:20 AM Hawaiian = 7:00 - 14:20 UTC
DEFAULT_TIME_RANGE = ((7, 0), (14, 20))


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


@dataclass
class CalibrationConfig:
    """Configuration for fisheye model calibration."""

    direction: Union[str, Literal["all"]] = "all"
    start_date: Union[str, Literal["all"]] = "all"
    calibrate_date: str = "20251207"
    file_range: Tuple[int, int] = (0, 99999)
    save_centroids: bool = False
    data_source: Literal["local", "api"] = "local"
    model_type: Literal["simple", "full"] = "full"
    output_dir: Optional[Path] = None
    time_range: Tuple[Tuple[int, int], Tuple[int, int]] = DEFAULT_TIME_RANGE
    calibrate_brightness: bool = True
    remove_adjacent: bool = False

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = PROJECT_ROOT / "data" / "Calibration"
        else:
            self.output_dir = Path(self.output_dir)

    @property
    def directions(self) -> List[str]:
        """Return list of directions to process."""
        if self.direction == "all":
            return DIRECTIONS
        return [self.direction]


def extract_centroids_for_direction(
    model: FisheyeCorrectionModel,
    direction: str,
    date: str,
    file_range: Tuple[int, int],
    time_range: Tuple[Tuple[int, int], Tuple[int, int]] = DEFAULT_TIME_RANGE,
    measure_brightness: bool = False,
    data_source: Literal["local", "api"] = "local",
    remove_adjacent: bool = False,
) -> pd.DataFrame:
    """
    Extract star centroids from FITS images for a single direction.

    Args:
        model: Trained fisheye correction model for predictions
        direction: Camera direction
        date: Date of FITS images (YYYYMMDD format)
        file_range: Range of file indices to process
        time_range: ((start_hour, start_min), (end_hour, end_min)) in UTC
        measure_brightness: Also measure brightness and collect metadata for calibration
        data_source: "local" for filesystem, "api" for remote API

    Returns:
        DataFrame with columns: source, TargetName, Az, El, x_pixel, y_pixel
        If measure_brightness=True, also includes: brightness, vmag, gain, exp_time
    """
    if data_source == "api":
        all_files = SkyImage.get_file_list_api(direction, date)
        data_dir = None  # Not used for API
    else:
        data_dir = PROJECT_ROOT / "data" / f"CloudCam{direction}" / date
        all_files = sorted(os.listdir(data_dir))
    az_range, el_range = SkyImage.get_azel_range(direction)

    # Apply file range
    files_in_range = all_files[file_range[0]:file_range[1]]

    # Filter by time range
    files_with_times = [(f, _parse_timestamp(f)) for f in files_in_range]
    filtered = [(f, t) for f, t in files_with_times
                if _time_in_range(t, time_range[0], time_range[1])]

    print(f"  Found {len(filtered)} files in time range (skipped {len(files_in_range) - len(filtered)})")

    slist = StarList()
    results = []

    for filename, time in tqdm(filtered, desc=f"Centroiding {direction}"):
        # Open file from API or local filesystem
        if data_source == "api":
            ctx = SkyImage.open_api(direction, date, filename)
        else:
            ctx = SkyImage.open(data_dir / filename)

        with ctx as hdul:
            # Get filtered catalog
            catalog = slist.filter_catalog(time, az_range, el_range)

            # Predict pixel coordinates
            x_pred, y_pred = model.predict(catalog[["Az", "El"]])
            catalog["x_transform"] = x_pred
            catalog["y_transform"] = y_pred

            if remove_adjacent:
                catalog = remove_adjacent_stars(catalog)

            # Perform centroiding (with brightness if requested)
            if measure_brightness:
                cx, cy, _, _, brightness = centroid_starlist(hdul, catalog, measure_brightness=True)
                gain = float(hdul[0].header["GAIN"])
                exp_time = float(hdul[0].header["EXP_TIME"])
            else:
                cx, cy, _, _ = centroid_starlist(hdul, catalog)

            # Collect successful centroids
            for i, star in enumerate(catalog):
                if cx[i] is not None and cy[i] is not None:
                    row = {
                        "source": filename,
                        "TargetName": star["TargetName"],
                        "Az": star["Az"],
                        "El": star["El"],
                        "x_pixel": cx[i],
                        "y_pixel": cy[i],
                    }
                    if measure_brightness:
                        row["brightness"] = brightness[i]
                        row["vmag"] = star["vmag"]
                        row["gain"] = gain
                        row["exp_time"] = exp_time
                    results.append(row)

    return pd.DataFrame(results)


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

    # Filter to valid vmag range (like images_to_brightness does)
    clean = df.dropna()
    clean = clean[(clean["vmag"] > 3.0) & (clean["vmag"] < 7.0)]

    # Fit gain/exp_time adjustment model
    X = clean[["gain", "exp_time"]].values
    y = clean["brightness"].values

    model = LinearRegression().fit(X, y)
    mean_brightness = y.mean()

    # Compute adjusted brightness
    predicted = model.predict(X)
    adjusted_brightness = y - predicted + mean_brightness

    # Filter out non-positive adjusted brightness (can't take log)
    valid_mask = adjusted_brightness > 0
    adjusted_brightness = adjusted_brightness[valid_mask]
    clean = clean[valid_mask]

    # Fit vmag prediction model: vmag = -2.5 * log10(brightness) + intercept
    log_brightness = np.log10(adjusted_brightness)
    vmag_intercept = np.mean(clean["vmag"].values - (-2.5 * log_brightness))

    # Remove outliers (>3 std from regression line) and refit
    predicted_vmag = -2.5 * log_brightness + vmag_intercept
    residuals = clean["vmag"].values - predicted_vmag
    std = residuals.std()
    inlier_mask = np.abs(residuals) <= 3 * std

    adjusted_brightness = adjusted_brightness[inlier_mask]
    clean = clean[inlier_mask]

    # Refit on inliers
    log_brightness = np.log10(adjusted_brightness)
    vmag_intercept = np.mean(clean["vmag"].values - (-2.5 * log_brightness))
    predicted_vmag = -2.5 * log_brightness + vmag_intercept
    residuals = clean["vmag"].values - predicted_vmag
    vmag_std = residuals.std()

    # Compute R²
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


def save_brightness_calibration(cal: dict, direction: str, calibrate_date: str, output_dir: Path, num_centroids: int = None) -> Path:
    """
    Save brightness calibration coefficients to file.

    Args:
        cal: Calibration dictionary from fit_brightness_calibration
        direction: Camera direction
        calibrate_date: Date used for calibration (YYYYMMDD format)
        output_dir: Output directory (will save to output_dir/cals/)
        num_centroids: Number of centroids used for calibration

    Returns:
        Path to saved file
    """
    import json

    cals_dir = output_dir / "cals"
    cals_dir.mkdir(parents=True, exist_ok=True)
    output_path = cals_dir / f"{direction.lower()}_brightness_cal.json"

    # Convert numpy arrays to lists for JSON serialization
    cal_json = {
        "calibrate_date": calibrate_date,
        "gain_coef": float(cal["coef"][0]),
        "exp_time_coef": float(cal["coef"][1]),
        "intercept": float(cal["intercept"]),
        "mean_brightness": float(cal["mean_brightness"]),
        "vmag_intercept": float(cal["vmag_intercept"]),
        "vmag_r2": float(cal["vmag_r2"]),
        "vmag_std": float(cal["vmag_std"]),
    }

    with open(output_path, "w") as f:
        json.dump(cal_json, f, indent=2)

    print(f"Saved brightness calibration to: {output_path}")

    # Also save to the brightness regression CSV
    _save_brightness_regression_csv(cal, direction, calibrate_date, output_dir, num_centroids)

    return output_path


def _save_brightness_regression_csv(cal: dict, direction: str, date: str, output_dir: Path, num_centroids: int = None) -> Path:
    """
    Append brightness regression results to CSV file for calibration night.

    Args:
        cal: Calibration dictionary from fit_brightness_calibration
        direction: Camera direction
        date: Date of calibration (YYYYMMDD format)
        output_dir: Output directory
        num_centroids: Number of centroids used

    Returns:
        Path to saved file
    """
    evals_dir = output_dir / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)
    output_path = evals_dir / f"{direction.lower()}_brightness_regression.csv"

    row = {
        "date": date,
        "direction": direction,
        "type": "calibration",
        "num_centroids": num_centroids if num_centroids is not None else 0,
        "vmag_intercept": float(cal["vmag_intercept"]),
        "vmag_r2": float(cal["vmag_r2"]),
        "vmag_std": float(cal["vmag_std"]),
    }

    # Check if file exists to determine if we need headers
    file_exists = output_path.exists()

    df = pd.DataFrame([row])
    df.to_csv(output_path, mode='a', header=not file_exists, index=False)

    print(f"Appended calibration to brightness regression CSV: {output_path}")
    return output_path


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
    import json

    if cals_dir is None:
        cals_dir = PROJECT_ROOT / "out" / "cals"
    cal_path = cals_dir / f"{direction.lower()}_brightness_cal.json"

    if not cal_path.exists():
        raise FileNotFoundError(
            f"No brightness calibration found at {cal_path}. "
            f"Run calibrate.py for {direction} first."
        )

    with open(cal_path) as f:
        return json.load(f)


def calibrate_direction(
    direction: str,
    config: CalibrationConfig,
) -> Tuple[FisheyeCorrectionModel, Optional[pd.DataFrame], Optional[dict]]:
    """
    Calibrate the fisheye model for a single direction.

    Loads model from existing centroids, extracts new centroids for brightness
    calibration, and saves the original model (no retraining on new centroids).

    Args:
        direction: Camera direction
        config: Calibration configuration

    Returns:
        Tuple of (model, centroids DataFrame or None, brightness calibration or None)
    """
    print(f"\n{'='*60}")
    print(f"Calibrating {direction} camera")
    print(f"{'='*60}")

    # Step 1: Load model from existing points
    start_date_param = None if config.start_date == "all" else config.start_date

    try:
        X_train, Y_train = load_points("Centroid", direction, start_date_param)
        model = FisheyeCorrectionModel(model_type=config.model_type)
        model.train(X_train, Y_train)
        model.train_inverse(Y_train, X_train)
        date_info = "all dates" if config.start_date == "all" else config.start_date
        print(f"Loaded model with {len(X_train)} points from {date_info}")
    except FileNotFoundError:
        try:
            X_train, Y_train = load_points("Manual", direction, start_date_param)
            model = FisheyeCorrectionModel(model_type=config.model_type)
            model.train(X_train, Y_train)
            model.train_inverse(Y_train, X_train)
            print(f"Loaded model with {len(X_train)} manual points")
        except FileNotFoundError:
            raise ValueError(
                f"No calibration data found for {direction}. "
                "Need either Centroid or Manual data to start."
            )

    # Step 2: Extract centroids for brightness calibration
    print(f"\nExtracting centroids from {config.calibrate_date}...")
    centroids_df = extract_centroids_for_direction(
        model,
        direction,
        config.calibrate_date,
        config.file_range,
        config.time_range,
        measure_brightness=config.calibrate_brightness,
        data_source=config.data_source,
        remove_adjacent=config.remove_adjacent,
    )
    print(f"Extracted {len(centroids_df)} centroids")

    # Step 3: Brightness calibration if requested
    brightness_cal = None
    if config.calibrate_brightness:
        print(f"\nFitting brightness calibration...")
        brightness_cal = fit_brightness_calibration(centroids_df)
        print(f"  Gain coef: {brightness_cal['coef'][0]:.4f}")
        print(f"  Exp_time coef: {brightness_cal['coef'][1]:.4f}")
        print(f"  Mean brightness: {brightness_cal['mean_brightness']:.4f}")
        print(f"  Vmag formula: vmag = -2.5 * log10(adjusted_brightness) + {brightness_cal['vmag_intercept']:.4f}")
        print(f"  Vmag R²: {brightness_cal['vmag_r2']:.4f}")

    return model, centroids_df if config.save_centroids else None, brightness_cal


def save_distort_matrix(
    model: FisheyeCorrectionModel,
    direction: str,
    output_dir: Path,
) -> Path:
    """
    Save the Distort matrix to a numpy file.

    Args:
        model: Trained model with Distort matrix
        direction: Camera direction
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{direction.lower()}_distort.npy"
    np.save(output_path, model.Distort)
    print(f"Saved Distort matrix to: {output_path}")
    return output_path


def save_distort_inv_matrix(
    model: FisheyeCorrectionModel,
    direction: str,
    output_dir: Path,
) -> Path:
    """
    Save the inverse Distort matrix to a numpy file.

    Args:
        model: Trained model with Distort_inv matrix
        direction: Camera direction
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{direction.lower()}_distort_inv.npy"
    np.save(output_path, model.Distort_inv)
    print(f"Saved Distort_inv matrix to: {output_path}")
    return output_path


def save_centroids_csv(
    centroids_df: pd.DataFrame,
    direction: str,
    date: str,
    output_dir: Path,
) -> Path:
    """
    Save centroids to CSV file.

    Args:
        centroids_df: DataFrame with centroid data
        direction: Camera direction
        date: Calibration date
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    date_dir = output_dir / date
    date_dir.mkdir(parents=True, exist_ok=True)
    output_path = date_dir / f"{direction.lower()}_centroid_mapping.csv"
    centroids_df.to_csv(output_path, index=False)
    print(f"Saved centroids to: {output_path}")
    return output_path


def run_calibration(config: CalibrationConfig) -> dict:
    """
    Run the complete calibration pipeline.

    Args:
        config: Calibration configuration

    Returns:
        Dictionary mapping direction to trained model
    """
    print(f"Starting calibration pipeline")
    print(f"  Directions: {config.directions}")
    print(f"  Start date: {config.start_date}")
    print(f"  Calibrate date: {config.calibrate_date}")
    print(f"  File range: {config.file_range}")
    print(f"  Time range (UTC): {config.time_range[0][0]:02d}:{config.time_range[0][1]:02d} - {config.time_range[1][0]:02d}:{config.time_range[1][1]:02d}")
    print(f"  Calibrate brightness: {config.calibrate_brightness}")
    print(f"  Save centroids: {config.save_centroids}")

    models = {}

    for direction in config.directions:
        model, centroids_df, brightness_cal = calibrate_direction(direction, config)
        models[direction] = model

        # Save Distort matrices
        save_distort_matrix(model, direction, config.output_dir)
        save_distort_inv_matrix(model, direction, config.output_dir)

        # Save centroids if requested
        if centroids_df is not None:
            save_centroids_csv(
                centroids_df, direction, config.calibrate_date, config.output_dir
            )

        # Save brightness calibration if computed
        if brightness_cal is not None:
            save_brightness_calibration(brightness_cal, direction, config.calibrate_date, PROJECT_ROOT / "out", brightness_cal["num_centroids"])

    print(f"\n{'='*60}")
    print("Calibration complete!")
    print(f"{'='*60}")

    return models


def main():
    """Main entry point for terminal execution."""
    parser = argparse.ArgumentParser(
        description="Calibrate fisheye distortion correction model"
    )

    parser.add_argument(
        "--direction",
        type=str,
        default="all",
        help="Camera direction (North, South, East, West, or 'all'). Default: all",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default="all",
        help="Date(s) for initial model training (YYYYMMDD or 'all'). Default: all",
    )

    parser.add_argument(
        "--calibrate-date",
        type=str,
        required=True,
        help="Date for calibration centroiding (YYYYMMDD format)",
    )

    parser.add_argument(
        "--file-range",
        type=int,
        nargs=2,
        default=[0, 99999],
        metavar=("START", "END"),
        help="Range of file indices to process. Default: 0 99999 (all files)",
    )

    parser.add_argument(
        "--save-centroids",
        action="store_true",
        help="Save computed centroids to CSV. Default: False",
    )

    parser.add_argument(
        "--data-source",
        type=str,
        default="local",
        choices=["local", "api"],
        help="Data source. Default: local",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="full",
        choices=["simple", "full"],
        help="Model complexity. Default: full",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Custom output directory path",
    )

    parser.add_argument(
        "--time-range",
        type=int,
        nargs=4,
        default=[7, 0, 14, 20],
        metavar=("START_H", "START_M", "END_H", "END_M"),
        help="Time range in UTC (start_hour start_min end_hour end_min). Default: 7 0 14 20 (9PM-4:20AM Hawaiian)",
    )

    parser.add_argument(
        "--skip-brightness-calibration",
        action="store_true",
        help="Skip brightness calibration for gain/exp_time adjustment. Default: False (calibration is performed)",
    )

    parser.add_argument(
        "--remove-adjacent",
        action="store_true",
        help="Remove stars within 13 pixels of another star (both removed). Default: False",
    )

    args = parser.parse_args()

    # Validate direction
    if args.direction != "all" and args.direction not in DIRECTIONS:
        parser.error(f"direction must be one of {DIRECTIONS} or 'all'")

    # Parse time range
    time_range = (
        (args.time_range[0], args.time_range[1]),
        (args.time_range[2], args.time_range[3]),
    )

    # Create configuration
    config = CalibrationConfig(
        direction=args.direction,
        start_date=args.start_date,
        calibrate_date=args.calibrate_date,
        file_range=tuple(args.file_range),
        save_centroids=args.save_centroids,
        data_source=args.data_source,
        model_type=args.model_type,
        output_dir=args.output_dir,
        time_range=time_range,
        calibrate_brightness=not args.skip_brightness_calibration,
        remove_adjacent=args.remove_adjacent,
    )

    # Run calibration
    run_calibration(config)


if __name__ == "__main__":
    main()
