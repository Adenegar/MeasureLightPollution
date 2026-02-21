"""
Calibrate the fisheye distortion correction model (AzEl -> xy transformation matrix).

Loads calibration points (Manual, Centroid, or Both), trains a FisheyeCorrectionModel,
and saves the Distort matrix.

Usage:
    python calibrate.py --direction South --point-source Centroid
    python calibrate.py --direction all --point-source Both
    python calibrate.py --direction North --point-source Manual --start-date 20260118

Arguments:
    --direction: Camera direction to calibrate (North, South, East, West, or "all")
    --point-source: Which calibration points to load - "Manual", "Centroid", or "Both"
    --start-date: Date(s) used to load points (default: "all" = all dates)
    --model-type: Model complexity - "simple" or "full" (default: "full")
"""

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
from astropy.coordinates import IllegalSecondWarning

from config.settings import PROJECT_ROOT
from data_management import load_points
from models.fisheye_correction import FisheyeCorrectionModel

warnings.filterwarnings("ignore", category=IllegalSecondWarning)

DIRECTIONS = ["North", "East", "South", "West"]


@dataclass
class CalibrationConfig:
    """Configuration for fisheye model calibration."""

    direction: Union[str, Literal["all"]] = "all"
    point_source: Literal["Manual", "Centroid", "Both"] = "Centroid"
    start_date: Union[str, Literal["all"]] = "all"
    model_type: Literal["simple", "full"] = "full"
    output_dir: Optional[Path] = None

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = PROJECT_ROOT / "data" / "Calibration" / "matrices"
        else:
            self.output_dir = Path(self.output_dir)

    @property
    def directions(self) -> List[str]:
        """Return list of directions to process."""
        if self.direction == "all":
            return DIRECTIONS
        return [self.direction]


def calibrate_direction(
    direction: str,
    config: CalibrationConfig,
) -> FisheyeCorrectionModel:
    """
    Calibrate the fisheye model for a single direction.

    Loads calibration points and trains the distortion model.

    Args:
        direction: Camera direction
        config: Calibration configuration

    Returns:
        Trained FisheyeCorrectionModel
    """
    print(f"\n{'='*60}")
    print(f"Calibrating {direction} camera")
    print(f"{'='*60}")

    start_date_param = None if config.start_date == "all" else config.start_date

    X_train, Y_train = load_points(config.point_source, direction, start_date_param)
    model = FisheyeCorrectionModel(model_type=config.model_type)
    model.train(X_train, Y_train)

    date_info = "all dates" if config.start_date == "all" else config.start_date
    print(f"Loaded {len(X_train)} {config.point_source} points from {date_info}")

    return model


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
    print(f"  Point source: {config.point_source}")
    print(f"  Start date: {config.start_date}")
    print(f"  Model type: {config.model_type}")

    models = {}

    for direction in config.directions:
        model = calibrate_direction(direction, config)
        models[direction] = model

        save_distort_matrix(model, direction, config.output_dir)

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
        "--direction", type=str, default="all",
        help="Camera direction (North, South, East, West, or 'all'). Default: all",
    )
    parser.add_argument(
        "--point-source", type=str, default="Centroid",
        choices=["Manual", "Centroid", "Both"],
        help="Which calibration points to load. Default: Centroid",
    )
    parser.add_argument(
        "--start-date", type=str, default="all",
        help="Date(s) for point loading (YYYYMMDD or 'all'). Default: all",
    )
    parser.add_argument(
        "--model-type", type=str, default="full",
        choices=["simple", "full"],
        help="Model complexity. Default: full",
    )
    parser.add_argument(
        "--output-dir", type=str,
        help="Custom output directory path",
    )

    args = parser.parse_args()

    if args.direction != "all" and args.direction not in DIRECTIONS:
        parser.error(f"direction must be one of {DIRECTIONS} or 'all'")

    config = CalibrationConfig(
        direction=args.direction,
        point_source=args.point_source,
        start_date=args.start_date,
        model_type=args.model_type,
        output_dir=args.output_dir,
    )

    run_calibration(config)


if __name__ == "__main__":
    main()
