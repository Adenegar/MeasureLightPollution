from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
from astropy.table import Table

from config.settings import PROJECT_ROOT


class FisheyeCorrectionModel:
    """
    Fisheye lens distortion correction model using polynomial feature transformations.

    Supports multiple model complexities:
    - "simple": Basic polynomial terms (9 features)
    - "full": 1st-3rd degree full cross terms (24 features) - DEFAULT

    Can be instantiated in two ways:
    1. Empty model for training: FisheyeCorrectionModel()
    2. Load pre-trained model: FisheyeCorrectionModel(direction="South")
    """

    def __init__(
        self,
        direction: Optional[str] = None,
        model_type: Literal["simple", "full"] = "full",
    ):
        """
        Initialize the fisheye correction model.

        Args:
            direction: Camera direction (North, South, East, West). If provided,
                      loads the pre-trained Distort matrix from disk.
            model_type: Type of model to use
                - "simple": Basic polynomial terms with selected cross terms
                - "full": 1st-3rd degree full cross terms (default)
        """
        self.model_type = model_type
        self.direction = direction
        self.Distort = None

        if direction is not None:
            self._load_distort(direction)

    def _load_distort(self, direction: str) -> None:
        """
        Load the Distort matrix from disk.

        Args:
            direction: Camera direction (North, South, East, West)

        Raises:
            FileNotFoundError: If no saved Distort matrix exists for the direction
        """
        distort_path = PROJECT_ROOT / "data" / "Calibration" / f"{direction.lower()}_distort.npy"
        if not distort_path.exists():
            raise FileNotFoundError(
                f"No saved Distort matrix found at {distort_path}. "
                f"Run calibrate.py for {direction} first."
            )
        self.Distort = np.load(distort_path)

    def train(self, X_train: Table, Y_train: Table) -> None:
        """
        Train the distortion correction model using least squares.

        Args:
            X_train: Table with Az and El columns
            Y_train: Table with x_actual and y_actual columns (pixel coordinates)
        """
        # Generate features
        X_train = self._generate_features(X_train.copy())

        # Convert to numpy arrays
        X = X_train.to_pandas().to_numpy()
        Y = np.column_stack([Y_train["x_actual"], Y_train["y_actual"]])

        # Solve least squares: minimize ||X @ Distort - Y||^2
        self.Distort, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)

    def predict(self, X_test: Table) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict pixel coordinates from Az/El using the trained model.

        Args:
            X_test: Table with Az and El columns

        Returns:
            Tuple of (x_corrected, y_corrected) arrays
        """
        if self.Distort is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        # Generate features
        X_test = self._generate_features(X_test.copy())

        # Convert to numpy array
        X = X_test.to_pandas().to_numpy()

        # Apply the distortion correction
        predictions = X @ self.Distort  # Shape: (n_samples, 2)

        # Extract x and y predictions
        x_corrected = predictions[:, 0]
        y_corrected = predictions[:, 1]

        return x_corrected, y_corrected

    def _generate_features_simple(self, data: Table) -> Table:
        """
        Generate simple polynomial features.

        Features (9 total):
        - 1st degree: sinAz, sinEl
        - 2nd degree: cosEl_cosEl, sinAz_cosEl
        - 3rd degree: sinAz^3, sinAz^2*sinEl, sinAz*cosEl^2, sinEl^3, cosAz*cosEl^2
        """
        # 1st degree terms
        data["sinAz"] = np.sin(np.radians(data["Az"]))
        data["sinEl"] = np.sin(np.radians(data["El"]))

        # 2nd degree term
        data["cosEl_cosEl"] = np.cos(np.radians(data["El"])) ** 2
        data["sinAz_cosEl"] = np.sin(np.radians(data["Az"])) * np.cos(np.radians(data["El"]))

        # 3rd degree terms
        data["sinAz_sinAz_sinAz"] = np.sin(np.radians(data["Az"])) ** 3
        data["sinAz_sinAz_sinEl"] = np.sin(np.radians(data["Az"])) ** 2 * np.sin(np.radians(data["El"]))
        data["sinAz_cosEl_cosEl"] = np.sin(np.radians(data["Az"])) * np.cos(np.radians(data["El"])) ** 2
        data["sinEl_sinEl_sinEl"] = np.sin(np.radians(data["El"])) ** 3
        data["cosAz_cosEl_cosEl"] = np.cos(np.radians(data["Az"])) * np.cos(np.radians(data["El"])) ** 2

        # Intercept term
        data["shift"] = 1

        return data

    def _generate_features_full(self, data: Table) -> Table:
        """
        Generate full 1st-3rd degree polynomial features with all cross terms.

        Features (24 total):
        - 1st degree: sinAz, cosAz, sinEl, cosEl (4)
        - 2nd degree: All cross terms (8)
        - 3rd degree: All cross terms (20)
        """
        # 1st degree terms
        data["sinAz"] = np.sin(np.radians(data["Az"]))
        data["cosAz"] = np.cos(np.radians(data["Az"]))
        data["sinEl"] = np.sin(np.radians(data["El"]))
        data["cosEl"] = np.cos(np.radians(data["El"]))

        # 2nd degree cross terms
        data["sinAz_sinAz"] = data["sinAz"] * data["sinAz"]
        data["sinAz_sinEl"] = data["sinAz"] * data["sinEl"]
        data["sinEl_sinEl"] = data["sinEl"] * data["sinEl"]
        data["cosAz_cosAz"] = data["cosAz"] * data["cosAz"]
        data["cosAz_cosEl"] = data["cosAz"] * data["cosEl"]
        data["cosEl_cosEl"] = data["cosEl"] * data["cosEl"]
        data["sinAz_cosEl"] = data["sinAz"] * data["cosEl"]
        data["cosAz_sinEl"] = data["cosAz"] * data["sinEl"]

        # 3rd degree cross terms
        data["sinAz_sinAz_sinAz"] = data["sinAz"] * data["sinAz"] * data["sinAz"]
        data["sinAz_sinAz_sinEl"] = data["sinAz"] * data["sinAz"] * data["sinEl"]
        data["sinAz_sinAz_cosAz"] = data["sinAz"] * data["sinAz"] * data["cosAz"]
        data["sinAz_sinAz_cosEl"] = data["sinAz"] * data["sinAz"] * data["cosEl"]
        data["sinAz_sinEl_sinEl"] = data["sinAz"] * data["sinEl"] * data["sinEl"]
        data["sinAz_sinEl_cosAz"] = data["sinAz"] * data["sinEl"] * data["cosAz"]
        data["sinAz_sinEl_cosEl"] = data["sinAz"] * data["sinEl"] * data["cosEl"]
        data["sinAz_cosAz_cosAz"] = data["sinAz"] * data["cosAz"] * data["cosAz"]
        data["sinAz_cosAz_cosEl"] = data["sinAz"] * data["cosAz"] * data["cosEl"]
        data["sinAz_cosEl_cosEl"] = data["sinAz"] * data["cosEl"] * data["cosEl"]
        data["sinEl_sinEl_sinEl"] = data["sinEl"] * data["sinEl"] * data["sinEl"]
        data["sinEl_sinEl_cosAz"] = data["sinEl"] * data["sinEl"] * data["cosAz"]
        data["sinEl_sinEl_cosEl"] = data["sinEl"] * data["sinEl"] * data["cosEl"]
        data["sinEl_cosAz_cosAz"] = data["sinEl"] * data["cosAz"] * data["cosAz"]
        data["sinEl_cosAz_cosEl"] = data["sinEl"] * data["cosAz"] * data["cosEl"]
        data["sinEl_cosEl_cosEl"] = data["sinEl"] * data["cosEl"] * data["cosEl"]
        data["cosAz_cosAz_cosAz"] = data["cosAz"] * data["cosAz"] * data["cosAz"]
        data["cosAz_cosAz_cosEl"] = data["cosAz"] * data["cosAz"] * data["cosEl"]
        data["cosAz_cosEl_cosEl"] = data["cosAz"] * data["cosEl"] * data["cosEl"]
        data["cosEl_cosEl_cosEl"] = data["cosEl"] * data["cosEl"] * data["cosEl"]

        # Intercept term
        data["shift"] = 1

        return data

    def _generate_features(self, data: Table) -> Table:
        """
        Generate features based on the selected model type.

        Args:
            data: Table with Az and El columns

        Returns:
            Table with added feature columns
        """
        if self.model_type == "simple":
            return self._generate_features_simple(data)
        elif self.model_type == "full":
            return self._generate_features_full(data)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}. Must be 'simple' or 'full'")
