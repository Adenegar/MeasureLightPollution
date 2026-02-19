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
        self.Distort_inv = None

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

        inv_path = PROJECT_ROOT / "data" / "Calibration" / f"{direction.lower()}_distort_inv.npy"
        if inv_path.exists():
            self.Distort_inv = np.load(inv_path)

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

    def train_inverse(self, X_train: Table, Y_train: Table) -> None:
        """
        Train the inverse model: pixel coordinates -> Az/El.

        Args:
            X_train: Table with x_actual and y_actual columns (pixel coordinates)
            Y_train: Table with Az and El columns
        """
        X_train = self._generate_pixel_features(X_train.copy())
        X = X_train.to_pandas().to_numpy()
        Y = np.column_stack([Y_train["Az"], Y_train["El"]])
        self.Distort_inv, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)

    def predict_inverse(self, X_test: Table) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict Az/El from pixel coordinates using the inverse model.

        Args:
            X_test: Table with x_actual and y_actual columns

        Returns:
            Tuple of (Az, El) arrays
        """
        if self.Distort_inv is None:
            raise ValueError("Inverse model has not been trained yet. Call train_inverse() first.")

        X_test = self._generate_pixel_features(X_test.copy())
        X = X_test.to_pandas().to_numpy()
        predictions = X @ self.Distort_inv
        return predictions[:, 0], predictions[:, 1]

    def _generate_pixel_features(self, data: Table) -> Table:
        """
        Generate trig-based features from pixel coordinates.

        Uses polar decomposition in the image plane:
        - phi = atan2(y, x): azimuth angle in the image plane
        - rho = atan(r): bounded radial angle from the optical axis

        Features mirror the forward model: sinPhi, cosPhi, sinRho, cosRho
        and their 2nd/3rd degree cross terms (33 features + intercept).
        """
        x = data["x_actual"]
        y = data["y_actual"]
        r = np.sqrt(x**2 + y**2)

        phi = np.arctan2(y, x)
        rho = np.arctan(r)

        # 1st degree
        data["sinPhi"] = np.sin(phi)
        data["cosPhi"] = np.cos(phi)
        data["sinRho"] = np.sin(rho)
        data["cosRho"] = np.cos(rho)

        # 2nd degree cross terms (mirrors forward model structure)
        data["sinPhi_sinPhi"] = data["sinPhi"] * data["sinPhi"]
        data["sinPhi_sinRho"] = data["sinPhi"] * data["sinRho"]
        data["sinRho_sinRho"] = data["sinRho"] * data["sinRho"]
        data["cosPhi_cosPhi"] = data["cosPhi"] * data["cosPhi"]
        data["cosPhi_cosRho"] = data["cosPhi"] * data["cosRho"]
        data["cosRho_cosRho"] = data["cosRho"] * data["cosRho"]
        data["sinPhi_cosRho"] = data["sinPhi"] * data["cosRho"]
        data["cosPhi_sinRho"] = data["cosPhi"] * data["sinRho"]

        # 3rd degree cross terms (mirrors forward model structure)
        data["sinPhi_sinPhi_sinPhi"] = data["sinPhi"] * data["sinPhi"] * data["sinPhi"]
        data["sinPhi_sinPhi_sinRho"] = data["sinPhi"] * data["sinPhi"] * data["sinRho"]
        data["sinPhi_sinPhi_cosPhi"] = data["sinPhi"] * data["sinPhi"] * data["cosPhi"]
        data["sinPhi_sinPhi_cosRho"] = data["sinPhi"] * data["sinPhi"] * data["cosRho"]
        data["sinPhi_sinRho_sinRho"] = data["sinPhi"] * data["sinRho"] * data["sinRho"]
        data["sinPhi_sinRho_cosPhi"] = data["sinPhi"] * data["sinRho"] * data["cosPhi"]
        data["sinPhi_sinRho_cosRho"] = data["sinPhi"] * data["sinRho"] * data["cosRho"]
        data["sinPhi_cosPhi_cosPhi"] = data["sinPhi"] * data["cosPhi"] * data["cosPhi"]
        data["sinPhi_cosPhi_cosRho"] = data["sinPhi"] * data["cosPhi"] * data["cosRho"]
        data["sinPhi_cosRho_cosRho"] = data["sinPhi"] * data["cosRho"] * data["cosRho"]
        data["sinRho_sinRho_sinRho"] = data["sinRho"] * data["sinRho"] * data["sinRho"]
        data["sinRho_sinRho_cosPhi"] = data["sinRho"] * data["sinRho"] * data["cosPhi"]
        data["sinRho_sinRho_cosRho"] = data["sinRho"] * data["sinRho"] * data["cosRho"]
        data["sinRho_cosPhi_cosPhi"] = data["sinRho"] * data["cosPhi"] * data["cosPhi"]
        data["sinRho_cosPhi_cosRho"] = data["sinRho"] * data["cosPhi"] * data["cosRho"]
        data["sinRho_cosRho_cosRho"] = data["sinRho"] * data["cosRho"] * data["cosRho"]
        data["cosPhi_cosPhi_cosPhi"] = data["cosPhi"] * data["cosPhi"] * data["cosPhi"]
        data["cosPhi_cosPhi_cosRho"] = data["cosPhi"] * data["cosPhi"] * data["cosRho"]
        data["cosPhi_cosRho_cosRho"] = data["cosPhi"] * data["cosRho"] * data["cosRho"]
        data["cosRho_cosRho_cosRho"] = data["cosRho"] * data["cosRho"] * data["cosRho"]

        # Intercept term
        data["shift"] = 1

        return data

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
