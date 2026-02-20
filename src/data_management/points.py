from astropy.table import Table, vstack
from typing import Literal, Optional, Tuple
from config.settings import PROJECT_ROOT
from pathlib import Path


def load_points(
    method: Literal["Manual", "Centroid", "Both"],
    direction: Literal["North", "East", "South", "West"],
    date: Optional[str] = None
) -> Tuple[Table, Table]:
    """
    Load calibration points for training distortion model.

    Args:
        method: Data source - "Manual", "Centroid", or "Both"
        direction: Camera direction
        date: Optional date string (YYYYMMDD format). If None, loads all dates.

    Returns:
        Tuple of (X_train, Y_train) where X_train contains Az/El and Y_train contains pixel coordinates
    """
    points_folder = PROJECT_ROOT / "data/Calibration"

    # Determine which date folders to load from
    if date is None:
        # Load from all date folders
        date_folders = [d for d in points_folder.iterdir() if d.is_dir() and d.name.isdigit()]
        if not date_folders:
            raise FileNotFoundError(f"No date folders found in {points_folder}")
    else:
        # Load from specific date folder
        date_folders = [points_folder / date]
        if not date_folders[0].exists():
            raise FileNotFoundError(f"Date folder {date_folders[0]} does not exist")

    def load_single_method(method_name: str, date_folder: Path) -> Optional[Table]:
        """Load data for a single method from a date folder."""
        file_path = date_folder / f"{direction.lower()}_{method_name.lower()}_mapping.csv"
        if file_path.exists():
            data = Table.read(str(file_path), format='ascii.csv')
            data.rename_columns(
                names=["x_pixel", "y_pixel"],
                new_names=["x_actual", "y_actual"]
            )
            return data
        return None

    # Collect all data across dates
    all_data = []

    for date_folder in date_folders:
        if method == "Manual" or method == "Centroid":
            data = load_single_method(method, date_folder)
            if data is not None:
                all_data.append(data)

        elif method == "Both":
            manual_data = load_single_method("Manual", date_folder)
            centroid_data = load_single_method("Centroid", date_folder)

            if manual_data is not None:
                all_data.append(manual_data)
            if centroid_data is not None:
                all_data.append(centroid_data)

    if not all_data:
        raise FileNotFoundError(
            f"No data found for method={method}, direction={direction}, date={date}"
        )

    # Combine all data
    if len(all_data) == 1:
        combined_data = all_data[0]
    else:
        combined_data = vstack(all_data)

    return combined_data[["Az", "El"]], combined_data[["x_actual", "y_actual"]]
    