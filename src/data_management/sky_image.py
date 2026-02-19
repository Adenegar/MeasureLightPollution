from pathlib import Path
from typing import Optional, List
from typing import Literal
from contextlib import contextmanager
import io
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from astropy.io import fits
import requests
from config.settings import PROJECT_ROOT
import os

# API Configuration
SKY_API_HOST = "vm-internship2:9100"

# Server configuration by direction
# North and West: server 2, prefix /skycams2/
# East and South: server 1, prefix /skycams1/
SKY_API_SERVER_CONFIG = {
    "North": {"server_nr": 2, "prefix": "/skycams2/"},
    "West":  {"server_nr": 2, "prefix": "/skycams2/"},
    "East":  {"server_nr": 1, "prefix": "/skycams1/"},
    "South": {"server_nr": 1, "prefix": "/skycams1/"},
}

class SkyImage:
    
    def __init__(self, direction: Literal["North", "East", "South", "West"], date: datetime, image_index: int):
        """
        Initialize a SkyImage using image index in the fits file directory (use of SkyImage.open is encouraged over this)
        
        :param direction: Description
        :param date: Description
        :param image_index: Description
        """
        date_str = datetime.strftime(date, "%Y%m%d")
        fits_dir = PROJECT_ROOT / f"data/CloudCam{direction}/{date_str}"
        files_in_cloudcam = os.listdir(fits_dir)
        files_in_cloudcam.sort()
        fits_path = files_in_cloudcam[image_index]

        # Get the time for this image
        time_str = fits_path.split('UTC')[1][0:6]
        time_str = time_str[:2] + ':' + time_str[2:4] + ':' + time_str[4:]
        self.time = datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H:%M:%S")
        
        # Save the image
        self.hdul = fits.open(fits_dir / fits_path)

    @staticmethod
    def get_azel_range(direction: Literal["North", "East", "South", "West"]):
        az_range_map = {"North": (0 - 67, 0 + 68),
                        "East":  (90 - 73, 90 + 64),
                        "South": (180 - 70, 180 + 70),
                        "West":  (195, 330)}
        el_range = (0, 52)
        return az_range_map[direction], el_range

    @staticmethod
    def get_file_list_api(
        direction: Literal["North", "East", "South", "West"],
        date: str,
    ) -> List[str]:
        """
        Get list of FITS files from the API for a given direction and date.

        Args:
            direction: Camera direction (North, East, South, West)
            date: Date in YYYYMMDD format

        Returns:
            Sorted list of filenames
        """
        config = SKY_API_SERVER_CONFIG[direction]
        url = f"http://{SKY_API_HOST}/listDir"
        path = f"CloudCam{direction}/{date}/"
        params = {"path": path, "serverNr": config["server_nr"]}
        response = requests.get(url, params=params)
        response.raise_for_status()
        files = response.json()
        files.sort()
        return files

    @staticmethod
    @contextmanager
    def open_api(
        direction: Literal["North", "East", "South", "West"],
        date: str,
        filename: str,
    ):
        """
        Open a FITS file from the API as a context manager.

        Args:
            direction: Camera direction
            date: Date in YYYYMMDD format
            filename: FITS filename

        Yields:
            HDUList object
        """
        config = SKY_API_SERVER_CONFIG[direction]
        url = f"http://{SKY_API_HOST}/getAnyFile"
        path = f"{config['prefix']}CloudCam{direction}/{date}/{filename}"
        params = {"path": path}
        response = requests.get(url, params=params)
        response.raise_for_status()
        hdul = fits.open(io.BytesIO(response.content))
        try:
            yield hdul
        finally:
            hdul.close()

    @staticmethod
    @contextmanager
    def open(path: Optional[str|Path] = None, api: Optional[str] = None):
        """
        Open hdul using either a path or an API

        :param path: either absolute or relative path to the FITS file
        """
        if path:
            try:
                hdul = fits.open(str(path), mode='readonly')
            except Exception as e:
                print(f"Error opening FITS file at {path}: {e}")
                raise e 
            try:
                yield hdul 
            finally:
                hdul.close()
        elif api:
            raise NotImplementedError("Opening an HDUList using the API is not yet implemented")
        else:
            print("input parameters for path or api")
    
    def plot_centroid(self, x1, y1, x2, y2, center_x, center_y, linear_size=12, centroid_size=15):

        img = np.flip(np.transpose(self.hdul[0].data, (1, 2, 0)), axis=1)
        img_gray = img.mean(axis=2)

        image_height, image_width = self.hdul[0].data.shape[1], self.hdul[0].data.shape[2]

        #### Plot image
        if x1 >= 0 and x2 <= image_width and y1 >= 0 and y2 <= image_height:
            sub_img = img_gray[y1:y2, x1:x2]
            data_vis = np.clip(sub_img - np.median(sub_img), 0, None)
            
            fig = go.Figure(data=go.Heatmap(z=data_vis, colorscale='gray', showscale=True))
            fig.update_layout(width=600, height=600, 
                            title=f"Star Centroid")
            
            if center_x is not None and center_y is not None:
                fig.add_trace(go.Scatter(x=[center_x - x1], y=[center_y - y1], mode='markers',
                                        marker=dict(symbol='cross', size=centroid_size, color='red', line=dict(width=2)),
                                        name='Centroid'))
            
            x_pred = (x1 + x2) // 2
            y_pred = (y1 + y2) // 2
            fig.add_trace(go.Scatter(x=[x_pred - x1], y=[y_pred - y1], mode='markers',
                                    marker=dict(symbol='square', size=linear_size, color='yellow', line=dict(width=2)),
                                    name='Predicted'))
            
            return fig
        return None
                