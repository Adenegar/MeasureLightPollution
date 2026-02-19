from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.table import Table
from astropy.time import Time
from astropy.io import ascii
from astropy import units as u
from typing import Tuple
from datetime import datetime
from config.settings import PROJECT_ROOT

from astropy.coordinates.angles.formats import IllegalSecondWarning
import warnings

class StarList:
    catalog: Table = None
    full_catalog: Table = None
    
    latitude = 19.826251721335527 * u.deg
    longitude = -155.47451249450162 * u.deg
    observing_location = EarthLocation(lat=latitude, lon=longitude)

    def __init__(self, catalog_name="data/StarLists/ybsc5.starlist"):
        self.load_catalog(catalog_name)

    def load_catalog(self, catalog_name):
        """
        Load a starlist into an astropy Table. Designed for usno-bsc and ybsc5 starlists.

        Args:
            catalog_name: Star catalog location from PROJECT_ROOT
        """
        with warnings.catch_warnings(): 
            warnings.simplefilter('ignore', IllegalSecondWarning) # Because some entries in the starlist have 60.0 seconds rather than +1 minute a warning is raised
            starlist = PROJECT_ROOT / catalog_name
            # Load stars from star catalog
            self.catalog = ascii.read(
                str(starlist),
                names=["TargetName", "RaH", "RaM", "RaS", "DecD", "DecM", "DecS", "Year", "vmag", "pmRA", "pmDEC"]
            )
            # Extract numeric values from keyword arguments
            self.catalog['vmag'] = [float(val.split('=')[1]) for val in self.catalog['vmag']]
            self.catalog['pmRA'] = [float(val.split('=')[1]) for val in self.catalog['pmRA']]
            self.catalog['pmDEC'] = [float(val.split('=')[1]) for val in self.catalog['pmDEC']]
            self.full_catalog = self.catalog.copy()
            self.full_catalog.sort("vmag")

            # Pre-compute RA/Dec coordinates to avoid recalculating on each get_azel_for_time call
            ra_strings = [f"{row['RaH']}h{row['RaM']}m{row['RaS']}s" for row in self.full_catalog]
            dec_strings = [f"{row['DecD']}d{row['DecM']}m{row['DecS']}s" for row in self.full_catalog]
            self.ra_dec_coords = SkyCoord(ra=ra_strings, dec=dec_strings, frame='icrs')
    
    def filter_catalog(
        self,
        time: datetime,
        az_range: Tuple[float, float],
        el_range: Tuple[float, float],
        top_k: int = None
    ) -> Table:
        """
        Filter the star catalog based on time, Az/El ranges, and brightness.

        This method:
        1. Converts RA/Dec to Az/El for the given time
        2. Filters stars within the specified Az/El ranges
        3. Sorts by visual magnitude (vmag) - brightest first
        4. Optionally limits to top_k brightest stars

        Args:
            time: datetime object for observation time
            az_range: Tuple of (min_az, max_az) in degrees
            el_range: Tuple of (min_el, max_el) in degrees
            top_k: Maximum number of brightest stars to return (None = all stars)

        Returns:
            Table: Filtered catalog with Az and El columns
        """
        # Get catalog with Az/El coordinates for this time
        catalog_with_azel = self.get_azel_for_time(time)

        # Filter by Az/El ranges
        az_min, az_max = az_range
        el_min, el_max = el_range
        
        if az_min < 0:
            az_mask = (
                (catalog_with_azel["Az"] > 360 + az_min) |
                (catalog_with_azel["Az"] < az_max)
            )
        else:
            az_mask = (
                (catalog_with_azel["Az"] > az_min) &
                (catalog_with_azel["Az"] < az_max)
            )
            

        filtered_catalog = catalog_with_azel[
            (catalog_with_azel["El"] > el_min) &
            (catalog_with_azel["El"] < el_max) &
            az_mask
        ]

        # Limit to top_k stars if specified
        if top_k is not None and len(filtered_catalog) > top_k:
            filtered_catalog = filtered_catalog[0:top_k]

        return filtered_catalog
    
    def get_azel_for_time(self, time: datetime) -> Table:
        """
        Requires a catalog to have already been loaded. Calculates the Azimuth and Elevation based on the time and returns a catalog with the newly computed Az and El columns.

        Args:
            time: datetime object for the observation time

        Returns:
            Table: Copy of catalog with Az and El columns added (in degrees)
        """
        if self.catalog is None:
            raise ValueError("Catalog not loaded. Call load_catalog() first.")

        # Create Time object
        observing_time = Time(time, scale='utc')

        # Create AltAz frame for this specific location and time
        altaz_frame = AltAz(obstime=observing_time, location=self.observing_location)

        # Transform celestial coordinates to AltAz
        altaz_coords = self.ra_dec_coords.transform_to(altaz_frame)

        # Extract azimuth and altitude (elevation) in degrees
        az = altaz_coords.az.to(u.deg).value
        el = altaz_coords.alt.to(u.deg).value

        # Create a copy of the catalog and add Az/El columns
        catalog_with_azel = self.full_catalog.copy()
        catalog_with_azel['Az'] = az
        catalog_with_azel['El'] = el

        return catalog_with_azel
    
    def update_catalog(self, new_catalog: Table):
        self.catalog = new_catalog

    def get_coords(self, index, delta = 10):
        x = self.catalog[index]["x_transform"]
        y = self.catalog[index]["y_transform"]
        centroid_x = self.catalog[index]["centroid_x"]
        centroid_y = self.catalog[index]["centroid_y"]
        return x - delta, y - delta, x + delta, y + delta, centroid_x, centroid_y
        