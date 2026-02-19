import numpy as np
from astropy.table import Table
from astropy.io.fits import HDUList


CAMERA_STDEV = np.float64(2.55)

def centroid_starlist(
    hdul: HDUList,
    starlist: Table,
    measure_brightness: bool = False,
    exclude_oversaturated: bool = True,
    subtract_background: bool = False,
    filter_background: bool = True,
):
    """
    Centroid stars in an image and optionally measure their brightness.

    Args:
        hdul: FITS HDU list
        starlist: Table with x_transform and y_transform columns
        measure_brightness: If True, brightness will be measured at the centroided positions.
        exclude_oversaturated: If True, stars with any pixel value of 255 within the
                               aperture are treated as non-centroided.
        subtract_background: If True, subtract the local median before centroiding.
        filter_background: If True, exclude annulus pixels >= 60 from the background median.

    Returns:
        If measure_brightness is False: (centroid_x, centroid_y, sigma_x, sigma_y)
        If measure_brightness is True: (centroid_x, centroid_y, sigma_x, sigma_y, brightness)
    """
    img_gray = _filter_noise(hdul)
    image_width = img_gray.shape[1]
    image_height = img_gray.shape[0]
    centroid_x = []
    centroid_y = []
    sigma_x = []
    sigma_y = []

    for star in starlist:
        x, y = int(star["x_transform"]), int(star["y_transform"])
        search_box = 20
        delta = search_box // 2
        x1, x2, y1, y2 = x - delta, x + delta, y - delta, y + delta

        if x1 < 0 or x2 > image_width or y1 < 0 or y2 > image_height:
            centroid_x.append(None)
            centroid_y.append(None)
            sigma_x.append(None)
            sigma_y.append(None)

            continue

        # max_sig_x = max_sig_y = search_box / 5
        max_sig_x = max_sig_y = 4.5

        cx, cy, sx, sy = centroid(img_gray, x1, y1, x2, y2, max_sigma_x=max_sig_x, max_sigma_y=max_sig_y, subtract_background=subtract_background)
        centroid_x.append(cx)
        centroid_y.append(cy)
        sigma_x.append(sx)
        sigma_y.append(sy)

    # Exclude oversaturated stars
    if exclude_oversaturated:
        inner_radius = _aperture_inner_radius()
        outer_radius = 25.0
        thickness = 3.0
        grid_size = int(np.ceil(outer_radius * 2)) + 4
        center = grid_size / 2
        y_g, x_g = np.ogrid[:grid_size, :grid_size]
        distances = np.sqrt((x_g - center) ** 2 + (y_g - center) ** 2)
        check_mask = (distances <= inner_radius) | ((distances >= (outer_radius - thickness)) & (distances <= outer_radius))
        half_grid = grid_size // 2

        for i in range(len(centroid_x)):
            if centroid_x[i] is None:
                continue
            x_int = int(round(centroid_x[i]))
            y_int = int(round(centroid_y[i]))
            x_min = x_int - half_grid
            x_max = x_int - half_grid + grid_size
            y_min = y_int - half_grid
            y_max = y_int - half_grid + grid_size
            if x_min < 0 or x_max >= image_width or y_min < 0 or y_max >= image_height:
                continue
            region = img_gray[y_min:y_max, x_min:x_max]
            if region.shape == (grid_size, grid_size) and np.any(region[check_mask] == 255):
                centroid_x[i] = None
                centroid_y[i] = None
                sigma_x[i] = None
                sigma_y[i] = None

    if not measure_brightness:
        return centroid_x, centroid_y, sigma_x, sigma_y

    # Measure brightness at centroided positions
    brightness = measure_brightness_at_centroids(
        img_gray, centroid_x, centroid_y, filter_background=filter_background
    )
    return centroid_x, centroid_y, sigma_x, sigma_y, brightness


def centroid(img_gray: np.ndarray, x1, y1, x2, y2, max_sigma_x = 3, max_sigma_y = 3, verbose = False, subtract_background = False):
    """Return tuple of (centroid_x, centroid_y, sigma_x, sigma_y), or (None, None, None, None) if no clear centroid is found"""
    sub_img_gray = img_gray[y1:y2, x1:x2]
    if subtract_background:
        data = np.clip(sub_img_gray - np.median(sub_img_gray), 0, None)
    else:
        data = sub_img_gray

    # Intensity-weighted centroid
    y_grid, x_grid = np.indices(data.shape)
    total = data.sum()

    if total == 0:
        return (None, None, None, None)

    x0 = (x_grid * data).sum() / total
    y0 = (y_grid * data).sum() / total

    # Calculate standard deviation using parallel axis theorem: Var(X) = E[X^2] - E[X]^2
    sigma_x = np.sqrt((x_grid**2 * data).sum() / total - x0**2)
    sigma_y = np.sqrt((y_grid**2 * data).sum() / total - y0**2)

    if not (0 <= x0 <= data.shape[1] and 0 <= y0 <= data.shape[0]):
        if verbose:
            print(f"out of bounds: {x0, y0}, for shape {data.shape[1], data.shape[0]}")
        return (None, None, sigma_x, sigma_y)

    if sigma_x >= max_sigma_x or sigma_y >= max_sigma_y:
        if verbose:
            print(f"std too high: sigma_x, sigma_y = ({sigma_x:.2f}, {sigma_y:.2f}), thresh_x, thresh_y = {max_sigma_x, max_sigma_y}")
        return (None, None, sigma_x, sigma_y)

    # Refine centroid with 16x16 box
    search_box = 16
    delta = search_box // 2
    rx1 = int(x1 + x0 - delta)
    rx2 = int(x1 + x0 + delta)
    ry1 = int(y1 + y0 - delta)
    ry2 = int(y1 + y0 + delta)

    # Clip to image bounds
    rx1 = max(0, rx1)
    rx2 = min(img_gray.shape[1], rx2)
    ry1 = max(0, ry1)
    ry2 = min(img_gray.shape[0], ry2)

    refined_img = img_gray[ry1:ry2, rx1:rx2]
    if subtract_background:
        refined_data = np.clip(refined_img - np.median(refined_img), 0, None)
    else:
        refined_data = refined_img

    ry_grid, rx_grid = np.indices(refined_data.shape)
    refined_total = refined_data.sum()

    if refined_total == 0:
        if verbose:
            print("refined total is zero")
        return (None, None, sigma_x, sigma_y)

    x0_refined = (rx_grid * refined_data).sum() / refined_total
    y0_refined = (ry_grid * refined_data).sum() / refined_total

    if not (0 <= x0_refined <= refined_data.shape[1] and 0 <= y0_refined <= refined_data.shape[0]):
        if verbose:
            print(f"refined out of bounds: {x0_refined, y0_refined}, for shape {refined_data.shape[1], refined_data.shape[0]}")
        return (None, None, sigma_x, sigma_y)

    # Calculate refined standard deviations using parallel axis theorem
    sigma_x_refined = np.sqrt((rx_grid**2 * refined_data).sum() / refined_total - x0_refined**2)
    sigma_y_refined = np.sqrt((ry_grid**2 * refined_data).sum() / refined_total - y0_refined**2)

    # Return coordinates and standard deviations in full image frame
    return (rx1 + x0_refined, ry1 + y0_refined, sigma_x_refined, sigma_y_refined)

def _get_gray_image(hdul: HDUList) -> np.ndarray:
    """
    Extract grayscale image from FITS HDU list.

    Uses the median channel (hdul[1]) if available, otherwise computes
    median from RGB data in hdul[0].

    Args:
        hdul: FITS HDU list

    Returns:
        2D grayscale image array
    """
    if len(hdul) > 1:
        # Use pre-computed median channel
        # Need to flip to match orientation of RGB processing
        img_gray = np.flip(hdul[1].data, axis=1)
    else:
        # Compute median from RGB data
        img = np.flip(np.transpose(hdul[0].data, (1, 2, 0)), axis=1)
        img_gray = np.sort(img, axis=2)[..., 1]

    return img_gray

def _aperture_inner_radius():
    """Return the inner aperture radius used for photometry and saturation checks."""
    aperture = 11.0
    return aperture / 2

def _filter_noise(hdul: HDUList):
    """Extract the green channel from the FITS data."""
    return np.flip(hdul[0].data[1], axis=1)


def measure_brightness_at_centroids(
    img_gray: np.ndarray,
    centroid_x: list,
    centroid_y: list,
    filter_background: bool = True,
) -> list:
    """
    Measure brightness at centroided star positions using aperture photometry.

    Args:
        img_gray: 2D grayscale image array
        centroid_x: List of x centroid positions (may contain None)
        centroid_y: List of y centroid positions (may contain None)

    Returns:
        List of background-subtracted brightness values
    """
    aperture = 11.0
    outer = 25.0
    thickness = 3.0

    inner_radius = aperture / 2
    outer_radius = outer / 2

    # Create grid large enough to contain the outer circle
    grid_size = int(np.ceil(outer_radius * 2)) + 4
    center = grid_size / 2

    # Create masks
    y, x = np.ogrid[:grid_size, :grid_size]
    distances = np.sqrt((x - center) ** 2 + (y - center) ** 2)

    inner_mask = distances <= inner_radius
    ring_mask = (distances >= (outer_radius - thickness)) & (distances <= outer_radius)

    image_height, image_width = img_gray.shape
    brightnesses = []

    for x_pos, y_pos in zip(centroid_x, centroid_y):
        if x_pos is None or y_pos is None:
            brightnesses.append(np.nan)
            continue

        x_int = int(round(x_pos))
        y_int = int(round(y_pos))

        half_grid = grid_size // 2
        x_min = x_int - half_grid
        x_max = x_int - half_grid + grid_size
        y_min = y_int - half_grid
        y_max = y_int - half_grid + grid_size

        if x_min < 0 or x_max >= image_width or y_min < 0 or y_max >= image_height:
            brightnesses.append(np.nan)
            continue

        star_region = img_gray[y_min:y_max, x_min:x_max]

        if star_region.shape != (grid_size, grid_size):
            brightnesses.append(np.nan)
            continue

        star_signal = np.sum(star_region[inner_mask])
        if filter_background:
            annulus_pixels = star_region[ring_mask]
            filtered = annulus_pixels[annulus_pixels < 45]
            background_per_pixel = np.median(filtered) if len(filtered) > 0 else 45
        else:
            background_per_pixel = np.median(star_region[ring_mask])
        background_total = background_per_pixel * np.sum(inner_mask)

        brightness = star_signal - background_total
        brightnesses.append(brightness)

    return brightnesses