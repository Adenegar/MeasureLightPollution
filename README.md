# Monitoring Light Pollution

This project provides tools to model camera distortion, model camera light sensitivity, and measure background brightness in the cloud camera fits files. 

## Setup

Install dependencies:

```zsh
uv pip install -e .
```

If you get an error for the incorrect version of Python:

```zsh
uv python install 3.13
uv venv --python 3.13
```
then install

**Data:** Data sources are documented on Sharepoint

Put calibration data in:

```zsh
data/CloudCam{direction}/{date}
```

Put starlists in 

```zsh
data/StarLists
```




## Pipelines

The three scripts in `src/pipelines/` are run in order for each camera direction and date.

### Step 1 — Calibrate the fisheye model

Fits the distortion correction model (Az/El ↔ pixel coordinates) from calibration points.

```zsh
python src/pipelines/calibrate.py --direction West --start-date 20260118
```

| Flag | Description |
|---|---|
| `--direction` | `North`, `East`, `South`, `West`, or `all` |
| `--point-source` | Calibration points to use: `Manual`, `Centroid`, or `Both` (default: `Both`) |
| `--start-date` | Date of calibration points in `YYYYMMDD` format (default: all dates) |
| `--model-type` | `simple` or `full` (default: `full`) |

### Step 2 — Extract centroids and fit brightness calibration

Validates the model against images for a specified night, centroids stars, and fits the brightness calibration used in Step 3.

```zsh
python src/pipelines/extract_centroids.py --direction West --date 20260118
```

| Flag | Description |
|---|---|
| `--direction` | `North`, `East`, `South`, `West`, or `all` |
| `--date` | Date of images in `YYYYMMDD` format |
| `--update-cam-cal` | Fit and save the brightness calibration (required before Step 3) |
| `--save-centroids` | Save centroid positions to `data/Calibration/{date}/` |
| `--save-plot` | Save a brightness regression plot to `out/evals/figs/` |
| `--data-source` | `local` or `api` (default: `local`) |

### Step 3 — Measure sky background brightness

For each nighttime image: centroids stars, fits a per-image photometric zero point, and measures median pixel brightness in a 25×25 pixel grid across the sky.

```zsh
python src/pipelines/measure_background.py --direction West --date 20260118
```

| Flag | Description |
|---|---|
| `--direction` | `North`, `East`, `South`, or `West` |
| `--date` | Date of images in `YYYYMMDD` format |
| `--median-filter` | Apply a median filter to the sky region before measuring |
| `--data` | `local` or `api` (default: `local`) |

## Outputs

`measure_background.py` writes two CSVs to `out/background/`:

| File | Contents |
|---|---|
| `{direction}_{date}_cells.csv` | One row per grid cell per image: `source`, `az`, `el`, `vmag_per_pixel` |
| `{direction}_{date}_images.csv` | One row per image: `source`, `zero_point`, `centroid_rate` |
