## Setup development mode for module resolution

```zsh
uv pip install -e .
```

## Move Cloud Cam Images
Rsync files from vm-internship2 to hqdevmacstudio01. Only sync files in the 07:00-14:20 UTC range.

```zsh
rsync -av \
    --include='CloudCam*UTC07*.fits' \
    --include='CloudCam*UTC08*.fits' \
    --include='CloudCam*UTC09*.fits' \
    --include='CloudCam*UTC10*.fits' \
    --include='CloudCam*UTC11*.fits' \
    --include='CloudCam*UTC12*.fits' \
    --include='CloudCam*UTC13*.fits' \
    --include='CloudCam*UTC140*.fits' \
    --include='CloudCam*UTC141*.fits' \
    --include='CloudCam*UTC1420*.fits' \
    --exclude='*' \
    adenegar@vm-internship2:/skycams2/CloudCamWest/20260201/ \
    /Users/adenegar/Projects/starMap/data/CloudCamWest/20260201/
```

Catalog data location:
vm-internship2:/data/skwok/catalogs

skycam data
vm-internship2:/skycams1
vm-internship2:/skycams2

TODO: Instructions for how to use

```zsh
python src/pipelines/calibrate.py --direction West --calibrate-date 20260118 --data-source local            
```

```zsh
python src/pipelines/images_to_brightness.py --direction West --date 20260118 --data-source local 
```


calibration and images to brightness:

python calibrate.py --direction East --calibrate-date 20260109 --data-source local 

python images_to_brightness.py --direction East --date 20260109 --data-source local

python -m src.pipelines.calibrate --direction East --calibrate-date 20260109 --data-source local 

TODO:
- [ ] fix pixel - vmag label
- Measurement, not residual

TODO: 
- [ ] Save data for each image
- [ ] Save data for each star