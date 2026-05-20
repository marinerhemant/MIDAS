# midas-nf-preprocess notebooks

Runnable Jupyter notebooks for the NF-HEDM preprocessing package.

All notebooks run on **CPU** with **synthetic** data — no external
datasets, no GPU. Source of truth is `_build.py` (cells as `(kind, source)`
tuples); the `.ipynb` files are derived artefacts.

## Rebuild

```bash
source /Users/hsharma/miniconda3/bin/activate midas_env
cd packages/midas_nf_preprocess/notebooks
python _build.py
jupyter nbconvert --to notebook --execute --inplace *.ipynb
```

## Index

| Notebook | What it covers | Status |
| --- | --- | --- |
| `01_image_processing_walkthrough.ipynb` | Synthetic TIFF stack → temporal/spatial median → LoG peak enhancement → connected-component peak detection → `SpotsBitMask` → `SpotsInfo.bin`; plus a differentiability check on `filtered` / `spot_prob`. | runs clean |

`SpotsInfo.bin` is the hand-off to `midas-nf-fitorientation`
(see that package's `notebooks/01_calibration_deep_dive.ipynb`).
