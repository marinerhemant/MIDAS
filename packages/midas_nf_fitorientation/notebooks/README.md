# midas-nf-fitorientation notebooks

Runnable Jupyter notebooks for the NF-HEDM orientation / calibration
fitter.

All notebooks run on **CPU** with the **bundled Au example**
(`NF_HEDM/Example/sim/`) — no external datasets, no GPU, no C build.
Source of truth is `_build.py`; the `.ipynb` files are derived artefacts.
Set `$MIDAS_HOME` if the repo is not at `~/opt/MIDAS`.

## Rebuild

```bash
source /Users/hsharma/miniconda3/bin/activate midas_env
cd packages/midas_nf_fitorientation/notebooks
python _build.py
jupyter nbconvert --to notebook --execute --inplace *.ipynb
```

## Index

| Notebook | What it covers | Status |
| --- | --- | --- |
| `01_calibration_deep_dive.ipynb` | Loads the bundled `SpotsInfo.bin` / `OrientMat.bin` / `grid.txt` / `hkls.csv`, builds the differentiable forward model, screens one voxel for candidate orientations, and refines orientation with L-BFGS over the Gaussian-splat soft overlap — the exact kernel the calibration drivers wrap. | runs clean |

### Driver note (current build)

The packaged single/cluster calibration drivers `fit_parameters_run` and
`fit_multipoint_run` are documented but **not executed end-to-end** in
notebook 01 on the bundled Au example:

* `fit_multipoint_run` requires a `GridPoints` block in the param file,
  which the bundled `test_ps_au.txt` does not contain.
* The joint-calibration phase takes a code path that needs the **dense**
  `ObsVolume` (`packed=False`), while the driver currently constructs the
  packed-bit volume → `TypeError: soft_fraction needs a dense
  floating-point ObsVolume`.

The screen + soft-overlap L-BFGS kernel (the heart of all three drivers)
runs cleanly and is demonstrated directly in the notebook.
