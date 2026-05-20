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
| `01_calibration_deep_dive.ipynb` | Loads the bundled `SpotsInfo.bin` / `OrientMat.bin` / `grid.txt` / `hkls.csv`, builds the differentiable forward model, screens one voxel and refines orientation with L-BFGS over the Gaussian-splat soft overlap, then runs both packaged calibration drivers **end-to-end** on the bundled Au example: `fit_parameters_run` (single-voxel) and `fit_multipoint_run` (cluster). | runs clean |

### Driver note

Both calibration drivers run end-to-end in notebook 01 on the bundled Au
example, CPU only:

* `fit_multipoint_run` derives its voxel set from the reconstructed
  `MicFileText` `.mic` (highest-confidence voxels above `MinConfidence`)
  when the param file carries no explicit `GridPoints` block — so the
  bundled `test_ps_au.txt` needs no hand-written block. An explicit
  `GridPoints` block still wins when present.
* Both drivers build the **dense** floating-point `ObsVolume`
  (`packed=False`) the Gaussian-splat soft-overlap path requires.
