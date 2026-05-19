# midas-zipper

Standalone zarr-zip generation for MIDAS FF/PF workflows.

`pip install midas-zipper` works with **no MIDAS source tree and no compiled C
binaries** — it depends only on `numpy`, `h5py`, `zarr`, `numcodecs`, `numba`,
and `pillow`, plus the system `zip`/`bzip2` tools.

It packages the zarr ingest that the FF and PF pipelines need to turn raw
detector data (HDF5/GE/TIFF) into the `*.MIDAS.zip` files the rest of MIDAS
consumes — previously the source-tree script `utils/ffGenerateZipRefactor.py`,
which broke `pip`-only installs because it lived outside any package.

## CLI

```bash
# FF zarr-zip generation (formerly ffGenerateZipRefactor.py)
midas-ff-zip -resultFolder <dir> -paramFN <params.txt> -LayerNr <n>

# Update a key inside an existing Zarr.zip
midas-update-zarr -fn <file.MIDAS.zip> -keyToUpdate <key> -updatedValue <...>
```

Both are also runnable as modules:

```bash
python -m midas_zipper.ff_zip   -resultFolder ... -paramFN ... -LayerNr ...
python -m midas_zipper.update_zarr -fn ... -keyToUpdate ... -updatedValue ...
```

## Python API

```python
from midas_zipper import generate_ff_zip

generate_ff_zip(
    result_folder="/path/to/scan",
    param_file="/path/to/Parameters.txt",
    layer_nr=1,
)
```

## Scope

This package intentionally **excludes** `AutoCalibrateZarr.py`: that tool drives
the C/OpenMP calibration binaries (`CalibrantPanelShiftsOMP`,
`CalibrantIntegratorOMP`, `GetHKLList`) and so cannot be pip-portable. It stays
with the calibration workflow.
