# midas-zipper

Standalone zarr-zip generation for MIDAS FF/PF workflows.

`pip install midas-zipper` works with **no MIDAS source tree and no compiled C
binaries** — it depends only on `numpy`, `h5py`, `zarr`, `numcodecs`, `numba`,
and `pillow`, plus the system `zip`/`bzip2` tools.

It packages the zarr ingest that the FF and PF pipelines need to turn raw
detector data (HDF5/GE/TIFF) into the `*.MIDAS.zip` files the rest of MIDAS
consumes — previously the source-tree script `utils/ffGenerateZipRefactor.py`,
which broke `pip`-only installs because it lived outside any package.

## Notebooks

Worked-example Jupyter notebooks live in `notebooks/`. They are **not shipped with `pip install`** — get them by cloning the [MIDAS repository](https://github.com/marinerhemant/MIDAS/tree/master/packages/midas_zipper/notebooks).

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

## Lossless archive zips

`-preProcThresh` zeroes pixels, and with a dark file the multi-file readers
(GE binary, TIFF, CBF) also clip negative residuals to 0, so those zips cannot
give back the raw frames. For an archive that can, add `-archiveLossless`:

```bash
midas-ff-zip -resultFolder <dir> -paramFN <params.txt> -LayerNr <n> -archiveLossless auto
```

Each frame is stored as `zigzag(frame - background)`, a bijection on n-bit
words, so the raw frames come back bit for bit. The background is the stored
dark (`dark`), the per-pixel median of evenly spaced frames of the scan
(`median`, the only choice when there is no dark file, as for NF), or whichever
is smaller (`auto`). A few frames are trial-encoded first: if nothing beats the
raw frames by 2 % the zip is left as an ordinary zip, and for cameras that
store 12 bits in 16 the background is rounded onto the data's bit grid (a dark
that is off the grid makes things worse otherwise). Every chunk is decoded and
compared with its source while writing.

An archive zip has no `exchange/data`; it has `exchange/data_resid` and
`exchange/background`, so a reader that does not know the encoding stops with a
`KeyError` instead of reading residuals. To read it:

```python
from midas_zipper.archive import read_frames, restore_zip
frames = read_frames("scan.MIDAS.zip", 0, 10)        # raw frames
restore_zip("scan.MIDAS.zip", "scan.ordinary.zip")   # ordinary zip, for the pipeline
```

or `midas-zip-archive convert|restore`, which works on any existing zip that has
`exchange/data` (and `exchange/dark` if there is a dark). Results from the
compression study behind this are provisional (`dev/compression_study`).

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
