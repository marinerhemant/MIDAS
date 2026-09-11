# midas-dfxm quickstart — Windows, macOS, Linux

For the NX school DFXM session, and anyone starting out. `midas-dfxm` is pure Python on
PyPI; the one large compiled dependency is PyTorch. No GPU, no CUDA, no admin rights needed.

**Tested 2026-09-10** in fresh virtual environments with `pip install "midas-dfxm[viz]"` from
PyPI: macOS arm64 with Python 3.12 (4 min 41 s including Jupyter; environment 1.4 GB) and
Linux x86_64 with Python 3.9 (70 s; 7.2 GB — see the CPU-only step below). **Windows was not
part of that test.**

## 1. Python

Python 3.9–3.12 (python.org or Miniconda), in an isolated environment.

Windows:
```bat
py -m venv dfxm_env
dfxm_env\Scripts\activate
python -m pip install --upgrade pip
```

macOS / Linux:
```bash
python3 -m venv dfxm_env
source dfxm_env/bin/activate
python -m pip install --upgrade pip
```

(conda: `conda create -n dfxm python=3.11 && conda activate dfxm`)

## 2. Install

**Linux without an NVIDIA GPU: do this first**, or pip downloads the CUDA build of PyTorch
(that is what made the tested Linux environment 7.2 GB):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then, on every system:
```bash
pip install "midas-dfxm[viz]" jupyterlab
```

This pulls PyTorch, NumPy, SciPy, h5py, scikit-image, numba, zarr, tifffile, matplotlib and 16
MIDAS packages. One of them, `midas-index`, is published as source and built during the
install; that worked on the tested macOS and Linux machines. If it fails on Windows, keep the
error message and ask a tutor.

## 3. One environment variable

PyTorch and the MIDAS packages can each load an OpenMP runtime, and on some machines that
aborts at import. Set this once:

```bat
set KMP_DUPLICATE_LIB_OK=TRUE        :: Windows, this shell
setx KMP_DUPLICATE_LIB_OK TRUE       :: Windows, future shells
```
```bash
export KMP_DUPLICATE_LIB_OK=TRUE     # macOS / Linux
```

## 4. Check the install

```bash
python -c "import midas_dfxm as dx; print('midas-dfxm', dx.__version__); from midas_dfxm import load_6idc_scan, reduce_rocking; print('OK')"
```

## 5. Get the tutorials

```bash
python -m midas_dfxm.examples.get_notebooks
jupyter lab midas_dfxm_tutorials
```

The first command copies the tutorials into `./midas_dfxm_tutorials`; it never overwrites a
copy you have edited unless you add `--force`.

- **`rocking_curve_reconstruction_tutorial.ipynb` — the school notebook.** Part A simulates
  and reconstructs rocking scans; nothing to download. Part B reduces a scan measured at APS
  6-ID-C: load it, check that every frame is paired with its own angle, subtract the detector
  pedestal, make a tilt or strain map with an error bar, then look at the raw curves to see
  whether that map can be read as tilt at all, and reduce the whole curve when it cannot. With
  no scan set, Part B runs on a synthetic scan built to show that failure, so the lesson works
  before any download.
- **`reduce_6idc_scan.ipynb`** — Part B on its own, for a scan you measured (theta rock or
  theta-2theta, 2021 or 2025 layout), with a cell to pick the ROI from the frames.
- **`tutorial_school_dfxm.py`** — the differentiable forward model and the full-F inverse on
  synthetic data. It opens as an interactive notebook in VS Code (Run Cell above each `# %%`),
  or runs with `python tutorial_school_dfxm.py`.

## 6. Practice data for Part B

Public NaMnO2 scans from 6-ID-C (Plumb *et al.*, Mater. Charact. 204, 113174 (2023)), on Dryad
at doi:10.25349/D9S03T. In a browser, download `Plumb_DFXM_Dec2021_data_S006.zip` (~220 MB) and
`Plumb_DFXM_Dec2021_motors.zip`, and unzip both. Use the motor table from the **December 2021**
archive: the July 2021 logs carry the same file names and the wrong angles for these frames.

## 7. Run the tests (optional, from a clone of the repository)

```bash
pip install pytest
python -m pytest tests -q
```
