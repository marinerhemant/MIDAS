# Tomography Reconstruction with MIDAS

**Version:** 9.0  
**Contact:** hsharma@anl.gov

---

## 1. Introduction

MIDAS includes a high-performance tomographic reconstruction module based on the **gridrec** algorithm — a Fourier-based filtered back-projection method that uses a prolate spheroidal wave function (PSWF) interpolation kernel for regridding in the frequency domain. This approach produces reconstructions of comparable quality to standard FBP while being significantly faster, thanks to FFTW-accelerated FFTs and OpenMP parallelism.

The tomography module is designed for absorption-contrast X-ray computed tomography (CT) data collected at APS, but it works with any parallel-beam projection dataset. It handles:

- **Dark-field and white-field normalization** of raw projection images.
- **Multiple reconstruction filters** (Shepp-Logan, Hann, Hamming, Ramp).
- **Rotation-axis shift search** across a range of candidate center positions.
- **Ring artifact removal** via sinogram-space filtering.
- **OpenMP parallelism** across slices for fast multi-core reconstruction.

---

## 2. Architecture

The tomography pipeline has three components:

```mermaid
flowchart LR
    subgraph "Python Front-Ends"
        A["process_hdf.py<br/>(HDF5 input)"]
        B["midas_tomo_python.py<br/>(NumPy arrays)"]
    end

    subgraph "C Engine"
        C["MIDAS_TOMO<br/>(gridrec + FFTW + OMP)"]
    end

    A --> |"param file +<br/>raw binary"| C
    B --> |"param file +<br/>raw binary"| C
    C --> D["Reconstructed<br/>slices (.bin)"]

    style A fill:#16213e,stroke:#0f3460,color:#fff
    style B fill:#16213e,stroke:#0f3460,color:#fff
    style C fill:#0f3460,stroke:#e94560,color:#fff
    style D fill:#2b2d42,stroke:#8d99ae,color:#fff
```

| Component | Location | Purpose |
|-----------|----------|---------|
| `MIDAS_TOMO` | `~/opt/MIDAS/TOMO/bin/MIDAS_TOMO` | C binary — performs the actual reconstruction |
| `process_hdf.py` | `~/opt/MIDAS/TOMO/process_hdf.py` | Reads HDF5 data, generates parameter file, calls `MIDAS_TOMO` |
| `midas_tomo_python.py` | `~/opt/MIDAS/TOMO/midas_tomo_python.py` | Python library — accepts NumPy arrays and returns reconstructions |

---

## 3. Quick Start

### 3.1. From HDF5 Files (Recommended)

If your tomography data is in an HDF5 file with the standard APS data exchange layout (`/exchange/data`, `/exchange/dark`, `/exchange/bright`), use `process_hdf.py`:

```bash
python ~/opt/MIDAS/TOMO/process_hdf.py \
  -dataFN /path/to/tomo_scan.h5 \
  -nCPUs 20
```

The script will:
1. Read dark, bright (white-field), and projection frames from the HDF5 file.
2. Apply any cropping specified in `/analysis/process/analysis_parameters/`.
3. Apply an optional in-plane rotation correction if `RotationAngle` is present.
4. Write a raw binary file and a parameter file (`mt_par.txt`).
5. Call `MIDAS_TOMO` to perform the reconstruction.

> [!NOTE]
> The HDF5 file must contain the following datasets:
> - `/exchange/data` — projection images (`uint16`, shape: `[nFrames, nZ, nX]`)
> - `/exchange/dark` — dark-field image(s)
> - `/exchange/bright` — white-field (flat-field) images (at least 2)
> - `/measurement/process/scan_parameters/start` — starting omega angle
> - `/measurement/process/scan_parameters/step` — omega step size
> - `/analysis/process/analysis_parameters/CropXL`, `CropXR`, `CropZL`, `CropZR` — cropping bounds
> - `/analysis/process/analysis_parameters/shift` — rotation axis shift (pixels)

### 3.2. Direct HDF5 Reconstruction (New)

You can also run `MIDAS_TOMO` directly on an HDF5 file without converting to binary first. This is efficient and avoids duplicating data.

**Parameter File for HDF5:**
```text
HDF5FileName /path/to/data.h5
ImageDatasetName /exchange/data
DarkDatasetName /exchange/dark
WhiteDatasetName /exchange/bright
reconFileName /path/to/output_recon
detXdim 2048
detYdim 1024
thetaRange -180 180 0.25
shiftValues 0 0 1
```

**Running:**
```bash
~/opt/MIDAS/TOMO/bin/MIDAS_TOMO my_hdf5_params.txt 20
```

### 3.3. From NumPy Arrays (Python API)

For programmatic use or when your data is not in HDF5 format:

```python
from midas_tomo_python import run_tomo
import numpy as np

# Load your data (example shapes)
data   = ...  # np.ndarray, shape (nThetas, nSlices, xDim), uint16
dark   = ...  # np.ndarray, shape (nSlices, xDim), float32
whites = ...  # np.ndarray, shape (2, nSlices, xDim), float32
thetas = np.arange(-180, 180.1, 0.25)  # rotation angles in degrees

recon = run_tomo(
    data, dark, whites,
    workingdir='/scratch/my_recon/',
    thetas=thetas,
    shifts=1.0,           # Single shift value (pixels)
    filterNr=2,           # Hann filter
    doLog=1,              # Take log for absorption contrast
    numCPUs=20
)
# recon shape: (nShifts, nSlices, xDimNew, xDimNew)
```

> [!IMPORTANT]
> The `data` array passed to `run_tomo` should have shape `(nThetas + 2, nSlices, xDim)` where the first two frames are tilt-corrected projections that get treated as extra frames. Internally, the function subtracts 2 from `nThetas`. See the docstring in `midas_tomo_python.py` for the exact binary layout.

---

## 4. Parameter File Reference

The `MIDAS_TOMO` binary reads a plain-text parameter file. Each line contains a keyword followed by its value(s).

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `dataFileName` | string | Path to the raw binary input file (Mutual exclusive with HDF5FileName) | *required* |
| `HDF5FileName` | string | Path to the HDF5 input file (Mutual exclusive with dataFileName) | — |
| `ImageDatasetName` | string | HDF5 path to projection data (e.g. `/exchange/data`) | *required if HDF5* |
| `DarkDatasetName` | string | HDF5 path to dark fields (e.g. `/exchange/dark`) | *required if HDF5* |
| `WhiteDatasetName` | string | HDF5 path to white fields (e.g. `/exchange/bright`) | *required if HDF5* |
| `reconFileName` | string | Base name for reconstruction output | *required* |
| `areSinos` | 0 or 1 | Set to 1 if input is pre-computed sinograms, 0 if raw projections | 0 |
| `detXdim` | int | Horizontal dimension of detector (pixels) | *required* |
| `detYdim` | int | Vertical dimension of detector (number of slices) | *required* |
| `thetaFileName` | string | Path to text file with one angle (°) per line | — |
| `thetaRange` | 3 floats | `startAngle endAngle angleInterval` (alternative to `thetaFileName`) | — |
| `filter` | int | Reconstruction filter (see table below) | 2 (Hann) |
| `shiftValues` | 3 floats | `start_shift end_shift shift_interval` — rotation axis shift search (pixels) | *required* |
| `ringRemovalCoefficient` | float | Ring-artifact removal strength (0 = off, 1.0 = typical) | 0 |
| `doLog` | 0 or 1 | Take the logarithm for absorption-contrast reconstruction | 1 |
| `slicesToProcess` | string | `-1` for all slices, or path to a file listing slice indices | -1 |
| `ExtraPad` | 0 or 1 | Extra zero-padding for better frequency resolution | 0 |
| `AutoCentering` | 0 or 1 | Shift reconstruction so the rotation axis is at the image center | 1 |
| `saveReconSeparate` | 0 or 1 | Save each slice in a separate file (1) or all in one file (0) | 0 |

### 4.1. Reconstruction Filters

| Filter ID | Name | Characteristics |
|-----------|------|-----------------|
| 0 | None | No filtering (not recommended) |
| 1 | Shepp-Logan | Moderate smoothing, good edge preservation |
| 2 | **Hann** | Good balance of noise suppression and resolution (default) |
| 3 | Hamming | Similar to Hann but slightly less smoothing |
| 4 | Ramp | Maximum resolution but maximum noise amplification |

> [!TIP]
> Start with the **Hann** filter (2). If features appear blurred, try Shepp-Logan (1) or Ramp (4). If the reconstruction is too noisy, try Hamming (3).

---

---

## 5. Input Data Format

### 5.1. Raw Binary Layout

When `areSinos = 0`, the input file must be a single binary file with the following layout:

| Segment | Data Type | Shape | Description |
|---------|-----------|-------|-------------|
| Dark frame | `float32` | `(detYdim, detXdim)` | Dark-current image |
| White frames | `float32` | `(2, detYdim, detXdim)` | Flat-field images (before and after) |
| Projections | `uint16` | `(nThetas, detYdim, detXdim)` | Raw projection images |

### 5.2. HDF5 Input

When using HDF5 input (`HDF5FileName` specified), the module reads datasets directly. 
- **Type Safety**: The module automatically checks if the datasets are numeric (Integer or Float).
- **Type Casting**: 
    - Darks and Whites are cast to `float32` (native HDF5 conversion).
    - Projections are cast to `uint16`.
- **Dimensions**:
    - Darks: Can be 2D `(Y, X)` or 3D `(1, Y, X)`.
    - Whites: Can be `(N, Y, X)` (reads first 2 frames) or 2D `(Y, X)` (duplicates for start/end).
    - Projections: `(nThetas, Y, X)`. 

### 5.3. Sinogram Input

When `areSinos = 1`, the input file contains pre-computed sinograms as `float32` data. Each sinogram has shape `(nThetas, detXdim)`.

---

## 6. Output Format

The reconstruction is saved as a single binary file (or individual files per slice if `saveReconSeparate = 1`). The filename convention is:

```
{reconFileName}_NrShifts_{NNN}_NrSlices_{NNNNN}_XDim_{NNNNNN}_YDim_{NNNNNN}_float32.bin
```

The data is `float32` with shape `(nShifts, nSlices, xDimNew, xDimNew)`, where `xDimNew` is the next power of 2 ≥ `detXdim` (or the next-next power of 2 if `ExtraPad = 1`).

### 6.1. Viewing the Reconstruction

```python
import numpy as np
import matplotlib.pyplot as plt

# Parse dimensions from filename
recon = np.fromfile('recon_output_NrShifts_001_NrSlices_01024_XDim_002048_YDim_002048_float32.bin',
                    dtype=np.float32).reshape((1, 1024, 2048, 2048))

plt.imshow(recon[0, 512, :, :], cmap='gray')
plt.colorbar()
plt.title('Reconstructed slice 512')
plt.show()
```

---

## 7. Rotation Axis Alignment

The most critical parameter for a good reconstruction is the **rotation axis position** (the `shiftValues` parameter). If the rotation axis is not centered on the detector, the reconstruction will exhibit characteristic arc-shaped artifacts.

### 7.1. Finding the Rotation Center

**Method 1 — Single value (if known):**
```
shiftValues 1.5 1.5 1
```

**Method 2 — Search a range:**
```
shiftValues -3.0 3.0 0.5
```
This produces 13 reconstructions, one for each shift. Inspect them to find the shift that gives the sharpest reconstruction.

> [!WARNING]
> The `shiftValues` range must produce an **even** number of shifts when more than one shift is specified. The number of slices must also be even. The C engine processes slices in pairs for efficiency.

### 7.2. Auto-Centering

When `AutoCentering = 1` (default), the reconstruction is shifted so the rotation axis appears at the center of the output image. Set to `0` if you want the rotation axis at its natural detector position.

---

## 8. Performance

### 8.1. Memory

The C engine reads `/proc/meminfo` at startup to determine available RAM and automatically limits the number of OpenMP threads to avoid out-of-memory conditions. Each thread requires memory proportional to the padded sinogram size for FFT buffers.

### 8.2. FFTW Wisdom Files

On the first run with a given detector size, the code generates FFTW wisdom files (`fftwf_wisdom_1d_*.txt` and `fftwf_wisdom_2d_*.txt`) in the working directory. These files cache optimized FFT plans and significantly speed up subsequent runs. **Do not delete them** unless you change detector dimensions.

### 8.3. Parallelism

The reconstruction is parallelized across slices using OpenMP. The number of threads is specified as the second argument to `MIDAS_TOMO`:

```bash
~/opt/MIDAS/TOMO/bin/MIDAS_TOMO params.txt 20
```

Typical performance: a 2048 × 2048 × 1800 dataset reconstructs in under 2 minutes on a 40-core workstation.

---

## 9. Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Arc-shaped artifacts | Incorrect rotation center | Search a range of `shiftValues` |
| Ring artifacts | Detector pixel defects or gain non-uniformity | Increase `ringRemovalCoefficient` (try 1.0) |
| Very noisy reconstruction | Using Ramp filter with noisy data | Switch to Hann (2) or Hamming (3) filter |
| Reconstruction all black or NaN | Normalization failure (bad dark/white) | Verify dark and white frames are valid |
| `Number of shifts must be even` error | Odd number of shift steps | Adjust `shiftValues` range to produce an even number |
| `Number of slices must be even` error | Odd slice count | Crop one row from your input or specify even slice range in `slicesToProcess` |
| Segmentation fault | Insufficient RAM for requested thread count | Reduce `nCPUs`; the code auto-limits but edge cases exist |

---

## 10. See Also

- [README.md](README.md) — High-level MIDAS overview and manual index
- [FF_RadialIntegration.md](FF_RadialIntegration.md) — Radial integration / caking (complementary analysis for diffraction data)
- [FF_calibration.md](FF_calibration.md) — FF-HEDM geometry calibration

---

If you encounter any issues or have questions, please open an issue on this repository.
