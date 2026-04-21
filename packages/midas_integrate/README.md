# midas-integrate

GPU-accelerated radial integration for synchrotron area detectors.

Part of the [MIDAS](https://github.com/marinerhemant/MIDAS) toolkit. Backs:
- **Paper B** — *MIDAS Radial Integration: Accelerating X-ray Diffraction Data Reduction for CPU and GPU* (J. Synchrotron Rad., pending).
- **Paper C** — *Cardinal-Angle Aliasing in Azimuthal Integration* (J. Appl. Cryst., pending).

Companion to [midas-auto-calibrate](https://pypi.org/project/midas-auto-calibrate/), which supplies the refined detector geometry that drives the integrator.

## Install

```bash
pip install midas-integrate                 # CPU + midas-auto-calibrate
pip install 'midas-integrate[gpu]'          # +GPU wheel (Linux x86_64, CUDA 12)
pip install 'midas-integrate[paper]'        # +matplotlib/pyFAI for Paper B/C regen
```

Binary wheels for **Linux x86_64 (manylinux_2_28)** and **macOS (x86_64 + arm64)**. The `MIDASIntegrator` and `MIDASDetectorMapper` C binaries ship inside the wheel — no compiler needed on the install machine.

**Windows**: use WSL2 + the Linux wheel. Native Windows wheels are not planned.

## Quick start — integrate a detector image

```python
import shutil, tempfile
from pathlib import Path
import midas_auto_calibrate as mac
import midas_integrate as mi

# Stage the wheel-bundled CeO₂ frame.
workdir = Path(tempfile.mkdtemp(prefix='mi_'))
shutil.copy(mac.data.CEO2_PILATUS, workdir / 'CeO2_00001.tif')

cfg = mi.IntegrationConfig(
    lsd=657_436.9, ybc=685.5, zbc=921.0,
    wavelength=0.172973, pixel_size=172.0,
    nr_pixels_y=1475, nr_pixels_z=1679,
    r_min=50, r_max=1200, r_bin_size=0.5,
    eta_min=-180, eta_max=180, eta_bin_size=1.0,
)

# 1. Pre-compute the pixel → (R, η) map.
artefacts = mi.Mapper(cfg).build(workdir, n_cpus=4)

# 2. Pack TIFF → MIDAS zarr.zip.
zarr_zip = workdir / 'CeO2.zarr.zip'
mi.make_zarr_zip(workdir / 'CeO2_00001.tif', cfg, zarr_zip)

# 3. Integrate.
result = mi.Integrator(cfg, artefacts).integrate(zarr_zip, n_cpus=4)
cake = result.load_cake()          # dict of numpy arrays
print(cake['I'].shape)             # (nR, nEta) — 2D cake
```

Full walkthroughs: see [`notebooks/`](notebooks/).

## Headline feature: `correct_image` (Dioptas / pyFAI / GSAS-II interop)

Rectify a calibrated image against the MIDAS geometry so downstream tools can integrate with flat-detector math (`R = sqrt((y-ybc)² + (z-zbc)²) · px`):

```python
import midas_integrate as mi

rectified = mi.correct_image('sample_0001.tif', geometry=calibration_result.geometry)
mi.write_tiff('sample_0001_corrected.tif', rectified, geometry=calibration_result.geometry)
# Feed the corrected TIFF into Dioptas / pyFAI / GSAS-II with no further geometry fuss.
```

See [`02_correct_image_for_dioptas.ipynb`](notebooks/02_correct_image_for_dioptas.ipynb).

## Live-feed streaming

Socket-based server + client for live beamline acquisition:

```python
with mi.stream.Server(cfg, artefacts, port=60439, backend='cpu') as srv:
    with mi.stream.Client('127.0.0.1:60439') as c:
        for frame in frames:
            cake = c.send_frame(frame)
```

See [`03_streaming_server.ipynb`](notebooks/03_streaming_server.ipynb). GPU backend (`backend='gpu'`) available when `midas-integrate-gpu` is also installed.

## Public API

| Symbol | Purpose |
|---|---|
| `mi.IntegrationConfig(...)` / `.from_geometry(geom)` | User-facing inputs; seed from a `DetectorGeometry`. |
| `mi.Mapper(config)` / `.build(work_dir)` | Wrap `MIDASDetectorMapper`; returns `MapArtifacts`. |
| `mi.Integrator(cfg, artefacts, backend=...)` / `.integrate(zarr_zip)` | Wrap `MIDASIntegrator` CPU/GPU. Returns `IntegrationResult`. |
| `mi.make_zarr_zip(image, cfg, out)` | Pack a 2D frame into MIDAS's zarr schema. |
| `mi.correct_image(image, geometry, …)` | ★ Geometric rectification for Dioptas interop. |
| `mi.correct_images(paths, geometry, out_dir)` | Batch form. |
| `mi.write_tiff(path, arr, geometry=...)` | 32-bit TIFF with MIDAS provenance in ImageDescription. |
| `mi.generate_panels(...)` / `mi.load_panel_shifts(...)` / `mi.Panel` | Per-panel detector support. |
| `mi.stream.Server(...)` / `mi.stream.Client(endpoint)` | Live-feed integration. |
| `mi.fit_peaks_1d(r, intensity, ...)` | scipy pseudo-Voigt peak fits. |
| `mi.load_peaks_h5(path)` | Parse MIDAS's `.caked_peaks.h5` output. |
| `mi.midas_bin(name)` | Cross-package binary discovery (own wheel → auto-calibrate → gpu). |

## Command-line entry points

```bash
midas-integrate <zarr.zip>                 # single-frame integration
midas-correct-image <tif> <geom.json>      # Dioptas interop
midas-integrate-batch <glob>               # parallel batch
midas-integrate-server <params.txt>        # live-feed server
midas-integrate-client host:port <tif...>  # push frames
midas-integrate-fig-b / -fig-c             # Paper B / C figure regen
```

## Notebooks

Under [`notebooks/`](notebooks/):

1. **[01_radial_integration.ipynb](notebooks/01_radial_integration.ipynb)** — Mapper → zarr.zip → Integrator end-to-end.
2. **[02_correct_image_for_dioptas.ipynb](notebooks/02_correct_image_for_dioptas.ipynb)** — ★ rectify a TIFF for downstream tools.
3. **[03_streaming_server.ipynb](notebooks/03_streaming_server.ipynb)** — live-feed server + client loopback.
4. **[04_round_trip_validation.ipynb](notebooks/04_round_trip_validation.ipynb)** — correct → re-calibrate residual check.
5. **[05_benchmark_vs_pyfai.ipynb](notebooks/05_benchmark_vs_pyfai.ipynb)** — head-to-head timing vs pyFAI.

All five execute end-to-end against the bundled CeO₂ data.

## Development

```bash
git clone https://github.com/marinerhemant/MIDAS
cd MIDAS/packages/midas_integrate
pip install -e ../midas_auto_calibrate            # hard dep
pip install -e '.[dev,paper]'
pytest                                            # binary-dependent tests skip in dev
MIDAS_BIN=$(python -c "import midas_integrate, pathlib; \
    print(pathlib.Path(midas_integrate.__file__).parent/'_bin')") pytest
```

Local C builds need CMake ≥ 3.20, C compiler, HDF5, libTIFF, **BLOSC2**, **libzip**
(`brew install cmake hdf5 libtiff libomp c-blosc2 libzip` on macOS;
`apt-get install cmake libhdf5-dev libtiff-dev libomp-dev libblosc2-dev libzip-dev` on Linux).
NLopt is fetched and built in-tree.

## License

BSD-3-Clause.

## For maintainers

Release procedure: [`RELEASING.md`](RELEASING.md).
