# midas-auto-calibrate

Fully automated detector geometry calibration for synchrotron area detectors.

Part of the [MIDAS](https://github.com/marinerhemant/MIDAS) toolkit. Backs
**Paper A** — *Fully Automated Detector Geometry Calibration for
High-Energy X-ray Diffraction* (IUCrJ, pending).

## Install

```bash
pip install midas-auto-calibrate                 # core + bundled CeO₂ test data
pip install 'midas-auto-calibrate[viz]'          # +matplotlib for static plots
pip install 'midas-auto-calibrate[viz-gui]'      # +PyQt5 interactive viewer
pip install 'midas-auto-calibrate[paper]'        # +pyFAI/fabio for Paper A regen
```

Binary wheels for **Linux x86_64 (manylinux_2_28)** and **macOS (x86_64 + arm64)**.
The `MIDASCalibrant` and `GetHKLList` C binaries ship inside the wheel — no
compiler needed on the install machine.

**Windows**: use WSL2 + the Linux wheel. Native Windows wheels are not planned.

## Quick start

```python
import shutil, tempfile
from pathlib import Path
import midas_auto_calibrate as mac

# The wheel ships a CeO₂ Pilatus frame. Stage it into a writable workdir.
workdir = Path(tempfile.mkdtemp(prefix='mac_'))
shutil.copy(mac.data.CEO2_PILATUS,      workdir / 'CeO2_00001.tif')
shutil.copy(mac.data.CEO2_PILATUS_DARK, workdir / 'dark.tif')
shutil.copy(mac.data.CEO2_PILATUS_MASK, workdir / 'mask_upd.tif')

config = mac.CalibrationConfig(
    material='CeO2',
    lattice_params=(5.4116, 5.4116, 5.4116, 90, 90, 90),
    wavelength=0.172973,
    pixel_size=172.0,
    lsd=657_436.9, ybc=685.5, zbc=921.0,
    nr_pixels_y=1475, nr_pixels_z=1679,
    dark_file='dark.tif', mask_file='mask_upd.tif',
    im_trans_opt=[2],
    n_iterations=5,
)

result = mac.auto_calibrate(config, workdir / 'CeO2_00001.tif', work_dir=workdir)
print(f'{result.pseudo_strain:.2f} µε — Lsd={result.geometry.lsd:.1f} µm')

result.geometry.to_json(workdir / 'calibration.json')
result.geometry.to_midas_params(workdir / 'Parameters.txt')
```

Full walkthrough: see [`notebooks/`](notebooks/).

## Why it matters

- **Fully automated** — no manual peak picking, no ring assignment, no
  interactive geometry refinement.
- **4.6–30× more accurate** than pyFAI's automated calibration across the
  four detector types benchmarked in Paper A.
- **15-parameter analytical distortion model** (tilt + spherical + dipole +
  trefoil + octupole) plus residual TPS spline.
- **Multi-module detector support** (Pilatus, Eiger) with per-panel
  shift + rotation refinement.
- **Cardinal-angle aliasing correction** built in (`GradientCorrection`).

## Public API

| Symbol | Purpose |
|---|---|
| `mac.auto_calibrate(config, image, …)` | Run the full calibration pipeline; returns `CalibrationResult`. Alias for `run_calibration`. |
| `mac.CalibrationConfig(…)` | User-facing inputs (material, wavelength, starting geometry, iteration count, dark/mask files). |
| `mac.CalibrationResult` | `.geometry`, `.pseudo_strain`, `.convergence_history`, `.corr_csv_path`, `.stdout`. |
| `mac.DetectorGeometry` | Refined output dataclass. `.to_json()`, `.to_midas_params()`, `.from_json()`, `.from_midas_params()`. |
| `mac.benchmark(image, material, wavelength, …)` | Head-to-head vs pyFAI. Returns `BenchmarkResult`; `.as_dict()` matches the paper's table format. |
| `mac.midas_bin(name)` | Resolve a MIDAS C binary (searches wheel `_bin/`, `MIDAS_BIN`, `MIDAS_INSTALL_DIR`, PATH). |
| `mac.data.CEO2_PILATUS` | Path to the bundled Pilatus calibrant frame. Also `CEO2_PILATUS_DARK`, `CEO2_PILATUS_MASK`. |
| `midas_auto_calibrate.viz.static` | `convergence`, `rings_overlay`, `residual_heatmap`, `fourier_harmonics`, `distortion_field`, `inspect`. |

## pyFAI / Dioptas / GSAS-II interop

This package deliberately **does not** export to pyFAI `.poni` — the tilt
conventions don't round-trip faithfully. If you need a downstream tool to
consume MIDAS-calibrated data, use
[midas-integrate](https://pypi.org/project/midas-integrate/)'s
`correct_image()` (ships in the same release wave) to produce a
geometrically rectified TIFF that any tool can integrate with simple
`R = sqrt(dy² + dz²) · px` geometry.

## Command-line entry points

```bash
midas-auto-calibrate                # full calibration pipeline
midas-calib-benchmark               # pyFAI head-to-head (needs [paper] extra)
midas-calib-validate                # synthetic-tilt validation matrix
midas-calib-inspect <work_dir>      # render the 5-plot static bundle ([viz])
midas-calib-viewer <work_dir>       # PyQt5 interactive viewer ([viz-gui])
midas-calib-fig-{convergence,fourier,distortion,stage}  # Paper A figure regen
```

## Notebooks

Under [`notebooks/`](notebooks/):

1. **[01_auto_calibrate.ipynb](notebooks/01_auto_calibrate.ipynb)** — fresh run on the bundled CeO₂ frame, saves refined geometry as JSON + Parameters.txt.
2. **[02_inspect_results.ipynb](notebooks/02_inspect_results.ipynb)** — the five static plots (convergence, rings overlay, residual heatmap, Fourier harmonics, distortion field) on a real result.
3. **[03_benchmark_vs_pyfai.ipynb](notebooks/03_benchmark_vs_pyfai.ipynb)** — MIDAS vs pyFAI head-to-head.

All three execute end-to-end against the bundled data (~25 s total).

## Paper A — Table 4 regeneration

```bash
python scripts/regen_table4.py --out /tmp/table4.csv
```

The bundled CeO₂ Pilatus frame is included in the wheel; the other three
detectors (Pilatus 6M, Varex 4343CT, Ceria) live on Zenodo to keep the
wheel small — see `mac.data.zenodo_url()` for the DOI (populated on paper
submission).

## Development

```bash
git clone https://github.com/marinerhemant/MIDAS
cd MIDAS/packages/midas_auto_calibrate
pip install -e '.[dev,viz,paper]'
pytest                               # binary-dependent tests skip in dev
MIDAS_BIN=$(python -c "import midas_auto_calibrate, pathlib; print(pathlib.Path(midas_auto_calibrate.__file__).parent/'_bin')") \
    pytest                           # full suite incl. end-to-end
```

Local C builds need CMake ≥ 3.20, a C compiler, HDF5, and libTIFF
(`brew install cmake hdf5 libtiff` on macOS;
`apt-get install cmake libhdf5-dev libtiff-dev libomp-dev` on Linux).
NLopt is fetched and built in-tree by CMake — no system install required.

## License

BSD-3-Clause.

## For maintainers

Release procedure: [`RELEASING.md`](RELEASING.md).
