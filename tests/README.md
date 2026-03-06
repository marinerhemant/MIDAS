# MIDAS Test Suite

Automated benchmark tests for validating MIDAS pipelines. Each test uses the example data shipped with MIDAS and compares results against known-good references.

## Quick Start

```bash
cd ~/opt/MIDAS

# Run all core tests (4 CPUs)
python tests/test_ff_hedm.py -nCPUs 4
python tests/test_nf_hedm.py -nCPUs 4
python tests/test_ff_calibration.py -nCPUs 4
python tests/test_integrator_peaks.py -nCPUs 4
python tests/test_phase_id.py -nCPUs 4
```

---

## Tests

### `test_ff_hedm.py` — Far-Field HEDM Full Pipeline

End-to-end test of the FF-HEDM analysis pipeline:

1. Runs `ForwardSimulationCompressed` on example `GrainsSim.csv`
2. Enriches the output Zarr with metadata
3. Runs `ff_MIDAS.py` (peaksearch → merge → radius → indexing → refinement)
4. Compares the consolidated HDF5 output stage-by-stage against a reference

**Optional modes:**
- `--px-overlap` — Run pixel-overlap peaksearch variant
- `--dual-dataset` — Run dual-dataset refinement (same data twice, zero offset)
- `--no-cleanup` — Keep generated files for inspection
- `--cleanup-only` — Remove stale test artifacts without running

```bash
python tests/test_ff_hedm.py -nCPUs 4 --px-overlap --dual-dataset
```

**Data:** `FF_HEDM/Example/`

---

### `test_nf_hedm.py` — Near-Field HEDM Reconstruction

1. Runs `simulateNF` to produce `SpotsInfo.bin`
2. Patches the parameter file for reconstruction
3. Runs `nf_MIDAS.py` (seed orientations → reconstruction)
4. Compares reconstructed mic against reference using misorientation statistics

**Pass criterion:** >80% of voxels within 0.25° misorientation of reference.

```bash
python tests/test_nf_hedm.py -nCPUs 4
```

**Data:** `NF_HEDM/Example/`

---

### `test_ff_calibration.py` — Calibrant Ring Fitting

Runs `CalibrantPanelShiftsOMP` on CeO2 calibration data and validates that the resulting mean strain is below threshold, ensuring detector geometry refinement is working correctly.

```bash
python tests/test_ff_calibration.py -nCPUs 4
```

**Data:** `FF_HEDM/Example/Calibration/`

---

### `test_integrator_peaks.py` — Integrator + Peak Fitting

1. Runs calibrant fitting to get optimized geometry
2. Creates a Zarr zip from the calibration TIFF
3. Runs `IntegratorZarrOMP` with peak fitting enabled
4. Compares fitted peak centers against theoretical ring radii

Validates that the integration + 1D peak fitting pipeline produces positions matching crystallographic predictions.

```bash
python tests/test_integrator_peaks.py -nCPUs 4
python tests/test_integrator_peaks.py -nCPUs 4 --keep-work-dir
```

**Data:** `FF_HEDM/Example/Calibration/`

---

### `test_phase_id.py` — Phase Identification

Tests both Zarr and direct-mode pipelines for phase identification:

1. Predicts ring positions for CeO2 (present) and Au (absent)
2. **Pipeline A:** Zarr integration + peak fitting
3. **Pipeline B:** Direct-mode integration + peak fitting
4. Verifies CeO2 detected, Au absent, lattice constant within 500 ppm

```bash
python tests/test_phase_id.py -nCPUs 4
python tests/test_phase_id.py --keep-work-dir --work-dir /tmp/phase_test
```

**Data:** `FF_HEDM/Example/Calibration/`

---

### `test_live_viewer.py` — Live Viewer Data Generator

Generates synthetic `lineout.bin` data with visible pseudo-Voigt peaks for testing `live_viewer.py`. Simulates the GPU integrator's interactive peak selection protocol:

- Starts with 3 visible peaks in the lineout, no `fit.bin`
- When `peak_update.txt` is received (from the viewer's Send Peaks button), begins writing `fit.bin`

```bash
# Terminal 1: start generator
python tests/test_live_viewer.py --fps 10

# Terminal 2: start viewer
python gui/viewers/live_viewer.py --lineout lineout.bin --fit fit.bin \
    --nRBins 500 --nPeaks 0 --params test_params.txt --theme dark
```

---

## Common Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-nCPUs` | Number of CPUs for parallel processing | 1–4 (varies) |
| `--no-cleanup` | Keep generated test artifacts | off |
| `--keep-work-dir` | Keep working directory after test | off |
| `-paramFN` | Override the default parameter file | Example data |

## Prerequisites

- MIDAS compiled (`cmake --build build/`)
- Python packages: `numpy`, `zarr`, `h5py`, `pandas`
- Example data present in `FF_HEDM/Example/` and `NF_HEDM/Example/`
