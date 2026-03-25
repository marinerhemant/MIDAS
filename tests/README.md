# MIDAS Test Suite

Automated benchmark tests for validating your MIDAS installation. Each test uses example data shipped with MIDAS, runs the real pipeline binaries, and compares results against known-good reference outputs. If all tests pass, your build is working correctly.

## Prerequisites

Before running any test, ensure:

1. **MIDAS is compiled:**
   ```bash
   cd ~/opt/MIDAS/build && cmake --build . -j$(nproc)
   ```
2. **Python packages** are installed: `numpy`, `zarr`, `h5py`, `pandas`
3. **Example data** exists in `FF_HEDM/Example/` and `NF_HEDM/Example/`
4. **`MIDAS_HOME`** environment variable is set (or run from the MIDAS root directory — the tests auto-detect it)

> [!TIP]
> Every test automatically runs pre-flight checks to verify these prerequisites before starting. If something is missing, you'll get a clear error message with the fix command.

## Quick Start

```bash
cd ~/opt/MIDAS

# Core pipeline tests (recommended for new installations)
python tests/test_ff_hedm.py -nCPUs 4          # ~60s
python tests/test_nf_hedm.py -nCPUs 4          # ~120s
python tests/test_calibration_integration.py -nCPUs 4  # ~90s (calibration + integration)
python tests/test_phase_id.py -nCPUs 4         # ~60s
python tests/test_tomo.py -nCPUs 4             # ~30s
```

---

## Built-in Diagnostics

All tests share a common diagnostics framework (`test_common.py`) that runs automatically:

### Automatic Checks

Every test starts with:

1. **Environment fingerprint** — printed at the top of every test run:
   ```
   ┌────────────────────────────────────────────────────────────────────┐
   │  MIDAS Test Environment                                          │
   ├────────────────────────────────────────────────────────────────────┤
   │  Platform : macOS-14.3-arm64-arm-64bit                           │
   │  Machine  : arm64                                                │
   │  Python   : 3.12.2                                               │
   │  Compiler : Apple clang version 15.0.0                           │
   │  MIDAS    : /Users/hsharma/opt/MIDAS                             │
   │  Packages : numpy=1.26.4, zarr=2.16.1, h5py=3.10.0              │
   └────────────────────────────────────────────────────────────────────┘
   ```

2. **Pre-flight checks:**
   - All required binaries exist and are executable
   - Required Python packages are importable
   - Required data files exist and are non-empty
   - Sufficient disk space (≥500 MB)

3. **Binary staleness detection** — warns if source files are newer than compiled binaries:
   ```
   ⚠️  Binary may be stale: PeaksFittingOMPZarrRefactor
       Source PeaksFittingOMPZarrRefactor.c is newer than compiled binary
       Recompile with: cd ~/opt/MIDAS/build && cmake --build . --target PeaksFittingOMPZarrRefactor
   ```

### Diagnostic Flags

When a test fails, use these flags for investigation:

| Flag | What It Does |
|---|---|
| `--diagnose` | Dumps a full diagnostic report (JSON) with environment info, per-dataset comparison results, and mismatch details |
| `--save-on-fail` | Saves the generated output file alongside the reference for manual comparison |
| `--skip-preflight` | Skip pre-flight checks (useful if you know what you're doing) |

```bash
# Example: investigate a failing FF-HEDM test
python tests/test_ff_hedm.py -nCPUs 4 --diagnose --save-on-fail
```

The `--diagnose` flag generates a JSON report like:
```
📋 Diagnostic report saved: tests/diagnostic_test_ff_hedm_20260307_141500.json
   Send this file to hsharma@anl.gov for investigation.
```

---

## Test 1: Far-Field HEDM Full Pipeline (`test_ff_hedm.py`)

### What It Tests

The complete FF-HEDM analysis pipeline, end-to-end: from simulated diffraction images through peak detection, spot merging, radius calculation, orientation indexing, and grain refinement.

### How It Works

```
Step 1: ForwardSimulationCompressed
   Input:  GrainsSim.csv (3 known Au grains with predefined orientations)
           Parameters.txt (detector geometry, wavelength=0.172979 Å, Lsd, etc.)
   Action: Simulates diffraction patterns for all omega angles
   Output: Au_FF_000001_pf_scanNr_0.zip (raw Zarr with simulated detector frames)

Step 2: enrich_zarr_metadata()
   Input:  The raw Zarr from Step 1
   Action: Injects analysis parameters into the Zarr archive:
           - SpaceGroup, LatticeConstant, Wavelength, Lsd
           - RingThresh (per-ring intensity thresholds)
           - OmegaRanges, BoxSizes, ImTransOpt
           Uses ffGenerateZipRefactor.write_analysis_parameters()
   Output: Enriched .MIDAS.zip ready for processing

Step 3: ff_MIDAS.py pipeline
   Action: Runs the full FF-HEDM workflow:
           a) PeaksFittingOMPZarrRefactor — finds peaks in each frame
           b) MergeOverlappingPeaksAll — merges peaks across omega frames
           c) CalcRadius — converts pixel positions to ring radii
           d) FitSetup + Indexing — assigns spots to grains
           e) FitGrain — refines grain orientations and positions
   Output: *_consolidated.h5 with all results
```

### What Gets Compared

The test compares the new `*_consolidated.h5` against a reference file (`consolidated_Output.h5`) across **6 pipeline stages**, each containing multiple HDF5 datasets:

| Stage | Datasets Compared | Comparison Method |
|---|---|---|
| **PeaksFitting (summary)** | `peaks/summary/data` | `np.allclose(atol=1e-6, rtol=1e-6)` |
| **PeaksFitting (per-frame)** | `peaks/per_frame/data` | `np.allclose(atol=1e-6, rtol=1e-6)` |
| **MergeOverlaps** | `merge_map/{MergedSpotID,FrameNr,PeakID}`, `id_rings/data`, `ids_hash/data` | Exact match for integers, `allclose` for floats |
| **CalcRadius** | `radius_data/{SpotID,IntegratedIntensity,Omega,YCen,ZCen,IMax,Radius,Eta,RingNr,GrainRadius,SigmaR,SigmaEta}` | `np.allclose(atol=1e-6, rtol=1e-6)` |
| **FitSetup (InputAll)** | `all_spots/data`, `spots_to_index/data` | `allclose` for floats, exact for integers |
| **Grains (final)** | `grains/summary`, `spot_matrix/data`, `grain_ids_key/data` | `allclose` for floats, exact for integers |

### Pass Criteria

**All 6 stages must show PASS** — every dataset must match within `atol=1e-6, rtol=1e-6` for floating-point arrays, or exactly for integer arrays.

### Optional Modes

```bash
# Pixel-overlap peaksearch variant
python tests/test_ff_hedm.py -nCPUs 4 --px-overlap

# Dual-dataset refinement (feeds same data twice with zero offset)
python tests/test_ff_hedm.py -nCPUs 4 --dual-dataset

# Keep generated files for inspection
python tests/test_ff_hedm.py -nCPUs 4 --no-cleanup

# Clean up stale test artifacts without running
python tests/test_ff_hedm.py --cleanup-only
```

**Dual-dataset mode** additionally compares two `Grains.csv` files, checking:
- Grain count matches
- Position mismatch < 1.0 µm
- Orientation mismatch < 0.01°

**Data directory:** `FF_HEDM/Example/`

---

## Test 2: Near-Field HEDM Reconstruction (`test_nf_hedm.py`)

### What It Tests

The NF-HEDM microstructure reconstruction pipeline: simulating diffraction spot patterns from a known microstructure, then reconstructing that microstructure from the simulated data.

### How It Works

```
Step 1: simulateNF
   Input:  Parameters.txt + reference .mic file (known microstructure)
   Action: Simulates NF diffraction patterns → SpotsInfo.bin
           The mic file contains per-voxel Euler angles for a polycrystal
   Output: SpotsInfo.bin (binary spot list)

Step 2: Prepare parameter file
   Action: Patches the parameter file with:
           - DataDirectory → working directory
           - GrainsFile and SeedOrientations paths

Step 3: nf_MIDAS.py
   Action: Runs NF reconstruction:
           a) Generates seed orientations
           b) Reconstructs microstructure from SpotsInfo.bin
   Output: Reconstructed .mic file

Step 4: Orientation comparison
   Action: Matches voxels by (X, Y) position between reference and test .mic
           For each matched voxel, computes misorientation angle using
           calcMiso.GetMisOrientationAngle() with SpaceGroup symmetry (SG=225 for FCC)
```

### What Gets Compared

- **Voxel matching:** By exact (X, Y) position between reference and reconstructed mic files
- **Misorientation calculation:** Uses the MIDAS `calcMiso` module which accounts for crystal symmetry (cubic here, SG 225)
- **Euler angles:** Read from mic file columns 7–9 (degrees), converted to radians for misorientation calculation

### Pass Criteria

**>80% of matched voxels must have misorientation < 0.25°**

The test also prints statistics at multiple thresholds for diagnostic purposes:

| Threshold | Typical Result |
|---|---|
| < 0.25° | >80% (must pass) |
| < 1.0° | ~95% |
| < 2.0° | ~98% |
| < 5.0° | ~99% |

```bash
python tests/test_nf_hedm.py -nCPUs 4
```

**Data directory:** `NF_HEDM/Example/`

---

## Test 3: Calibration + Integration (`test_calibration_integration.py`)

### What It Tests

Combined test for detector geometry calibration and azimuthal integration peak fitting. First validates that `CalibrantPanelShiftsOMP` produces low strain residuals, then validates that `IntegratorZarrOMP` produces peak positions matching crystallographic predictions from CeO2.

### How It Works

```
Step 0: Calibration (CalibrantPanelShiftsOMP)
   Action: Runs calibrant fitting to optimize detector geometry
   Validates: MeanStrain ≤ 50 µε (configurable via -strainThreshold)

   With --calibration-only, stops here.

Step 1: Create Zarr zip
   Action: Creates a MIDAS Zarr zip from the calibration TIFF

Step 2: GetHKLList → ring radii
   Action: Generates theoretical ring radii in µm
           Converts to pixels: R_px = R_µm / PixelSize

Step 3: DetectorMapper
   Action: Pre-computes the detector geometry mapping

Step 4: IntegratorZarrOMP (with peak fitting)
   Input:  Zarr zip + peak_params.txt (with PeakLocation entries)
   Action: Azimuthal integration followed by 1D pseudo-Voigt peak fitting
   Output: _lineout.bin (1D profile), _fit.bin (fitted peak parameters)

Step 5: Strain benchmark comparison
   Action: For each ring, compute:
           ΔR = R_fitted - R_theory
           strain (ppm) = (ΔR / R_theory) × 10⁶
```

### Pass Criteria

- **Calibration:** MeanStrain ≤ 50 µε (configurable via `-strainThreshold`)
- **Integration:** Max |strain residual| < 500 ppm

```bash
python tests/test_calibration_integration.py -nCPUs 4
python tests/test_calibration_integration.py -nCPUs 4 --calibration-only
python tests/test_calibration_integration.py -nCPUs 4 -strainThreshold 200
python tests/test_calibration_integration.py -nCPUs 4 --mode autodetect
python tests/test_calibration_integration.py -nCPUs 4 --keep-work-dir
python tests/test_calibration_integration.py -nCPUs 4 --robustness-test  # 4 configs: baseline, outlier removal, trimmed mean, both
```

**Data directory:** `FF_HEDM/Example/Calibration/`

---

## Test 4: Phase Identification (`test_phase_id.py`)

### What It Tests

The phase identification pipeline, validating that MIDAS can correctly identify which crystallographic phases are present in diffraction data. Uses CeO2 calibrant data and tests that CeO2 is detected as **present** and Au as **absent**.

### How It Works

```
Step 1: Define test phases
   Phases file with two entries:
   - CeO2: SpaceGroup=225, a=5.4116 Å (PRESENT in data)
   - Au:   SpaceGroup=225, a=4.0782 Å (ABSENT from data)

Step 2: Pipeline A — Zarr integration
   Action: Creates Zarr zip → DetectorMapper → IntegratorZarrOMP
           Fits peaks at predicted ring positions for both phases

Step 3: Pipeline B — Direct mode
   Action: IntegratorZarrOMP reads TIFF directly (no Zarr creation)
           Same peak fitting as Pipeline A

Step 4: Assertions (run on both pipelines)
   For each fitted peak:
   a) Detection: peak is "detected" if Imax > 0, SNR ≥ 5.0,
      Imax ≥ 1% of global max, and Sigma ≥ 0.5 × RBinSize
   b) Lattice back-calculation: from the fitted peak center,
      compute the lattice parameter a using Bragg's law:
      a = d_hkl × √(h² + k² + l²)
      where d_hkl = λ / (2 sin(θ)) and θ is derived from R_fitted
```

### What Gets Compared

| Assertion | Condition | Threshold |
|---|---|---|
| CeO2 detected | ≥30% of CeO2 rings have peaks | 30% |
| Au absent | 0 Au rings detected | Exactly 0 |
| CeO2 lattice constant | \|a_measured − 5.4116\| / 5.4116 < 500 ppm | 500 ppm |
| CeO2 confidence score | Combined score ≥ 50 | 50 |

The **confidence score** combines:
- Ring detection fraction (exclusive + overlap)
- Lattice parameter accuracy
- Intensity fraction relative to total

### Pass Criteria

**All 4 assertions must pass on both pipelines (Zarr and Direct).**

```bash
python tests/test_phase_id.py -nCPUs 4
python tests/test_phase_id.py --keep-work-dir --work-dir /tmp/phase_test
```

**Data directory:** `FF_HEDM/Example/Calibration/`

---

## Test 5: Tomography Reconstruction (`test_tomo.py`)

### What It Tests

The MIDAS tomography reconstruction pipeline using a fully synthetic Shepp-Logan phantom. No example data needed — everything is generated from scratch.

### How It Works

```
Step 1: Generate Shepp-Logan phantom
   Action: Creates a 256×256 (configurable) phantom image with known
           analytical features. Pixel values in [0, 1].
   Records: centre pixel value for later comparison

Step 2: Compute sinograms
   Action: Radon transform (scikit-image) at 1800 projection angles
           over [0°, 360°)
   Output: sinogram matrix (nAngles × nDetectors)

Step 3: Simulate raw detector data
   Action: Convert sinograms to I/I₀ transmission:
           I = I₀ × exp(-µ × sinogram) + dark_level
           where I₀=30000, dark_level=100, µ is scaled so max
           attenuation ≈ 3.0
           Clipped to uint16 [0, 65535]
   Output: data (nThetas+2, nSlices, xDim), dark field, white fields

Step 4: Pipeline A — run_tomo() (full pipeline)
   Action: Takes raw detector data → dark/white normalization →
           negative-log transform → filtered back-projection (FBP)
           using gridrec algorithm
   Output: Reconstructed slice

Step 5: Pipeline B — run_tomo_from_sinos() (sinogram pipeline)
   Action: Takes pre-computed sinograms → FBP (no log transform)
   Output: Reconstructed slice

Step 6: Quality comparison
   a) Crop: reconstruction is padded to power-of-2; crop back to
      phantom size with 90° rotation correction
   b) Normalize: scale to [0, 1]
   c) Compute metrics against original phantom
```

### What Gets Compared

| Metric | How Computed | Threshold |
|---|---|---|
| Pearson correlation | `Σ((a-ā)(b-b̄)) / √(Σ(a-ā)² × Σ(b-b̄)²)` | > 0.85 |
| Centre value accuracy | `|recon_centre − phantom_centre| / phantom_centre` | < 10% |

Both metrics are checked for **both pipelines** (full and sinogram).

### Pass Criteria

**Both pipelines must achieve Pearson correlation > 0.85 AND centre value error < 10%.**

```bash
python tests/test_tomo.py -nCPUs 4

# Faster (smaller phantom)
python tests/test_tomo.py --phantom-size 128 --n-thetas 360

# Keep output for inspection + show plots
python tests/test_tomo.py --keep-work-dir --plot
```

**Data:** Fully synthetic — no example data needed.

---

## Test 6: Live Viewer Data Generator (`test_live_viewer.py`)

### What It Tests

This is **not an automated pass/fail test** — it is a data generator for manually testing the `live_viewer.py` interactive GUI. It simulates a GPU integrator's binary output stream for real-time visualization.

### How It Works

```
Step 1: Write mock parameter file
   Content: Lsd=1,000,000 µm, px=200 µm, λ=0.172979 Å
            NrPixels=2048, RMin=100, RMax=600, RBinSize=1.0

Step 2: Generate lineout data
   Action: Creates 3 visible pseudo-Voigt peaks in a 1D radial profile:
           Peak 1: R=200 px, σ=8, A=800
           Peak 2: R=350 px, σ=10, A=600
           Peak 3: R=480 px, σ=7, A=500
   Format: lineout.bin — interleaved [R₀, I₀, R₁, I₁, ...]
           (nRBins × 2 doubles per frame)

Step 3: Wait for peak_update.txt
   Action: Does NOT write fit.bin initially
           When live_viewer sends peak_update.txt (user clicks
           Pick → selects peaks → Send), starts writing fit.bin
   Format: fit.bin — 7 doubles per peak per frame:
           [Imax, BG, η, Center, σ, GoF, Area]
```

### How to Run

```bash
# Terminal 1: start data generator
python tests/test_live_viewer.py --fps 10

# Terminal 2: start the viewer
python gui/viewers/live_viewer.py --lineout lineout.bin --fit fit.bin \
    --nRBins 500 --nPeaks 0 --params test_params.txt --theme dark

# In the viewer: click 🎯 Pick → click on peaks → 📤 Send (Replace)
```

### Pass Criteria

**Manual verification** — the viewer should show 3 visible peaks in the lineout. After clicking Send, fitted curves should appear on top of the peaks.

---

## Common Arguments

| Argument | Description | Default |
|---|---|---|
| `-nCPUs` | Number of CPUs for parallel processing | 1–4 (varies) |
| `--no-cleanup` | Keep generated test artifacts | off |
| `--keep-work-dir` | Keep working directory after test | off |
| `--cleanup-only` | Remove stale test artifacts without running | off |
| `-paramFN` | Override the default parameter file | Example data |
| `--diagnose` | Generate diagnostic report on failure | off |
| `--save-on-fail` | Save generated output for comparison | off |
| `--skip-preflight` | Skip pre-flight checks | off |

## Interpreting Failures

### What Happens on Failure

When a test fails, the diagnostics system provides:

1. **Actionable error message** explaining the likely cause:
   ```
   💡 Shape mismatch on 'peaks/summary/data': ref=(3, 20), got=(2, 20).
      The number of peaks/frames differs — check if the parameter file
      was modified or if a pipeline stage crashed early.
   ```

2. **First few mismatched values** (expected vs. actual):
   ```
   First 5 mismatched values:
   Index    Reference        Got              Diff
   42       1234.56789       1234.56800       1.10e-04
   ```

3. **Difference histogram** showing the distribution:
   ```
   Distribution of differences:
   exact match :  15432 (95.2%) ████████████████████████████████████████████████
   < 1e-10     :    450 ( 2.8%) █
   1e-10 – 1e-6:    325 ( 2.0%) █
   ```

4. **Instructions to generate a diagnostic report:**
   ```
   To investigate further:
     python tests/test_ff_hedm.py --diagnose
     python tests/test_ff_hedm.py --save-on-fail

   Send the generated diagnostic report to hsharma@anl.gov
   for assistance with debugging.
   ```

### Common Failure Causes

| Symptom | Likely Cause | Fix |
|---|---|---|
| Binary not found | Not compiled | `cd build && cmake --build . --target <name>` |
| Binary may be stale | Source edited after compile | `cd build && cmake --build .` |
| Shape mismatch | Parameter file changed | Restore original `Parameters.txt` |
| Small float diffs (<1e-4) | Platform numeric noise | Normal on some architectures |
| All zeros in output | Pipeline crash | Check stderr output |
| Reference file missing | Example data incomplete | Re-clone the repository |
| NF misorientation >0.25° | Seed orientation issue | Check SpaceGroup is correct |
| Calibrant strain >50 µε | Corrupted TIFF or wrong geometry | Verify calibration data |

### Getting Help

If a test fails and you can't resolve it:

1. Run with `--diagnose` to generate a diagnostic JSON report
2. Email the report file to **hsharma@anl.gov**
3. Include the test command you ran and the full terminal output

## File Structure

```
tests/
├── README.md                       # This file
├── test_common.py                  # Shared diagnostics framework
├── test_ff_hedm.py                 # FF-HEDM full pipeline benchmark
├── test_nf_hedm.py                 # NF-HEDM reconstruction benchmark
├── test_calibration_integration.py  # Calibration + integration benchmark
├── test_phase_id.py                # Phase identification benchmark
├── test_tomo.py                    # Tomography reconstruction benchmark
├── test_pf_hedm.py                 # PF/scanning HEDM pipeline benchmark (consolidated I/O)
├── test_tomo_parity.py             # GPU vs CPU tomography parity test
└── test_live_viewer.py             # Live viewer data generator (manual)
```

### GPU Testing Flags

Several tests support GPU-specific options:

| Test | GPU Flag | Description |
|---|---|---|
| `test_ff_hedm.py` | `-useGPU` | GPU-accelerated indexing and fitting |
| `test_nf_hedm.py` | `--gpu-fit` | GPU-accelerated NF orientation fitting |
| `test_pf_hedm.py` | `-useGPU` | GPU scanning indexer and fitter |
| `test_tomo_parity.py` | `--gpu-only` | Skip preprocessing and CPU comparison |

Additional flags for `test_tomo_parity.py`: `--phantom-size`, `--n-thetas`, `--plot`, `--small`.
