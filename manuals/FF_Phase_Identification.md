# FF Phase Identification

Multi-phase identification from powder diffraction images using MIDAS integration and peak fitting.

## Overview

`phase_id.py` identifies crystallographic phases present in 2D diffraction images by:
1. Predicting diffraction ring positions for each candidate phase
2. Integrating the 2D image to a 1D radial profile
3. Fitting pseudo-Voigt peaks at predicted ring positions
4. Back-calculating lattice parameters from fitted peak centers
5. Reporting per-ring results and per-phase detection summary

The tool supports both CPU (`IntegratorZarrOMP`) and GPU (`IntegratorFitPeaksGPUStream`) backends.

## Workflow

```mermaid
flowchart TD
    subgraph Input
        IMG["📷 Diffraction Image(s)"]
        GEOM["📄 geometry.txt<br/>Lsd, BC, tilts, λ, px"]
        PHASES["📄 phases.txt<br/>name, SG, a per phase"]
    end

    subgraph S1["Step 1 · Ring Prediction"]
        LOOP["For each phase"]
        TMP["Temp param file<br/>with SG + a override"]
        HKL["GetHKLList --stdout<br/>→ h,k,l, d, R per reflection"]
        DEDUP["Sort by R, merge<br/>overlapping rings<br/>within 2×RBinSize"]
    end

    PHASES --> LOOP --> TMP --> HKL --> DEDUP
    GEOM --> TMP

    subgraph S2["Step 2 · Integration + Peak Fitting"]
        direction LR
        CPU["CPU: ffGenZip →<br/>DetectorMapperZarr →<br/>IntegratorZarrOMP"]
        GPU["GPU: integrator_batch_process →<br/>IntegratorFitPeaksGPUStream"]
    end

    IMG --> S2
    DEDUP -->|"peak_params.txt<br/>or PeakLocation lines"| S2

    subgraph S3["Step 3 · Analysis"]
        PARSE["Parse fit.bin<br/>7 doubles per peak"]
        FILTER["Dual filter:<br/>SNR ≥ threshold<br/>Imax ≥ 1% of max"]
        CALC["Back-calculate<br/>a = d × √ h²+k²+l² "]
    end

    S2 -->|"fit.bin"| PARSE --> FILTER --> CALC

    subgraph S4["Step 4 · Report"]
        T1["Table A: Per-ring<br/>R, Imax, SNR, a_fitted"]
        T2["Table B: Phase summary<br/>coverage, mean/std a"]
        T3["Table C: Intensity<br/>statistics"]
    end

    CALC --> T1 --> T2 --> T3
```

## Quick Start

```bash
# Basic usage — identify phases in a single image
python utils/phase_id.py \
    -paramFN geometry.txt \
    -dataFN scan_001.tif \
    -phases phases.txt \
    -darkFN dark.tif \
    -nCPUs 4

# Multiple explicit files
python utils/phase_id.py \
    -paramFN geometry.txt \
    -dataFN scan_001.tif scan_002.tif scan_003.tif \
    -phases phases.txt

# Number range (replaces last number in filename)
python utils/phase_id.py \
    -paramFN geometry.txt \
    -dataFN scan_000001.tif \
    -startNr 1 -endNr 100 \
    -phases phases.txt

# Process all files in a folder
python utils/phase_id.py \
    -paramFN geometry.txt \
    -dataFolder /data/pressure_series/ \
    -phases phases.txt

# GPU backend
python utils/phase_id.py \
    -paramFN geometry.txt \
    -dataFN scan_001.tif \
    -phases phases.txt \
    -backend gpu
```

## Input Files

### Parameter File (`-paramFN`)

Standard MIDAS parameter file with detector geometry. Required keys:
- `Lsd` — sample-to-detector distance (µm)
- `BC` — beam center (y z) in pixels
- `px` — pixel size (µm)
- `Wavelength` — X-ray wavelength (Å)
- `RMin`, `RMax`, `RBinSize` — radial integration limits and bin size
- `ty`, `tz`, `p0`–`p4` — detector tilts and distortion coefficients
- `SpaceGroup`, `LatticeConstant` — placeholder values (overridden per phase)
- `MaskFile` — bad pixel mask (optional but recommended)

### Phases File (`-phases`)

Simple text file defining candidate crystal structures. One phase per line:
```text
# name  spacegroup  lattice_a(Å)
CeO2    225         5.4116
LaB6    221         4.1569
Au      225         4.0782
Fe_bcc  229         2.8665
```

Columns:
| Column | Description |
|--------|-------------|
| `name` | Phase label (no spaces) |
| `spacegroup` | Space group number (International Tables) |
| `lattice_a` | Cubic lattice parameter in Ångströms |

> [!NOTE]
> Only **cubic** crystal systems are currently supported. All six lattice parameters are set to `a a a 90 90 90`.

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-paramFN` | required | Geometry parameter file |
| `-dataFN` | — | Data file(s). Multiple accepted. Used as template with `-startNr`/`-endNr`. |
| `-dataFolder` | — | Process all TIFF/HDF5 files in this folder |
| `-startNr` | — | Starting file number (requires `-dataFN` template) |
| `-endNr` | — | Ending file number (requires `-startNr`) |
| `-phases` | required | Phase definitions file |
| `-darkFN` | — | Dark frame for subtraction |
| `-nCPUs` | 4 | Number of CPU threads |
| `-backend` | `cpu` | `cpu` (IntegratorZarrOMP) or `gpu` (IntegratorFitPeaksGPUStream) |
| `--snr-threshold` | 5.0 | Minimum SNR for a peak to be "detected" |
| `--rel-intensity-threshold` | 0.01 | Minimum Imax as fraction of strongest peak (1% default) |
| `--max-rings` | 20 | Maximum rings to fit per phase |
| `--roi-padding` | 30 | Peak fit ROI half-width in radial bins |
| `--merge-threshold` | 2×RBinSize | Ring deduplication threshold (pixels) |
| `--keep-work-dir` | — | Preserve temporary working directory |
| `--work-dir` | — | Use specific working directory |

## Detection Filters

A peak is considered "detected" only if **both** conditions are met:

1. **SNR filter**: `SNR ≥ snr-threshold` (default 5.0) — ensures the peak is statistically significant above background noise
2. **Relative intensity filter**: `Imax ≥ rel-intensity-threshold × max(Imax)` (default 1%) — eliminates weak noise peaks that happen to pass the SNR filter but are orders of magnitude weaker than real diffraction peaks

When a peak fails detection, the rejection reason is displayed:
- `SNR=2.1<5.0` — failed SNR filter
- `Imax=0.02%<1%` — failed relative intensity filter
- `NOT DET` — no peak found at all (Imax ≈ 0)

## Overlapping Rings

When rings from different phases have similar radii (ΔR < merge threshold), they are **merged** into a single fit position. After fitting:
- The fitted Rcen is used to back-calculate `a` for **each** contributing phase
- Both assignments are reported with their respective `Δa/a` values
- Overlapping peaks are flagged with ⚠️ in the output

| Proximity | Behavior |
|-----------|----------|
| ΔR > 2 × ROIPadding | Independent fits |
| ΔR < ROIPadding, > 2×FWHM | Joint fit, both resolved |
| ΔR < 2×FWHM, > merge threshold | Joint fit, flagged as "blended" |
| ΔR < merge threshold | Merged into single peak |

## Output

### Table A: Per-Ring Results

For each fitted ring:
- `Phase` + `(hkl)` — phase assignment and Miller indices
- `R_theory` / `R_fitted` — predicted vs. fitted radius (pixels)
- `Imax` — peak amplitude (integrated counts)
- `BG` — fitted background level
- `Sigma` — peak width (bins)
- `SNR` — signal-to-noise ratio
- `a_fitted` — back-calculated cubic lattice parameter (Å)
- `Δa/a(ppm)` — fractional deviation from nominal lattice parameter
- `Notes` — overlap warnings, rejection reasons

### Table B: Phase Summary

Per-phase aggregate statistics:
- `Detected` — count of detected/total rings
- `Coverage` — detection percentage
- `Mean/Std/Min/Max a(Å)` — lattice parameter statistics
- `Δa/a_nom(ppm)` — mean fractional deviation from nominal
- `Status`: ✅ PRESENT (≥30% of exclusive rings detected), ⚠️ MARGINAL (<30% but >0%), ❌ ABSENT

### Table C: Intensity Statistics

Per-phase intensity breakdown:
- `Sum/Mean/Max/Min Imax` — intensity distribution
- `Frac of Total` — fraction of total detected intensity belonging to each phase

## Lattice Parameter Back-Calculation

For cubic crystals, the lattice parameter is computed from the fitted peak center:

```
R_µm = R_fitted × pixel_size
2θ = atan(R_µm / Lsd)
d = λ / (2 · sin(θ))
a = d × √(h² + k² + l²)
```

The mean `a` across all detected rings provides the refined lattice parameter. The standard deviation indicates the consistency of the refinement. Values within ~100 ppm of nominal indicate good geometry calibration.

## CPU vs GPU Backend

| Feature | CPU (`-backend cpu`) | GPU (`-backend gpu`) |
|---------|---------------------|---------------------|
| Binary | `IntegratorZarrOMP` | `IntegratorFitPeaksGPUStream` |
| Orchestration | Direct: `ffGenerateZipRefactor → DetectorMapperZarr → IntegratorZarrOMP` | Via `integrator_batch_process.py` |
| Peak params | `peak_params.txt` file | `PeakLocation` lines in param file |
| Best for | Few images, no GPU | Streaming, large datasets |
| Output | `fit.bin` (same format) | `fit.bin` (same format) |

## Benchmark Test

```bash
# Run the phase identification benchmark
python utils/test_phase_id.py -nCPUs 4

# Via build.sh
./build.sh --test phaseid
```

The benchmark:
1. Runs CeO₂ + Au phases against the CeO₂ calibration data
2. Asserts CeO₂ is detected (≥30% coverage)
3. Asserts Au is absent (0 detections)
4. Validates CeO₂ lattice parameter within 500 ppm of 5.4116 Å

## Application: High-Pressure Compression Experiments

In dynamic compression or diamond anvil cell (DAC) experiments, the lattice parameter evolves under pressure. `phase_id.py` handles this naturally:

### How It Works

The tool predicts ring positions using the **nominal** (ambient) lattice parameter, then fits peaks wherever they actually appear within the ROI window. Even if the lattice parameter changes significantly under pressure, the **back-calculation** from the fitted Rcen automatically yields the compressed lattice parameter:

```
nominal a = 5.4116 Å → predicted R = 211.85 px
compressed a = 5.30 Å → actual R shifts to ~216 px
→ fitted Rcen = 216.03 px
→ back-calculated a = 5.2998 Å (correct!)
```

The key requirement is that the shifted peak must remain **within the ROI window** (`--roi-padding`). With the default `--roi-padding 30` (= 30 bins × 0.25 px/bin = 7.5 px), peaks can shift by up to ~7 px and still be captured.

### Recommended Settings for Large Compressions

| Compression | Expected ΔR | Recommended `--roi-padding` |
|-------------|------------|----------------------------|
| < 1% volume | < 3 px | 30 (default) |
| 1–5% volume | 3–15 px | 60–80 |
| 5–20% volume | 15–60 px | 100–200 |
| > 20% volume | > 60 px | 250+ or update nominal `a` |

> [!TIP]
> For very large compressions (>10% volume change), consider updating the nominal `a` in `phases.txt` to a mid-range value. For example, if you expect `a` to go from 5.41 to 5.00 Å, use `a = 5.20` as the nominal. This centers the ROI windows on the expected peak positions and reduces required padding.

### Multi-Image Time Series

For tracking lattice parameter evolution across a series of images (e.g., pressure steps, time-resolved), run `phase_id.py` on each image independently and collect the per-image fitted lattice parameters:

```bash
# Process a series of images
for i in $(seq 1 100); do
    python utils/phase_id.py \
        -paramFN geometry.txt \
        -dataFN "scan_$(printf '%06d' $i).tif" \
        -phases phases.txt \
        -nCPUs 4 \
        --roi-padding 100 \
        --work-dir output/frame_$i
done
```

The per-phase `Mean a(Å)` from each run gives `a(t)` or `a(P)`, which can be used to compute:
- **Equation of state**: `V(P) = a³(P)` for cubic
- **Strain evolution**: `ε = (a - a₀) / a₀`
- **Phase transitions**: sudden appearance/disappearance of phases

> [!NOTE]
> For streaming real-time experiments at the beamline, use `-backend gpu` with `integrator_batch_process.py` for much faster throughput.

## See Also

- [FF_Radial_Integration.md](FF_Radial_Integration.md) — Integration and peak fitting details
- [FF_Integrator_Benchmark.md](FF_Integrator_Benchmark.md) — Peak fitting benchmark
- [FF_Calibration.md](FF_Calibration.md) — Geometry calibration prerequisite
