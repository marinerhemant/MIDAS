# MIDAS: High-Energy Diffraction Microscopy — Overview & Quick Start

**Version:** 9.0  
**Contact:** hsharma@anl.gov

---

## 1. Introduction

**MIDAS** (Microstructure Diffraction Analysis Software) is a suite of tools for reconstructing three-dimensional microstructures from High-Energy Diffraction Microscopy (HEDM) data. Developed at the Advanced Photon Source (APS) at Argonne National Laboratory, MIDAS supports the complete data-reduction pipeline — from raw detector frames to grain maps, strain tensors, and spatially resolved orientation fields.

HEDM exploits the high penetration depth of synchrotron X-rays to non-destructively characterize the internal grain structure of polycrystalline materials. Unlike surface-sensitive techniques such as EBSD, HEDM probes the full bulk of a specimen while it remains intact, making it ideal for *in situ* mechanical loading, thermal cycling, and other experiments where preserving the sample is essential.

This manual provides a bird's-eye view of the MIDAS workflow and points you to the detailed sub-manuals for each step.

---

## 2. HEDM Techniques Supported by MIDAS

MIDAS supports three complementary diffraction geometries. Each probes a different length scale and provides different information about the microstructure.

```mermaid
graph LR
    subgraph "HEDM Geometries"
        FF["Far-Field (FF-HEDM)<br/>Detector ≈ 1 m"]
        NF["Near-Field (NF-HEDM)<br/>Detector ≈ 5–10 mm"]
        PF["Pencil-Beam (PF-HEDM)<br/>Focused beam"]
    end

    FF --> G1["Grain centroids<br/>Average orientations<br/>Elastic strain tensors"]
    NF --> G2["Spatially resolved<br/>orientation maps<br/>Grain morphology"]
    PF --> G3["High-resolution<br/>grain orientations<br/>from pencil beam"]
```

### 2.1. Far-Field HEDM (FF-HEDM)

A large-area detector is placed approximately 1 meter downstream of the sample. As the sample rotates, each grain produces a set of discrete diffraction spots that trace out arcs on the detector. By associating spots across many rotation angles, MIDAS determines:

- The **center-of-mass position** of each grain.
- The **average crystallographic orientation** (as an orientation matrix).
- The **full elastic strain tensor** of each grain (6 independent components).

FF-HEDM is well-suited for specimens containing tens to thousands of grains.

### 2.2. Near-Field HEDM (NF-HEDM)

The detector sits only a few millimeters from the sample. At this distance, individual diffraction spots from neighboring grains are spatially separated and can be mapped back to specific voxels in the sample. MIDAS reconstructs a 3D grid where every voxel is assigned a crystallographic orientation and a confidence metric. This produces:

- **3D grain-shape maps** with sub-micrometer spatial resolution.
- **Intragranular orientation gradients** (misorientation within a grain).

NF-HEDM typically uses orientation seeds from a prior FF-HEDM analysis to reduce the search space and speed up reconstruction.

### 2.3. Pencil-Beam FF-HEDM (PF-HEDM)

A variant of FF-HEDM that uses a focused (pencil) beam instead of a box beam. Because the illuminated volume is much smaller, the number of overlapping spots is reduced, which improves indexing for fine-grained samples or high-symmetry materials.

---

## 3. End-to-End Workflow

A typical MIDAS analysis proceeds through the stages shown below. Not every stage is required for every experiment — for example, WAXS caking is independent of the grain-reconstruction pipeline.

```mermaid
flowchart TD
    RAW["Raw Detector Data<br/>(HDF5 / GE / TIFF)"] --> CAL

    subgraph "Calibration"
        CAL["FF Calibration<br/>(powder reference)"]
        CAL --> NFCAL["NF Calibration<br/>(if doing NF-HEDM)"]
    end

    CAL --> INT["Radial Integration<br/>(caking)"]
    INT --> GSAS["GSAS-II / Rietveld<br/>Refinement"]

    CAL --> FF["FF-HEDM Analysis<br/>(indexing + fitting)"]
    FF --> VIZ["Interactive<br/>Visualization"]
    FF --> SEED["Orientation Seeds"]
    SEED --> NF

    NFCAL --> NF["NF-HEDM Analysis"]
    NF --> MIC["Microstructure<br/>(.mic file)"]

    CAL --> PF["PF-HEDM Analysis"]
    PF --> VIZ

    FF --> FWD["Forward Simulation<br/>(validation)"]
    NF --> FWD

    style RAW fill:#1a1a2e,stroke:#e94560,color:#fff
    style CAL fill:#16213e,stroke:#0f3460,color:#fff
    style NFCAL fill:#16213e,stroke:#0f3460,color:#fff
    style FF fill:#0f3460,stroke:#e94560,color:#fff
    style NF fill:#0f3460,stroke:#e94560,color:#fff
    style PF fill:#0f3460,stroke:#e94560,color:#fff
    style INT fill:#533483,stroke:#e94560,color:#fff
    style GSAS fill:#533483,stroke:#e94560,color:#fff
    style VIZ fill:#2b2d42,stroke:#8d99ae,color:#fff
    style MIC fill:#2b2d42,stroke:#8d99ae,color:#fff
    style FWD fill:#2b2d42,stroke:#8d99ae,color:#fff
    style SEED fill:#2b2d42,stroke:#8d99ae,color:#fff
```

---

## 4. Coordinate Systems

Understanding the coordinate system is critical for correctly loading data into MIDAS and interpreting results.

### 4.1. Laboratory Frame

MIDAS uses the **ESRF convention** for the laboratory coordinate system:

| Axis | Direction |
|------|-----------|
| **Z_L** | X-ray beam propagation direction (downstream) |
| **Y'_L** | Tomographic rotation axis |
| **X_L** | Plane normal of the (Z_L, Y'_L) plane |
| **Y_L** | Z_L × X_L |

The angle between **Y_L** and **Y'_L** is the **wedge angle** (Ω). For standard HEDM setups, the wedge angle is zero — the rotation axis is perpendicular to the beam.

> [!NOTE]
> The right-handed rotation about Y'_L is denoted **ω** (omega). A full ω-scan typically covers 360° (or ±180°) to capture all accessible diffraction conditions.

### 4.2. Detector Frame

The detector plane is described by three tilt angles relative to the laboratory frame and a beam-center position (BC). The tilt about Z_L (called **tx**) requires Friedel pairs from a single crystal and cannot be determined from powder rings alone.

| Azimuth η | Detector Axis |
|-----------|---------------|
| 0° | +Y_D |
| +90° | −X_D |
| −90° | +X_D |
| ±180° | −Y_D |

### 4.3. Image Transformation

When loading raw detector images, MIDAS may need to apply a transformation to make the image consistent with the laboratory coordinate system. Transformations are specified via the `ImTransOpt` parameter:

| `ImTransOpt` | Transformation |
|--------------|----------------|
| 0 | No change |
| 1 | Flip left/right |
| 2 | Flip top/bottom |
| 3 | Transpose |

> [!IMPORTANT]
> Getting `ImTransOpt` right is essential. If the image orientation does not match the coordinate system, calibration will fail silently and all downstream results will be incorrect. Verify using a physical feature on the detector (e.g., a beam-stop wire, a fiducial marker) whose position you can track.

### 4.4. Sample Frame

The sample coordinate system is an intermediate frame **unknown** to the diffraction geometry. MIDAS treats crystals in the sample purely as scatterers that produce spots as the sample rotates through diffraction conditions. It is the user's responsibility to:

1. Track the sample orientation relative to the beam during mounting.
2. Use fiducial markers (e.g., Au dots, notches, tomography) to relate the lab frame to the sample frame for post-analysis interpretation.

---

## 5. Detector Setup at APS 1-ID

### 5.1. GE Amorphous-Si Detector

The single-panel GE amorphous-Si detector has a pixel pitch of **200 µm** and a coverage of **2048 × 2048** pixels. With this detector, `ImTransOpt` is typically **0** (no transformation needed).

### 5.2. Varex 4343 Detector

The Varex 4343 detector has a pixel pitch of **150 µm** and covers **2880 × 2880** pixels. When mounted at APS 1-ID-C with the standard AreaDetector plugin configuration (Rot90 transform in Trans1), `ImTransOpt` should be set to **2** (vertical flip).

> [!TIP]
> Always verify the detector orientation by checking that the image, when loaded in a viewer (ImageJ, VSCode H5Web, Fit2D), matches the expected physical layout: **outboard** → +X_L and **up** → +Y_L. Place physical markers on the detector (a blue Allen key, tape on the pin diode cable) and confirm their projection appears in the correct quadrant.

---

## 6. Sample Alignment and Tomographic Motion

### 6.1. FF-HEDM Alignment

While placing the sample precisely on the rotation axis is not strictly required for FF-HEDM, centering the sample as closely as possible is good practice to minimize artifacts.

### 6.2. WAXS Alignment

For Wide-Angle X-ray Scattering (WAXS) / caking, the sample must be placed precisely on the rotation axis (or at a well-known position relative to the detector and beam). The standard sample-alignment procedure is covered during beam time at APS 1-ID.

### 6.3. Rotation Handedness

MIDAS assumes a **right-handed** rotation about the ω axis. Some rotation stages (e.g., Aerotech) perform a **left-handed** rotation. If your stage uses left-handed rotation, the omega angles must be negated before passing them to MIDAS.

| Stage | Rotation | Conversion Needed? |
|-------|----------|-------------------|
| Aerotech | Left-handed | Yes — negate ω |
| MTSphi | Right-handed | No |
| RAMS3 | Right-handed | No |

### 6.4. Shadowing

If a custom loading frame or environmental cell causes shadowing at certain ω angles, those angular ranges must be excluded using the `OmegaRange` and `BoxSize` parameters in the MIDAS parameter file. Multiple disjoint omega ranges can be specified.

---

## 7. Calibration Parameter Reference

The FF-HEDM instrument is characterized by the following parameters, typically determined during calibration:

| Parameter | Unit | Description |
|-----------|------|-------------|
| `Lsd` | µm | Sample-to-detector distance |
| `BC` | pixels | Beam center (horizontal, vertical) |
| `ty` | degrees | Detector tilt about Y |
| `tz` | degrees | Detector tilt about Z |
| `tx` | degrees | Detector tilt about beam axis (requires single-crystal calibration) |
| `p0`, `p1`, `p2` | — | Spatial distortion polynomial coefficients |
| `p3` | — | Spatial distortion parameter |
| `Wavelength` | Å | Incident X-ray wavelength |
| `px` | µm | Pixel size |

> [!NOTE]
> The `tx` parameter cannot be determined from powder rings alone. It requires Friedel pairs from a single crystal (typically Au). If you are only performing WAXS caking, `tx = 0` is usually sufficient.

### 7.1. Cross-Software Parameter Conversion

If you obtain initial calibration guesses from other software, the following table provides first-order translations:

| Software | BC 1st term (px) | BC 2nd term (px) | Lsd (mm) |
|----------|-------------------|-------------------|----------|
| **MIDAS** | horizontal | vertical | in µm ÷ 1000 |
| **GSAS-II** | horizontal (µm → px) | vertical (µm → px) | direct |
| **HEXRD** | check coordinate convention | may differ | direct |
| **Fit2D** | horizontal | vertical | direct |

> [!CAUTION]
> Each software package has its own coordinate conventions. Always verify the physical meaning of the beam center and detector tilt parameters against the detector coordinate system before transferring values between packages.

---

## 8. Manual Index

The table below lists every MIDAS sub-manual and what it covers. Start with the manual most relevant to your current analysis step.

| Manual | Topic | When to Use |
|--------|-------|-------------|
| [FF_calibration.md](FF_calibration.md) | FF-HEDM geometry calibration | Setting up a new experiment; determining detector parameters |
| [FF_Analysis.md](FF_Analysis.md) | FF-HEDM grain indexing and fitting | Extracting grain orientations, positions, and strain tensors |
| [FF_RadialIntegration.md](FF_RadialIntegration.md) | Radial integration / caking | Converting 2D detector images to 1D intensity vs. 2θ profiles |
| [FF_Interactive_Plotting.md](FF_Interactive_Plotting.md) | Interactive FF-HEDM visualization | Exploring grains, spots, and raw data interactively in a browser |
| [FF_dual_datasets.md](FF_dual_datasets.md) | Dual-dataset FF-HEDM analysis | Combining two FF datasets (e.g., different load states) |
| [PF_Analysis.md](PF_Analysis.md) | Pencil-beam FF-HEDM analysis | Analyzing data from a focused beam geometry |
| [NF_calibration.md](NF_calibration.md) | NF-HEDM detector calibration | Calibrating NF detector distances and beam center |
| [NF_Analysis.md](NF_Analysis.md) | NF-HEDM reconstruction workflow | Reconstructing spatially resolved orientation maps |
| [NF_MultiResolution_Analysis.md](NF_MultiResolution_Analysis.md) | Multi-resolution NF-HEDM | Iterative NF reconstruction at increasing grid resolution |
| [NF_gui.md](NF_gui.md) | NF-HEDM interactive GUI | Calibrating detectors, visualizing NF data, and inspecting results |
| [ForwardSimulationManual.md](ForwardSimulationManual.md) | Forward simulation (`simulateNF`) | Validating reconstructions by simulating expected diffraction patterns |
| [GSAS-II_Integration.md](GSAS-II_Integration.md) | Importing MIDAS output into GSAS-II | Rietveld refinement of caked powder data |
| [Tomography_Reconstruction.md](Tomography_Reconstruction.md) | Absorption-contrast CT reconstruction | Reconstructing tomographic slices from projection data |

---

## 9. Getting Started Checklist

For a first-time user performing a combined FF + NF experiment, follow this sequence:

1. **Calibrate** the FF-HEDM detector using a powder calibrant (CeO₂ or LaB₆) → [FF_calibration.md](FF_calibration.md)
2. **Integrate / cake** the powder data (if WAXS analysis is needed) → [FF_RadialIntegration.md](FF_RadialIntegration.md)
3. **(Optional)** Import caked data into GSAS-II for Rietveld refinement → [GSAS-II_Integration.md](GSAS-II_Integration.md)
4. **Run FF-HEDM** analysis on the specimen → [FF_Analysis.md](FF_Analysis.md)
5. **Visualize** FF results interactively → [FF_Interactive_Plotting.md](FF_Interactive_Plotting.md)
6. **Calibrate** the NF-HEDM detectors → [NF_calibration.md](NF_calibration.md)
7. **Run NF-HEDM** reconstruction using FF orientation seeds → [NF_Analysis.md](NF_Analysis.md) or [NF_MultiResolution_Analysis.md](NF_MultiResolution_Analysis.md)
8. **Validate** results with forward simulation → [ForwardSimulationManual.md](ForwardSimulationManual.md)

---

## 10. See Also

- [FF_calibration.md](FF_calibration.md) — FF-HEDM geometry calibration
- [FF_Analysis.md](FF_Analysis.md) — FF-HEDM grain indexing and strain fitting
- [FF_RadialIntegration.md](FF_RadialIntegration.md) — Radial integration / caking
- [FF_Interactive_Plotting.md](FF_Interactive_Plotting.md) — Interactive FF-HEDM visualization
- [FF_dual_datasets.md](FF_dual_datasets.md) — Dual-dataset FF-HEDM analysis
- [PF_Analysis.md](PF_Analysis.md) — Pencil-beam FF-HEDM analysis
- [NF_calibration.md](NF_calibration.md) — NF detector geometry calibration
- [NF_Analysis.md](NF_Analysis.md) — NF-HEDM reconstruction workflow
- [NF_MultiResolution_Analysis.md](NF_MultiResolution_Analysis.md) — Multi-resolution NF-HEDM
- [NF_gui.md](NF_gui.md) — Interactive NF-HEDM analysis GUI
- [ForwardSimulationManual.md](ForwardSimulationManual.md) — Forward simulation (simulateNF)
- [GSAS-II_Integration.md](GSAS-II_Integration.md) — Importing MIDAS output into GSAS-II
- [Tomography_Reconstruction.md](Tomography_Reconstruction.md) — Absorption-contrast CT reconstruction

---

If you encounter any issues or have questions, please open an issue on this repository.
