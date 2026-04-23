"""The MIDAS parameter registry.

Single source of truth for every key recognized by the FF/NF/PF/RI
pipelines. Walked by both the validator (to check user input) and the
wizard (to prompt for values).

Defaults are sourced from:
  - FF_HEDM/src/MIDAS_ParamParser.c (midas_config_defaults)
  - Per-executable inline defaults documented in manuals/FF_Parameters_Reference.md
    and manuals/NF_Parameters_Reference.md

When adding entries:
  - `default` is what the parser sets if the key is absent (from source).
  - `typical` is what Example files recommend when different from parser default.
  - `hidden_in_wizard=True` for calibration-only / forward-sim-only keys.
"""

from __future__ import annotations

from .schema import ParamSpec, ParamType, Path, Stage

# Path sets (readability)
FF = Path.FF
NF = Path.NF
PF = Path.PF
RI = Path.RI
ALL = frozenset({FF, NF, PF, RI})
FF_PF = frozenset({FF, PF})
FF_NF_PF = frozenset({FF, NF, PF})

# FF/PF/RI — "everything except NF": pipelines that share DetectorMapper's
# distortion model (p0..p14, tolP*, DistortionFile). NF uses a direct
# pinhole+tilts inversion with no distortion polynomial.
FF_PF_RI = frozenset({FF, PF, RI})

# FNP is an alias for the "indexing / grain-analysis pipelines" — everything
# except radial integration. Used for indexing/sample-geometry keys that
# don't apply to RI.
FNP = FF_NF_PF

# Stage sets (readability)
S_FILE = frozenset({Stage.FILE_DISCOVERY})
S_INDEX = frozenset({Stage.INDEXING})
S_REFINE = frozenset({Stage.REFINEMENT})
S_PEAK = frozenset({Stage.PEAK_SEARCH})
S_CALIB = frozenset({Stage.CALIBRATION})
S_SIM = frozenset({Stage.FORWARD_SIM})
S_IMG = frozenset({Stage.IMAGE_PREPROC})
S_INT = frozenset({Stage.INTEGRATION})


PARAMS: list[ParamSpec] = [

    # ═══════════════════════════════════════════════════════════════════════
    # 1. DATA SOURCE AND FILE DISCOVERY
    # ═══════════════════════════════════════════════════════════════════════

    ParamSpec(
        name="RawFolder", type=ParamType.PATH, category="Data source",
        description="Directory holding raw detector frames.",
        applies_to=FF_PF, required_for=FF_PF, stages=S_FILE,
        validators=("directory_exists",),
        notes="Both FF and PF call ffGenerateZipRefactor on raw data by default "
              "(convertFiles=1). The zip-only mode (-convertFiles 0 with a "
              "pre-built .MIDAS.zip, or -dataFN with an HDF5 container) is a "
              "re-run / simulation optimization and doesn't need this — but the "
              "primary workflow always does.",
    ),
    ParamSpec(
        name="DataDirectory", type=ParamType.PATH, category="Data source",
        description="Directory holding raw frames and mmap'd binary files (NF).",
        applies_to=frozenset({NF}), required_for=frozenset({NF}),
        stages=S_FILE | S_INDEX,
        validators=("directory_exists",),
        notes="NF equivalent of RawFolder. Also the mmap target for SpotsInfo.bin, "
              "DiffractionSpots.bin, Key.bin, OrientMat.bin.",
    ),
    ParamSpec(
        name="Folder", type=ParamType.PATH, category="Data source",
        description="Generic folder; some legacy executables look here.",
        applies_to=ALL, stages=S_FILE,
    ),
    ParamSpec(
        name="FileStem", type=ParamType.STR, category="Data source",
        description="Filename prefix before the zero-padded frame number.",
        applies_to=FF_PF, required_for=FF_PF, stages=S_FILE,
    ),
    ParamSpec(
        name="Ext", type=ParamType.STR, category="Data source",
        description="File extension with leading dot (.ge3, .tif, .h5).",
        applies_to=FF_PF, required_for=FF_PF, stages=S_FILE,
    ),
    ParamSpec(
        name="Padding", type=ParamType.INT, category="Data source",
        description="Zero-padding width for file numbers.",
        applies_to=FF_PF, default=6, units="digits", stages=S_FILE,
        validators=("positive",),
    ),
    ParamSpec(
        name="StartNr", type=ParamType.INT, category="Data source",
        description="First frame number in sequence.",
        applies_to=ALL, required_for=ALL, stages=S_FILE,
        notes="Frame index (one-indexed for GE/TIF where frame=file, or "
              "HDF5/Zarr slab start for multi-frame containers).",
    ),
    ParamSpec(
        name="EndNr", type=ParamType.INT, category="Data source",
        description="Last frame number in sequence.",
        applies_to=ALL, required_for=ALL, stages=S_FILE,
    ),
    ParamSpec(
        name="StartFileNrFirstLayer", type=ParamType.INT, category="Data source",
        description="Starting file number for layer 1 (multi-layer runs).",
        applies_to=FF_PF, default=1, stages=S_FILE,
    ),
    ParamSpec(
        name="NrFilesPerSweep", type=ParamType.INT, category="Data source",
        description="Files per ω sweep (multi-wedge scans).",
        applies_to=FF_PF, default=1, units="count", stages=S_FILE,
        validators=("positive",),
    ),
    ParamSpec(
        name="NrFilesPerDistance", type=ParamType.INT, category="Data source",
        description="Frames per NF detector distance.",
        applies_to=frozenset({NF}), required_for=frozenset({NF}),
        default=1, units="count", stages=S_FILE,
        validators=("positive",),
    ),
    ParamSpec(
        name="Dark", type=ParamType.PATH, category="Data source",
        description="Dark/background frame file.",
        applies_to=FF_PF, stages=S_PEAK,
        validators=("file_exists",),
    ),
    ParamSpec(
        name="SkipFrame", type=ParamType.INT, category="Data source",
        description="Frames to skip at start of each sweep.",
        applies_to=ALL, default=0, units="count", stages=S_FILE,
        validators=("non_negative",),
    ),
    ParamSpec(
        name="HeadSize", type=ParamType.INT, category="Data source",
        description="Header size to skip in raw binary files.",
        applies_to=ALL, default=0, units="bytes", stages=S_FILE,
        notes="Auto-set to 8192 when DataType=1 (GE format).",
    ),
    ParamSpec(
        name="DataType", type=ParamType.INT, category="Data source",
        description="Raw frame data type code (1 = GE uint16, 2 = Pilatus, etc.).",
        applies_to=ALL, default=1, stages=S_FILE,
    ),
    ParamSpec(
        name="ScanStep", type=ParamType.FLOAT, category="Data source",
        description="Translation step between scan points (multi-scan).",
        applies_to=FF_PF, units="um", stages=S_FILE,
        notes="Primarily PF (where nScans > 1). FF with nScans > 1 also reads it.",
    ),
    ParamSpec(
        name="RawStartNr", type=ParamType.INT, category="Data source",
        description="Starting raw frame number (NF image preprocessing).",
        applies_to=frozenset({NF}), required_for=frozenset({NF}), stages=S_IMG,
    ),
    ParamSpec(
        name="OrigFileName", type=ParamType.STR, category="Data source",
        description="Input filename stem for NF image preprocessing.",
        applies_to=frozenset({NF}), required_for=frozenset({NF}), stages=S_IMG,
    ),
    ParamSpec(
        name="ReducedFileName", type=ParamType.STR, category="Data source",
        description="Output stem for median-subtracted/processed NF frames.",
        applies_to=frozenset({NF}), required_for=frozenset({NF}), stages=S_IMG | S_INDEX,
    ),
    ParamSpec(
        name="extOrig", type=ParamType.STR, category="Data source",
        description="Input file extension for NF preprocessing (no dot).",
        applies_to=frozenset({NF}), default="tif", stages=S_IMG,
    ),
    ParamSpec(
        name="extReduced", type=ParamType.STR, category="Data source",
        description="Reduced output extension (no dot).",
        applies_to=frozenset({NF}), default="bin", stages=S_IMG,
    ),
    ParamSpec(
        name="WFImages", type=ParamType.INT, category="Data source",
        description="Number of wide-field calibration images per layer (skipped).",
        applies_to=frozenset({NF}), default=0, units="count", stages=S_IMG,
        validators=("non_negative",),
    ),
    ParamSpec(
        name="OutputDirectory", type=ParamType.PATH, category="Data source",
        description="Where NF outputs are written (defaults to DataDirectory).",
        applies_to=frozenset({NF}), stages=S_FILE,
    ),
    ParamSpec(
        name="resultFolder", type=ParamType.PATH, category="Data source",
        description="Top-level output directory for the workflow driver.",
        applies_to=ALL, stages=S_FILE,
    ),
    ParamSpec(
        name="logDir", type=ParamType.PATH, category="Data source",
        description="Per-layer log directory (defaults to <resultFolder>/logs).",
        applies_to=frozenset({NF}), stages=S_FILE,
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # 2. HDF5 / ZARR DATASET NAMES
    # ═══════════════════════════════════════════════════════════════════════

    ParamSpec(
        name="darkDataset", type=ParamType.STR, category="Data source",
        description="HDF5 path for dark frames.",
        applies_to=FF_PF, default="exchange/dark",
        aliases=("darkLoc",), stages=S_FILE,
    ),
    ParamSpec(
        name="dataDataset", type=ParamType.STR, category="Data source",
        description="HDF5 path for data frames.",
        applies_to=FF_PF, default="exchange/data",
        aliases=("dataLoc",), stages=S_FILE,
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # 3. DETECTOR GEOMETRY
    # ═══════════════════════════════════════════════════════════════════════

    ParamSpec(
        name="NrPixels", type=ParamType.INT, category="Detector geometry",
        description="Square detector size (shortcut: sets both NrPixelsY and NrPixelsZ).",
        applies_to=ALL, units="pixels", stages=S_INDEX,
        validators=("positive",),
        notes="One of NrPixels OR (NrPixelsY AND NrPixelsZ) is required. "
              "The MIDAS_ParamParser.c NrPixels handler copies its value "
              "into both NrPixelsY and NrPixelsZ, so setting NrPixels is "
              "equivalent to setting them both to the same value. The wizard "
              "auto-derives NrPixels from max(Y, Z) when Y and Z are set.",
    ),
    ParamSpec(
        name="NrPixelsY", type=ParamType.INT, category="Detector geometry",
        description="Horizontal detector pixel count.",
        applies_to=ALL, units="pixels", stages=S_INDEX,
        validators=("positive",),
        notes="Satisfies the NrPixels requirement together with NrPixelsZ. "
              "Set these two if your detector is not square.",
    ),
    ParamSpec(
        name="NrPixelsZ", type=ParamType.INT, category="Detector geometry",
        description="Vertical detector pixel count.",
        applies_to=ALL, units="pixels", stages=S_INDEX,
        validators=("positive",),
        notes="Satisfies the NrPixels requirement together with NrPixelsY.",
    ),
    ParamSpec(
        name="px", type=ParamType.FLOAT, category="Detector geometry",
        description="Pixel size (square).",
        applies_to=ALL, required_for=ALL, units="um", stages=S_INDEX | S_INT,
        aliases=("PixelSize",),
        zarr_rename="PixelSize",
        validators=("positive",),
        typical=200,
        notes="FF typical ~200 µm, NF typical 1–2 µm. Accidentally copying an FF "
              "value into an NF config silently breaks ring radii. "
              "IntegratorZarrOMP accepts `PixelSize` as an alias.",
    ),
    ParamSpec(
        name="YPixelSize", type=ParamType.FLOAT, category="Detector geometry",
        description="Y-direction pixel size override (non-square detectors).",
        applies_to=FF_PF_RI, units="um", stages=S_INDEX | S_INT, hidden_in_wizard=True,
        validators=("positive",),
        notes="When set, overrides `px` for the Y axis. DetectorMapper and "
              "IntegratorZarrOMP read this for detectors with different "
              "horizontal and vertical pixel dimensions.",
    ),
    ParamSpec(
        name="Lsd", type=ParamType.FLOAT, category="Detector geometry",
        description="Sample-to-detector distance.",
        applies_to=ALL, required_for=ALL, units="um", stages=S_INDEX | S_CALIB,
        aliases=("Distance", "DetDist"), multi_entry=True,
        validators=("positive", "lsd_plausible"),
        notes="NF has multiple Lsd lines (one per detector distance); FF has a "
              "single Lsd line. IntegratorZarrOMP accepts `DetDist` as an alias.",
    ),
    ParamSpec(
        name="BC", type=ParamType.FLOAT_LIST, category="Detector geometry",
        description="Beam center position `Y Z` in pixel coordinates.",
        applies_to=ALL, required_for=ALL, units="pixels",
        stages=S_INDEX | S_CALIB,
        multi_entry=True, zarr_rename="YCen+ZCen",
        validators=("bc_in_detector",),
    ),
    ParamSpec(
        name="tx", type=ParamType.FLOAT, category="Detector geometry",
        description="Detector rotation about X-ray beam axis.",
        applies_to=ALL, default=0, units="deg", stages=S_INDEX,
    ),
    ParamSpec(
        name="ty", type=ParamType.FLOAT, category="Detector geometry",
        description="Detector rotation about horizontal axis.",
        applies_to=ALL, default=0, units="deg", stages=S_INDEX,
    ),
    ParamSpec(
        name="tz", type=ParamType.FLOAT, category="Detector geometry",
        description="Detector rotation about vertical axis.",
        applies_to=ALL, default=0, units="deg", stages=S_INDEX,
    ),
    ParamSpec(
        name="Wedge", type=ParamType.FLOAT, category="Detector geometry",
        description="Deviation from 90° between rotation axis and beam.",
        applies_to=ALL, default=0, units="deg", stages=S_INDEX,
    ),
    ParamSpec(
        name="RhoD", type=ParamType.FLOAT, category="Detector geometry",
        description="Maximum ring radius for distortion model.",
        applies_to=ALL, units="um", stages=S_INDEX,
        aliases=("MaxRingRad",),
        validators=("positive",),
    ),
    ParamSpec(
        name="Parallax", type=ParamType.FLOAT, category="Detector geometry",
        description="Parallax correction term.",
        applies_to=ALL, default=0, stages=S_REFINE,
    ),
    ParamSpec(
        name="ResidualCorrectionMap", type=ParamType.PATH, category="Detector geometry",
        description="Residual distortion correction map file.",
        applies_to=ALL, stages=S_CALIB,
        validators=("file_exists",),
    ),
    ParamSpec(
        name="nDistances", type=ParamType.INT, category="Detector geometry",
        description="Number of detector distances in a NF scan.",
        applies_to=frozenset({NF}), required_for=frozenset({NF}),
        stages=S_INDEX, typical=2,
        validators=("positive",),
        notes="Multi-entry keys (Lsd, BC, OmegaRange, BoxSize) must each appear "
              "exactly nDistances times.",
    ),

    # ─── Detector distortion coefficients ──────────────────────────────────
    # Physical model (DetectorMapper.c / FitSetupParamsAllZarr.c):
    #
    #   Rcorrected = Rmeasured × (1 + DistortFunc(Rnorm, η))
    #
    #   DistortFunc = p0·Rnorm^n0·cos(2(90-η) + p6)           [2-fold amplitude]
    #               + p1·Rnorm^n1·cos(4(90-η) + p3)           [4-fold amplitude]
    #               + p2·Rnorm² + p5·Rnorm⁴ + p4·Rnorm⁶       [isotropic radial]
    #               + p7·Rnorm⁴·cos((90-η) + p8)              [dipole 2-fold]
    #               + p9·Rnorm³·cos(3(90-η) + p10)            [trefoil 3-fold]
    #               + p11·Rnorm⁵·cos(5(90-η) + p12)           [pentafoil 5-fold]
    #               + p13·Rnorm⁶·cos(6(90-η) + p14)           [hexafoil 6-fold]
    #
    # NF does NOT use this distortion model — NF's geometry is a direct
    # pinhole+tilts inversion. Keep scope to FF_PF_RI.
    ParamSpec(
        name="p0", type=ParamType.FLOAT, category="Detector distortion",
        description="2-fold azimuthal amplitude (paired with phase p6).",
        applies_to=FF_PF_RI, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="p1", type=ParamType.FLOAT, category="Detector distortion",
        description="4-fold azimuthal amplitude (paired with phase p3).",
        applies_to=FF_PF_RI, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="p2", type=ParamType.FLOAT, category="Detector distortion",
        description="Isotropic radial distortion, R² term.",
        applies_to=FF_PF_RI, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="p3", type=ParamType.FLOAT, category="Detector distortion",
        description="4-fold azimuthal phase angle (pairs with p1).",
        applies_to=FF_PF_RI, default=0, units="deg", stages=S_CALIB, hidden_in_wizard=True,
        notes="Periodicity constant; tolerance tolP3 defaults to 45° and "
              "this value is typically not user-refined.",
    ),
    ParamSpec(
        name="p4", type=ParamType.FLOAT, category="Detector distortion",
        description="Isotropic radial distortion, R⁶ term.",
        applies_to=FF_PF_RI, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="p5", type=ParamType.FLOAT, category="Detector distortion",
        description="Isotropic radial distortion, R⁴ term.",
        applies_to=FF_PF_RI, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="p6", type=ParamType.FLOAT, category="Detector distortion",
        description="2-fold azimuthal phase angle (pairs with p0).",
        applies_to=FF_PF_RI, default=0, units="deg", stages=S_CALIB, hidden_in_wizard=True,
        notes="Periodicity constant; tolerance tolP6 defaults to 90° and "
              "this value is typically not user-refined.",
    ),
    ParamSpec(
        name="p7", type=ParamType.FLOAT, category="Detector distortion",
        description="Dipole amplitude, R⁴ azimuthal (pairs with phase p8).",
        applies_to=FF_PF_RI, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="p8", type=ParamType.FLOAT, category="Detector distortion",
        description="Dipole phase angle (pairs with p7).",
        applies_to=FF_PF_RI, default=0, units="deg", stages=S_CALIB, hidden_in_wizard=True,
        notes="Periodicity constant; tolerance tolP8 defaults to 180°.",
    ),
    ParamSpec(
        name="p9", type=ParamType.FLOAT, category="Detector distortion",
        description="Trefoil (3-fold) amplitude, R³ azimuthal (pairs with phase p10).",
        applies_to=FF_PF_RI, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="p10", type=ParamType.FLOAT, category="Detector distortion",
        description="Trefoil phase angle (pairs with p9).",
        applies_to=FF_PF_RI, default=0, units="deg", stages=S_CALIB, hidden_in_wizard=True,
        notes="Periodicity constant; tolerance tolP10 defaults to 180°.",
    ),
    ParamSpec(
        name="p11", type=ParamType.FLOAT, category="Detector distortion",
        description="Pentafoil (5-fold) amplitude, R⁵ azimuthal (pairs with phase p12).",
        applies_to=FF_PF_RI, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="p12", type=ParamType.FLOAT, category="Detector distortion",
        description="Pentafoil phase angle (pairs with p11).",
        applies_to=FF_PF_RI, default=0, units="deg", stages=S_CALIB, hidden_in_wizard=True,
        notes="Periodicity constant; tolerance tolP12 defaults to 180°.",
    ),
    ParamSpec(
        name="p13", type=ParamType.FLOAT, category="Detector distortion",
        description="Hexafoil (6-fold) amplitude, R⁶ azimuthal (pairs with phase p14).",
        applies_to=FF_PF_RI, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="p14", type=ParamType.FLOAT, category="Detector distortion",
        description="Hexafoil phase angle (pairs with p13).",
        applies_to=FF_PF_RI, default=0, units="deg", stages=S_CALIB, hidden_in_wizard=True,
        notes="Periodicity constant; tolerance tolP14 defaults to 180°.",
    ),
    ParamSpec(
        name="DistortionFile", type=ParamType.PATH, category="Detector distortion",
        description="Binary distortion map (double-precision Y then Z shifts).",
        applies_to=FF_PF_RI, stages=S_CALIB, hidden_in_wizard=True,
        validators=("file_exists",),
        notes="Alternative to analytical p0..p14 polynomial: pixel-by-pixel "
              "shift table read by DetectorMapper.",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # 4. CRYSTALLOGRAPHY
    # ═══════════════════════════════════════════════════════════════════════

    ParamSpec(
        name="LatticeConstant", type=ParamType.FLOAT_LIST, category="Crystallography",
        description="Lattice constants `a b c α β γ`.",
        applies_to=ALL, required_for=ALL, units="Å / deg", stages=S_INDEX,
        aliases=("LatticeParameter",), zarr_rename="LatticeParameter",
        validators=("positive",),
    ),
    ParamSpec(
        name="SpaceGroup", type=ParamType.INT, category="Crystallography",
        description="Space group number.",
        applies_to=ALL, required_for=ALL, default=225, stages=S_INDEX,
        aliases=("SGNr",),
        validators=("space_group_range", "space_group_default_smell"),
    ),
    ParamSpec(
        name="Wavelength", type=ParamType.FLOAT, category="Crystallography",
        description="X-ray wavelength.",
        applies_to=ALL, required_for=ALL, units="Å", stages=S_INDEX,
        validators=("positive", "wavelength_plausible"),
    ),
    ParamSpec(
        name="NumPhases", type=ParamType.INT, category="Crystallography",
        description="Number of phases in sample.",
        applies_to=FNP, default=1, units="count", stages=S_INDEX,
        validators=("positive",),
    ),
    ParamSpec(
        name="PhaseNr", type=ParamType.INT, category="Crystallography",
        description="Phase number being analyzed.",
        applies_to=FNP, default=1, stages=S_INDEX,
        validators=("positive",),
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # 5. OMEGA SCAN
    # ═══════════════════════════════════════════════════════════════════════

    ParamSpec(
        name="OmegaStart", type=ParamType.FLOAT, category="Omega scan",
        description="First frame omega angle.",
        applies_to=ALL, required_for=FF_NF_PF, units="deg",
        stages=S_INDEX | S_INT,
        notes="FF/NF/PF use this as the scan-start angle (required). RI "
              "consumes it as frame metadata for per-lineout ω stamps — "
              "optional on the RI path. In RI, IntegratorZarrOMP.c also "
              "accepts `OmegaFirstFile` (see its own registry entry) as an "
              "alias, sharing semantics with PF's per-scan first-frame ω.",
    ),
    ParamSpec(
        name="OmegaEnd", type=ParamType.FLOAT, category="Omega scan",
        description="Last frame omega angle (optional — derivable).",
        applies_to=FF_PF, units="deg", stages=S_INDEX,
        notes="Optional. If omitted, derived as OmegaStart + OmegaStep × "
              "(EndNr − StartNr + 1). Specify only if your scan stops "
              "before reaching the computed end.",
    ),
    ParamSpec(
        name="OmegaStep", type=ParamType.FLOAT, category="Omega scan",
        description="Rotation step per frame (sign = direction).",
        applies_to=ALL, required_for=FF_NF_PF, units="deg",
        stages=S_INDEX | S_INT,
        zarr_rename="step",
        notes="FF/NF/PF: required scan step. RI: optional frame metadata for "
              "per-lineout ω stamps.",
    ),
    ParamSpec(
        name="OmegaSumFrames", type=ParamType.INT, category="Integration",
        description="Number of frames to chunk per integration output (RI).",
        applies_to=frozenset({RI}), default=1, units="count", stages=S_INT,
        validators=("positive",),
        notes="Maps to `chunkFiles` in IntegratorZarrOMP. Set >1 to sum "
              "consecutive frames into a single output lineout.",
    ),
    ParamSpec(
        name="OmegaRange", type=ParamType.FLOAT_LIST, category="Omega scan",
        description="Analysis ω window `ω_min ω_max`.",
        applies_to=FF_NF_PF, required_for=FF_NF_PF,
        units="deg", stages=S_INDEX,
        multi_entry=True, zarr_rename="OmegaRanges",
        validators=("omega_range_arity", "omega_range_ordered"),
        notes="At least one OmegaRange is required. May appear multiple times "
              "to define subset windows within the full scan range.",
    ),
    ParamSpec(
        name="BoxSize", type=ParamType.FLOAT_LIST, category="Omega scan",
        description="Virtual detector box `Ymin Ymax Zmin Zmax`.",
        applies_to=FF_NF_PF,
        default=[[-1e6, 1e6, -1e6, 1e6]],
        units="um", stages=S_INDEX,
        multi_entry=True, zarr_rename="BoxSizes",
        validators=("box_size_arity", "box_size_ordered"),
        notes="Optional. Default `-1e6 1e6 -1e6 1e6` disables clipping. "
              "Override only if you want to constrain which detector "
              "region is used for indexing.",
    ),
    ParamSpec(
        name="MinOmeSpotIDsToIndex", type=ParamType.FLOAT, category="Indexing",
        description="ω lower bound for spots used as indexing seeds.",
        applies_to=FF_PF, units="deg", stages=S_INDEX, typical=-90,
    ),
    ParamSpec(
        name="MaxOmeSpotIDsToIndex", type=ParamType.FLOAT, category="Indexing",
        description="ω upper bound for indexing seeds.",
        applies_to=FF_PF, units="deg", stages=S_INDEX, typical=90,
    ),
    ParamSpec(
        name="OmeBinSize", type=ParamType.FLOAT, category="Indexing",
        description="ω look-up-table bin size.",
        applies_to=FNP, default=0, units="deg", stages=S_INDEX, typical=0.1,
    ),
    ParamSpec(
        name="OmegaSigma", type=ParamType.FLOAT, category="Forward simulation",
        description="Simulated peak broadening in ω.",
        applies_to=FNP, default=0, units="deg", stages=S_SIM,
        hidden_in_wizard=True,
    ),
    ParamSpec(
        name="ExcludePoleAngle", type=ParamType.FLOAT, category="Indexing",
        description="Azimuthal exclusion near poles (NF-specific name).",
        applies_to=frozenset({NF}), default=6, units="deg", stages=S_INDEX,
        notes="FF uses 'MinEta' for the same concept. They are intentionally "
              "NOT aliased here: the pipelines have different defaults (FF=0, "
              "NF=6) and users should write the name that matches their path.",
    ),
    ParamSpec(
        name="MinEta", type=ParamType.FLOAT, category="Indexing",
        description="Azimuthal exclusion near poles (FF).",
        applies_to=FF_PF, default=0, units="deg", stages=S_INDEX, typical=6,
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # 6. RING SELECTION
    # ═══════════════════════════════════════════════════════════════════════

    ParamSpec(
        name="RingThresh", type=ParamType.INT_LIST, category="Ring selection",
        description="`<ring_nr> <intensity_threshold>` — ring + min intensity.",
        applies_to=frozenset({FF, PF, RI}), required_for=FF_PF,
        multi_entry=True, stages=S_PEAK | S_INDEX,
    ),
    ParamSpec(
        name="RingsToExclude", type=ParamType.INT, category="Ring selection",
        description="Ring numbers to exclude from analysis.",
        applies_to=ALL, multi_entry=True, stages=S_INDEX,
        aliases=("RingsToReject",),
    ),
    ParamSpec(
        name="RingNumbers", type=ParamType.INT, category="Ring selection",
        description="Explicit ring list (alternate form).",
        applies_to=ALL, multi_entry=True, stages=S_INDEX,
    ),
    ParamSpec(
        name="RingRadii", type=ParamType.FLOAT, category="Ring selection",
        description="Corresponding radii (must match RingNumbers count).",
        applies_to=ALL, multi_entry=True, units="Å", stages=S_INDEX,
    ),
    ParamSpec(
        name="MaxRingNumber", type=ParamType.INT, category="Ring selection",
        description="Upper cap on ring numbers to consider.",
        applies_to=ALL, default=0, stages=S_INDEX,
        validators=("non_negative",),
    ),
    ParamSpec(
        name="OverAllRingToIndex", type=ParamType.INT, category="Indexing",
        description="Ring used to generate candidate orientations.",
        applies_to=FF_PF, required_for=FF_PF, stages=S_INDEX,
        zarr_rename="OverallRingToIndex", typical=2,
        validators=("positive",),
        notes="Zarr dataset is 'OverallRingToIndex' with double L (spelling change).",
    ),
    ParamSpec(
        name="RingsToUse", type=ParamType.INT, category="Ring selection",
        description="Explicit rings for NF simulation / forward-sim.",
        applies_to=frozenset({NF}), multi_entry=True, stages=S_INDEX | S_SIM,
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # 7. IMAGE TRANSFORMS AND MASKING
    # ═══════════════════════════════════════════════════════════════════════

    ParamSpec(
        name="ImTransOpt", type=ParamType.INT, category="Image processing",
        description="Image transform code (0 = none; see manual).",
        applies_to=ALL, multi_entry=True, stages=S_PEAK,
    ),
    ParamSpec(
        name="MaskFile", type=ParamType.PATH, category="Image processing",
        description="Binary mask (1 = data, 0 = masked).",
        applies_to=ALL, stages=S_PEAK,
        validators=("file_exists",),
    ),
    ParamSpec(
        name="GapFile", type=ParamType.PATH, category="Image processing",
        description="Gap / dead-zone mask file.",
        applies_to=ALL, stages=S_PEAK,
        validators=("file_exists",),
    ),
    ParamSpec(
        name="BadPxFile", type=ParamType.PATH, category="Image processing",
        description="Bad-pixel mask file.",
        applies_to=ALL, stages=S_PEAK,
        validators=("file_exists",),
    ),
    ParamSpec(
        name="GapIntensity", type=ParamType.INT, category="Image processing",
        description="Fill value for gap pixels.",
        applies_to=ALL, default=0, units="counts", stages=S_PEAK,
    ),
    ParamSpec(
        name="BadPxIntensity", type=ParamType.INT, category="Image processing",
        description="Fill value for bad pixels.",
        applies_to=ALL, default=0, units="counts", stages=S_PEAK,
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # 8. PEAK SEARCH AND INTEGRATION
    # ═══════════════════════════════════════════════════════════════════════

    ParamSpec(
        name="Width", type=ParamType.FLOAT, category="Peak search",
        description="Ring half-width in radial (2θ) direction.",
        applies_to=ALL, default=1500, units="um", stages=S_PEAK | S_INDEX,
    ),

    # ─── Radial integration bounds (RI + calibration) ──────────────────────
    ParamSpec(
        name="RMin", type=ParamType.FLOAT, category="Integration",
        description="Minimum radius for radial integration.",
        applies_to=frozenset({RI}), required_for=frozenset({RI}),
        default=10, units="um", stages=S_INT,
        validators=("non_negative",),
        notes="Stored as lineoutRMin in MIDASConfig; RI uses this to "
              "bound the 1D lineout integration range.",
    ),
    ParamSpec(
        name="RMax", type=ParamType.FLOAT, category="Integration",
        description="Maximum radius for radial integration.",
        applies_to=frozenset({RI}), required_for=frozenset({RI}),
        units="um", stages=S_INT,
        validators=("positive",),
    ),
    ParamSpec(
        name="RBinSize", type=ParamType.FLOAT, category="Integration",
        description="Radial bin width for 1D lineout.",
        applies_to=frozenset({RI}), default=0.25, units="um", stages=S_INT,
        validators=("positive",),
    ),
    ParamSpec(
        name="EtaMin", type=ParamType.FLOAT, category="Integration",
        description="Minimum azimuth for radial integration.",
        applies_to=frozenset({RI}), default=-180, units="deg", stages=S_INT,
    ),
    ParamSpec(
        name="EtaMax", type=ParamType.FLOAT, category="Integration",
        description="Maximum azimuth for radial integration.",
        applies_to=frozenset({RI}), default=180, units="deg", stages=S_INT,
    ),
    ParamSpec(
        name="EtaBinSize", type=ParamType.FLOAT, category="Integration",
        description="η bin size (shared: indexing LUT + radial integration).",
        applies_to=ALL, default=0, units="deg", stages=S_INDEX | S_INT, typical=0.1,
        notes="Dual-use: the FF/NF/PF indexing executables consume it as the "
              "η look-up-table bin size; DetectorMapper and IntegratorZarrOMP "
              "consume it as the azimuthal bin size for the 2D caked output.",
    ),

    # ─── Q-spacing mode (DetectorMapper + IntegratorZarrOMP) ───────────────
    ParamSpec(
        name="QBinSize", type=ParamType.FLOAT, category="Integration",
        description="Q-spacing bin width (enables non-uniform R bins).",
        applies_to=frozenset({RI}), default=0, units="Å⁻¹", stages=S_INT,
        notes="When QBinSize, QMin, QMax, and Wavelength are all positive, "
              "DetectorMapper switches to Q-uniform binning. See "
              "FF_Radial_Integration §A.2a.",
    ),
    ParamSpec(
        name="QMin", type=ParamType.FLOAT, category="Integration",
        description="Minimum Q for Q-spacing mode.",
        applies_to=frozenset({RI}), default=0, units="Å⁻¹", stages=S_INT,
    ),
    ParamSpec(
        name="QMax", type=ParamType.FLOAT, category="Integration",
        description="Maximum Q for Q-spacing mode.",
        applies_to=frozenset({RI}), default=0, units="Å⁻¹", stages=S_INT,
    ),

    # ─── RI frame-aggregation and corrections ──────────────────────────────
    ParamSpec(
        name="SumImages", type=ParamType.INT, category="Integration",
        description="Number of frames to sum per output lineout (0 = per-frame).",
        applies_to=frozenset({RI}), default=0, units="count", stages=S_INT,
    ),
    ParamSpec(
        name="SaveIndividualFrames", type=ParamType.BOOL, category="Integration",
        description="Save per-frame lineouts in addition to summed output.",
        applies_to=frozenset({RI}), default=1, stages=S_INT,
    ),
    ParamSpec(
        name="Normalize", type=ParamType.BOOL, category="Integration",
        description="Per-frame lineout normalization (intensity/monitor scaling).",
        applies_to=frozenset({RI}), default=1, stages=S_INT,
    ),
    ParamSpec(
        name="GradientCorrection", type=ParamType.BOOL, category="Integration",
        description="Apply radial gradient correction.",
        applies_to=frozenset({FF, PF, RI}), default=0, stages=S_INT | S_CALIB,
        hidden_in_wizard=True,
        notes="Read by IntegratorZarrOMP (radial gradient flattening) and by "
              "CalibrantIntegratorOMP (beam-gradient correction during "
              "calibration).",
    ),
    ParamSpec(
        name="SolidAngleCorrection", type=ParamType.BOOL, category="Integration",
        description="Apply cos³(2θ) solid-angle correction (DetectorMapper).",
        applies_to=frozenset({RI}), default=0, stages=S_INT,
    ),
    ParamSpec(
        name="PolarizationCorrection", type=ParamType.BOOL, category="Integration",
        description="Apply polarization correction (DetectorMapper).",
        applies_to=frozenset({RI}), default=0, stages=S_INT,
    ),
    ParamSpec(
        name="PolarizationFraction", type=ParamType.FLOAT, category="Integration",
        description="Polarization σ/π fraction for the correction.",
        applies_to=frozenset({RI}), default=0.99, units="fraction", stages=S_INT,
        notes="Distinct from `Polariz` (GSAS-II instrument profile mixing): "
              "`PolarizationFraction` drives DetectorMapper's pixel-weight "
              "correction; `Polariz` is written into the output zarr's "
              "InstrumentParameters for downstream Rietveld fits.",
    ),

    # ─── RI 1D peak fitting (IntegratorZarrOMP / IntegratorFitPeaksGPUStream) ─
    ParamSpec(
        name="DoSmoothing", type=ParamType.BOOL, category="Integration",
        description="Savitzky-Golay smoothing before automatic peak detection.",
        applies_to=frozenset({RI}), default=0, stages=S_INT,
    ),
    ParamSpec(
        name="MultiplePeaks", type=ParamType.BOOL, category="Integration",
        description="Allow multi-peak fitting on the 1D lineout.",
        applies_to=frozenset({RI}), default=0, stages=S_INT,
    ),
    ParamSpec(
        name="PeakLocation", type=ParamType.FLOAT, category="Integration",
        description="Expected peak radius (pixels); repeatable, one per ring.",
        applies_to=frozenset({RI}), multi_entry=True, units="pixels", stages=S_INT,
        notes="Setting one or more PeakLocation implicitly enables DoPeakFit "
              "and MultiplePeaks, and disables DoSmoothing.",
    ),
    ParamSpec(
        name="AutoDetectPeaks", type=ParamType.INT, category="Integration",
        description="Number of peaks to auto-detect (0 = disabled).",
        applies_to=frozenset({RI}), default=0, units="count", stages=S_INT,
    ),
    ParamSpec(
        name="SNIPIterations", type=ParamType.INT, category="Integration",
        description="SNIP baseline iterations for auto peak detection.",
        applies_to=frozenset({RI}), default=50, units="count", stages=S_INT,
        validators=("positive",),
    ),
    ParamSpec(
        name="FitROIPadding", type=ParamType.INT, category="Integration",
        description="Half-width of peak-fit ROI (radial bins).",
        applies_to=frozenset({RI}), default=20, units="count", stages=S_INT,
    ),
    ParamSpec(
        name="FitROIAuto", type=ParamType.BOOL, category="Integration",
        description="Auto-size ROI from estimated FWHM (overrides FitROIPadding).",
        applies_to=frozenset({RI}), default=0, stages=S_INT,
    ),

    ParamSpec(
        name="UpperBoundThreshold", type=ParamType.INT, category="Peak search",
        description="Saturation cap; pixels above this are ignored in peak search.",
        applies_to=FF_PF, units="counts", stages=S_PEAK,
        validators=("positive",),
    ),
    ParamSpec(
        name="PeakFitMode", type=ParamType.INT, category="Peak search",
        description="Peak fitting model (0 = pseudo-Voigt, 1 = GSAS-II TCH).",
        applies_to=ALL, default=0, stages=S_PEAK, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="SubPixelLevel", type=ParamType.INT, category="Peak search",
        description="Sub-pixel splitting at cardinal η (1 = off).",
        applies_to=ALL, default=1, stages=S_PEAK, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="DoubletSeparation", type=ParamType.FLOAT, category="Calibration",
        description="Minimum separation for resolving doublet rings (calibration).",
        applies_to=FF_PF, default=0, units="um", stages=S_CALIB, hidden_in_wizard=True,
        notes="Read by the calibration executables (CalibrantPanelShiftsOMP, "
              "CalibrantIntegratorOMP) to decide when two closely-spaced "
              "Debye–Scherrer rings are resolved in the lineout. Not used by "
              "the FF peak-fitting pipeline.",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # 9. INDEXING
    # ═══════════════════════════════════════════════════════════════════════

    ParamSpec(
        name="Completeness", type=ParamType.FLOAT, category="Indexing",
        description="Minimum matching fraction to accept a grain.",
        applies_to=FF_PF, required_for=frozenset({FF}), units="fraction",
        typical=0.8, stages=S_INDEX,
        zarr_rename="MinMatchesToAcceptFrac",
        validators=("positive",),
        notes="Applies to both FF and PF. In PF, the value flows through the "
              "Zarr to FitSetupParamsAllZarr → paramstest.txt → "
              "IndexerScanningOMP. If omitted, FitSetupParamsAllZarr.c uses "
              "its internal default (0 = accept everything). MaxAng/TolEta/"
              "TolOme are separate PF-specific tolerances, not replacements.",
    ),
    ParamSpec(
        name="MinNrSpots", type=ParamType.INT, category="Indexing",
        description="Minimum unique solutions before confirming a grain.",
        applies_to=FNP, default=1, typical=3, units="count", stages=S_INDEX,
        validators=("positive",),
    ),
    ParamSpec(
        name="UseFriedelPairs", type=ParamType.BOOL, category="Indexing",
        description="Use Friedel-pair speedup in indexing.",
        applies_to=FF_PF, default=1, typical=1, stages=S_INDEX,
    ),
    ParamSpec(
        name="StepSizeOrient", type=ParamType.FLOAT, category="Indexing",
        description="Orientation search step size.",
        applies_to=FF_PF, required_for=frozenset({FF}), units="deg",
        typical=0.2, stages=S_INDEX,
        aliases=("StepsizeOrient",),  # note lowercase S typo supported
        validators=("positive",),
        notes="Applies to FF and PF (PF reads it via Zarr → "
              "FitSetupParamsAllZarr → paramstest.txt). NF uses OrientTol. "
              "If omitted in PF, FitSetupParamsAllZarr.c default is 0.2.",
    ),
    ParamSpec(
        name="StepSizePos", type=ParamType.FLOAT, category="Indexing",
        description="Position search step size for first-pass indexing.",
        applies_to=FF_PF, required_for=frozenset({FF}), units="um",
        typical=100, stages=S_INDEX,
        validators=("positive",),
        notes="Applies to FF and PF (PF reads it via Zarr → "
              "FitSetupParamsAllZarr → paramstest.txt). If omitted in PF, "
              "FitSetupParamsAllZarr.c default is 5.",
    ),
    ParamSpec(
        name="MarginOme", type=ParamType.FLOAT, category="Indexing",
        description="ω tolerance for spot matching.",
        applies_to=FF_PF, default=0, units="deg", typical=0.5, stages=S_INDEX,
        notes="FF/PF only — NF uses ExcludePoleAngle for its own eta "
              "exclusion scheme.",
    ),
    ParamSpec(
        name="MarginEta", type=ParamType.FLOAT, category="Indexing",
        description="η tolerance for spot matching.",
        applies_to=FF_PF, default=0, units="deg", typical=500, stages=S_INDEX,
        notes="FF/PF only.",
    ),
    ParamSpec(
        name="MarginRadial", type=ParamType.FLOAT, category="Indexing",
        description="2θ (radial) spot-matching tolerance.",
        applies_to=FF_PF, units="um", typical=500, stages=S_INDEX,
    ),
    ParamSpec(
        name="MarginRadius", type=ParamType.FLOAT, category="Indexing",
        description="Equivalent grain radius filter.",
        applies_to=FF_PF, units="um", typical=500, stages=S_INDEX,
    ),
    ParamSpec(
        name="MinConfidence", type=ParamType.FLOAT, category="Indexing",
        description="Minimum confidence threshold for accepting a voxel orientation.",
        applies_to=frozenset({NF}), default=0.5, units="fraction", stages=S_INDEX,
        notes="NF-only. FF/PF use Completeness for the same role.",
    ),
    ParamSpec(
        name="MinFracAccept", type=ParamType.FLOAT, category="Indexing",
        description="Minimum overlap fraction in NF candidate search.",
        applies_to=frozenset({NF}), default=0.5, typical=0.04, units="fraction",
        stages=S_INDEX,
        notes="NF-only. Typical: 0.1 seeded, 0.04 unseeded, 0.01 deformed. "
              "FF/PF use Completeness instead.",
    ),

    # NF-specific indexing
    ParamSpec(
        name="NrOrientations", type=ParamType.INT, category="Indexing",
        description="Number of candidate orientations in NF search space.",
        applies_to=frozenset({NF}), required_for=frozenset({NF}),
        typical=243129, stages=S_INDEX,
        validators=("positive",),
    ),
    ParamSpec(
        name="SeedOrientations", type=ParamType.PATH, category="Indexing",
        description="Seed orientations file.",
        applies_to=frozenset({NF}), required_for=frozenset({NF}),
        stages=frozenset({Stage.SEED_GEN, Stage.INDEXING}),
        validators=("file_exists",),
    ),
    ParamSpec(
        name="SeedOrientationsAll", type=ParamType.PATH, category="Indexing",
        description="Combined seed set across resolutions (multi-res driver).",
        applies_to=frozenset({NF}), stages=S_INDEX,
        validators=("file_exists",),
    ),
    ParamSpec(
        name="OrientTol", type=ParamType.FLOAT, category="Indexing",
        description="Optimization radius after first-pass candidates.",
        applies_to=frozenset({NF}), default=2, units="deg", stages=S_INDEX,
        validators=("positive",),
    ),
    ParamSpec(
        name="SaveNSolutions", type=ParamType.INT, category="Indexing",
        description="Number of top orientations to save per voxel.",
        applies_to=frozenset({NF}), default=1, units="count", stages=S_INDEX,
        validators=("positive",),
    ),
    ParamSpec(
        name="NearestMisorientation", type=ParamType.BOOL, category="Indexing",
        description="Enforce nearest-neighbor misorientation constraint.",
        applies_to=frozenset({NF}), default=0, stages=S_INDEX,
        hidden_in_wizard=True,
    ),
    ParamSpec(
        name="MinMisoNSaves", type=ParamType.FLOAT, category="Indexing",
        description="Minimum misorientation when NearestMisorientation=1.",
        applies_to=frozenset({NF}), default=0, units="deg", stages=S_INDEX,
        hidden_in_wizard=True,
    ),
    ParamSpec(
        name="GridRefactor", type=ParamType.FLOAT, category="Indexing",
        description="Grid subdivision factor per resolution step (multi-res).",
        applies_to=frozenset({NF}), default=2, stages=S_INDEX,
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # 10. SAMPLE GEOMETRY
    # ═══════════════════════════════════════════════════════════════════════

    ParamSpec(
        name="Rsample", type=ParamType.FLOAT, category="Sample geometry",
        description="Horizontal sample radius (grain positions limited to ±Rsample).",
        applies_to=FNP, required_for=frozenset({NF}), units="um", stages=S_INDEX,
        validators=("positive",),
    ),
    ParamSpec(
        name="Hbeam", type=ParamType.FLOAT, category="Sample geometry",
        description="Beam vertical size (grain positions limited to ±Hbeam/2).",
        applies_to=FNP, default=0, units="um", stages=S_INDEX,
    ),
    ParamSpec(
        name="Vsample", type=ParamType.FLOAT, category="Sample geometry",
        description="Illuminated volume for grain size calculation.",
        applies_to=FF_PF, units="um³", stages=S_INDEX,
    ),
    ParamSpec(
        name="BeamThickness", type=ParamType.FLOAT, category="Sample geometry",
        description="Beam vertical thickness.",
        applies_to=FNP, default=0, units="um", stages=S_INDEX,
    ),
    ParamSpec(
        name="BeamSize", type=ParamType.FLOAT, category="Sample geometry",
        description="Horizontal beam size (PF-HEDM).",
        applies_to=frozenset({PF}), default=0, units="um", stages=S_INDEX,
        notes="PF-only: used to compute per-scan positions when no "
              "PositionsFile is provided. Only PF-unique key besides nScans.",
    ),
    ParamSpec(
        name="GlobalPosition", type=ParamType.FLOAT, category="Sample geometry",
        description="Sample starting position along beam (current layer).",
        applies_to=FNP, default=0, units="um", stages=S_INDEX,
    ),
    ParamSpec(
        name="GlobalPositionFirstLayer", type=ParamType.FLOAT, category="Sample geometry",
        description="Sample position of layer 1 (multi-layer driver).",
        applies_to=frozenset({NF}), default=0, units="um", stages=S_INDEX,
    ),
    ParamSpec(
        name="LayerThickness", type=ParamType.FLOAT, category="Sample geometry",
        description="Layer-to-layer Z step.",
        applies_to=frozenset({NF}), default=0, units="um", stages=S_INDEX,
    ),
    ParamSpec(
        name="DiscModel", type=ParamType.INT, category="Sample geometry",
        description="Beam model (0 = parallel, 1 = focused).",
        applies_to=FF_PF, default=0, stages=S_INDEX,
    ),
    ParamSpec(
        name="DiscArea", type=ParamType.FLOAT, category="Sample geometry",
        description="Illuminated area for focused beam.",
        applies_to=FF_PF, units="um²", stages=S_INDEX,
    ),
    ParamSpec(
        name="tInt", type=ParamType.FLOAT, category="Sample geometry",
        description="Integration time per frame.",
        applies_to=FF_PF, units="s", stages=S_INDEX, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="tGap", type=ParamType.FLOAT, category="Sample geometry",
        description="Dead time between frames.",
        applies_to=FF_PF, units="s", stages=S_INDEX, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="GridSize", type=ParamType.FLOAT, category="Sample geometry",
        description="Voxel grid spacing.",
        applies_to=frozenset({NF}), required_for=frozenset({NF}),
        units="um", stages=S_INDEX,
        validators=("positive",),
    ),
    ParamSpec(
        name="EdgeLength", type=ParamType.FLOAT, category="Sample geometry",
        description="Triangle edge length for mic grid.",
        applies_to=frozenset({NF}), units="um", stages=S_INDEX,
    ),
    ParamSpec(
        name="GridFileName", type=ParamType.PATH, category="Sample geometry",
        description="Hexagonal grid file produced by MakeHexGrid.",
        applies_to=frozenset({NF}), default="grid.txt", stages=S_INDEX,
    ),
    ParamSpec(
        name="GridMask", type=ParamType.FLOAT_LIST, category="Sample geometry",
        description="Grid bounding box `ymin ymax zmin zmax`.",
        applies_to=frozenset({NF}), units="um", stages=S_INDEX,
    ),
    ParamSpec(
        name="TomoImage", type=ParamType.PATH, category="Sample geometry",
        description="Tomography reconstruction used to mask the grid.",
        applies_to=frozenset({NF}), stages=S_INDEX,
        validators=("file_exists",),
    ),
    ParamSpec(
        name="TomoPixelSize", type=ParamType.FLOAT, category="Sample geometry",
        description="Pixel size of tomo image for scaling.",
        applies_to=frozenset({NF}), units="um", stages=S_INDEX,
        validators=("positive",),
    ),
    ParamSpec(
        name="nScans", type=ParamType.INT, category="Sample geometry",
        description="Number of scan positions (1 = FF, >1 = PF with positions.csv).",
        applies_to=FF_PF, default=1, units="count", stages=S_INDEX,
        validators=("positive",),
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # Shared FF/PF multi-scan + indexing vocabulary
    # Most of these are read by the shared MIDAS_ParamParser and are visible
    # on the FF path even when only PF actually uses them at runtime.
    # (PF = FF + {nScans, BeamSize} in practice — no separate parser.)
    # ═══════════════════════════════════════════════════════════════════════
    ParamSpec(
        name="MaxAng", type=ParamType.FLOAT, category="Indexing",
        description="Post-indexing misorientation tolerance for grouping "
                    "voxels into grains / refining orientations (PF workflow).",
        applies_to=FF_PF, required_for=frozenset({PF}),
        units="deg", typical=1.0, stages=S_INDEX,
        validators=("positive",),
        notes="Passed directly by pf_MIDAS.py to findMultipleSolutionsPF and "
              "refinement binaries. Additional to StepSizeOrient/Completeness; "
              "not a replacement. FF accepts the key (shared parser) but does "
              "not consume it.",
    ),
    ParamSpec(
        name="TolEta", type=ParamType.FLOAT, category="Indexing",
        description="Post-indexing η tolerance for sinogram spot matching (PF workflow).",
        applies_to=FF_PF, required_for=frozenset({PF}),
        units="deg", typical=1.0, stages=S_INDEX,
        validators=("positive",),
    ),
    ParamSpec(
        name="TolOme", type=ParamType.FLOAT, category="Indexing",
        description="Post-indexing ω tolerance for sinogram spot matching (PF workflow).",
        applies_to=FF_PF, required_for=frozenset({PF}),
        units="deg", typical=1.0, stages=S_INDEX,
        validators=("positive",),
    ),
    ParamSpec(
        name="OmegaFirstFile", type=ParamType.FLOAT, category="Omega scan",
        description="Starting ω of the first frame of each scan file.",
        applies_to=frozenset({FF, PF, RI}), required_for=frozenset({PF}),
        units="deg", stages=S_INDEX | S_INT,
        notes="PF: distinct from OmegaStart (global start) — per-scan "
              "first-frame omega. RI: IntegratorZarrOMP accepts this as an "
              "alias for OmegaStart (frame metadata only). FF accepts it "
              "via the shared parser but does not consume it.",
    ),
    ParamSpec(
        name="RingToIndex", type=ParamType.INT, category="Ring selection",
        description="Individual ring numbers to use in indexing (PF convention; multi-entry).",
        applies_to=FF_PF, multi_entry=True, stages=S_INDEX,
        notes="Distinct from RingThresh — RingToIndex picks WHICH rings to "
              "consider in indexing. PF workflow only; FF pipelines use "
              "OverAllRingToIndex + RingThresh.",
    ),
    ParamSpec(
        name="PositionsFile", type=ParamType.PATH, category="Data source",
        description="Path to positions.csv (per-scan sample positions).",
        applies_to=FF_PF, stages=S_INDEX,
        validators=("file_exists",),
        notes="Read when nScans > 1. PF can auto-generate this from "
              "BeamSize + ScanStep; FF rarely uses it.",
    ),
    ParamSpec(
        name="MicFile", type=ParamType.PATH, category="Output",
        description="Output .mic file with reconstructed voxel orientations (PF).",
        applies_to=FF_PF, stages=frozenset({Stage.POST_ANALYSIS}),
    ),
    ParamSpec(
        name="nStepsToMerge", type=ParamType.INT, category="Indexing",
        description="Number of scan steps to merge during reconstruction (PF).",
        applies_to=FF_PF, default=1, units="count", stages=S_INDEX,
        validators=("positive",),
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # Zarr-only keys (recognized by ffGenerateZipRefactor, not by central parser)
    # These are written into the Zarr analysis file by name; MIDAS reads
    # them as calibration/integration controls. Surfacing so users who copy
    # them into their text config don't get "unknown key" warnings.
    # ═══════════════════════════════════════════════════════════════════════
    ParamSpec(
        name="UsePixelOverlap", type=ParamType.BOOL, category="Peak search",
        description="Use pixel-overlap deblurring for closely-spaced spots.",
        applies_to=frozenset({FF, PF}), default=0, stages=S_PEAK,
        hidden_in_wizard=True,
    ),
    ParamSpec(
        name="doPeakFit", type=ParamType.BOOL, category="Peak search",
        description="Fit peaks (0 disables — e.g. for LocalMaximaOnly mode).",
        applies_to=frozenset({FF, PF, RI}), default=1, stages=S_PEAK | S_INT,
        aliases=("DoPeakFit",),
        hidden_in_wizard=True,
        notes="Central MIDAS_ParamParser accepts `doPeakFit` (lowercase). "
              "IntegratorZarrOMP accepts `DoPeakFit` (capital D) for the RI "
              "1D lineout fitter; treated as an alias here.",
    ),
    ParamSpec(
        name="UseMaximaPositions", type=ParamType.BOOL, category="Peak search",
        description="Use local-maxima positions instead of fitted centroids.",
        applies_to=frozenset({FF, PF}), default=0, stages=S_PEAK,
        hidden_in_wizard=True,
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # 11. REFINEMENT — control flags
    # ═══════════════════════════════════════════════════════════════════════

    ParamSpec(
        name="MargABC", type=ParamType.FLOAT, category="Refinement",
        description="Lattice `a,b,c` refinement tolerance.",
        applies_to=FF_PF, default=0.3, units="%", stages=S_REFINE,
        typical=4,
    ),
    ParamSpec(
        name="MargABG", type=ParamType.FLOAT, category="Refinement",
        description="Lattice `α,β,γ` refinement tolerance.",
        applies_to=FF_PF, default=0.3, units="%", stages=S_REFINE,
        typical=4,
    ),
    ParamSpec(
        name="FitAllAtOnce", type=ParamType.BOOL, category="Refinement",
        description="Fit all grains simultaneously vs sequentially.",
        applies_to=FF_PF, default=0, stages=S_REFINE, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="Twins", type=ParamType.BOOL, category="Refinement",
        description="Enable twin analysis.",
        applies_to=FF_PF, default=0, stages=S_REFINE,
    ),
    ParamSpec(
        name="TakeGrainMax", type=ParamType.BOOL, category="Refinement",
        description="Twin-analysis: take max-solution grain (redundant with Twins).",
        applies_to=FF_PF, default=0, stages=S_REFINE, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="LocalMaximaOnly", type=ParamType.BOOL, category="Peak search",
        description="Use only local maxima in peak detection (forces doPeakFit=0).",
        applies_to=FF_PF, default=0, stages=S_PEAK, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="GBAngle", type=ParamType.FLOAT, category="Refinement",
        description="Grain-boundary angle tolerance.",
        applies_to=FNP, default=5.0, units="deg", stages=S_REFINE,
    ),
    ParamSpec(
        name="DebugMode", type=ParamType.BOOL, category="Refinement",
        description="Verbose diagnostic output.",
        applies_to=ALL, default=0, stages=frozenset(), hidden_in_wizard=True,
    ),
    ParamSpec(
        name="DoDynamicReassignment", type=ParamType.BOOL, category="Refinement",
        description="Dynamically reassign spots to grains between refinement passes.",
        applies_to=FF_PF, default=0, stages=S_REFINE, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="WeightMask", type=ParamType.FLOAT, category="Refinement",
        description="Weight applied to masked regions in the refinement objective.",
        applies_to=FF_PF, default=0, units="fraction", stages=S_REFINE, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="WeightFitRMSE", type=ParamType.FLOAT, category="Refinement",
        description="Weight applied to the fit RMSE term in the refinement objective.",
        applies_to=FF_PF, default=0, units="fraction", stages=S_REFINE, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="TopLayer", type=ParamType.INT, category="Refinement",
        description="Top layer index used by the calibration/refinement driver.",
        applies_to=FF_PF, default=0, stages=S_REFINE, hidden_in_wizard=True,
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # 11b. CALIBRATION INNER LOOP (peak-fit sweep, iteration control)
    # Read by MIDAS_ParamParser.c and consumed by CalibrantIntegratorOMP /
    # CalibrationCore during iterative detector refinement. All FF/PF.
    # ═══════════════════════════════════════════════════════════════════════

    ParamSpec(
        name="RBinDivisions", type=ParamType.INT, category="Calibration inner loop",
        description="Radial bin sub-divisions during calibrant fitting.",
        applies_to=FF_PF, default=4, stages=S_CALIB, hidden_in_wizard=True,
        validators=("positive",),
    ),
    ParamSpec(
        name="MultFactor", type=ParamType.FLOAT, category="Calibration inner loop",
        description="Outlier detection multiplier.",
        applies_to=FF_PF, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="MinIndicesForFit", type=ParamType.INT, category="Calibration inner loop",
        description="Minimum matched indices required to accept a ring fit.",
        applies_to=FF_PF, default=1, units="count", stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="FitOrWeightedMean", type=ParamType.INT, category="Calibration inner loop",
        description="0 = nonlinear fit, 1 = weighted mean centroid.",
        applies_to=FF_PF, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="nIterations", type=ParamType.INT, category="Calibration inner loop",
        description="Calibration refinement iterations.",
        applies_to=FF_PF, default=1, units="count", stages=S_CALIB, hidden_in_wizard=True,
        validators=("positive",),
    ),
    ParamSpec(
        name="IterOffset", type=ParamType.INT, category="Calibration inner loop",
        description="Iteration numbering offset (for resumed calibration runs).",
        applies_to=FF_PF, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="NormalizeRingWeights", type=ParamType.BOOL, category="Calibration inner loop",
        description="Normalize fit weights across rings.",
        applies_to=FF_PF, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="OutlierIterations", type=ParamType.INT, category="Calibration inner loop",
        description="Outlier-removal iteration count.",
        applies_to=FF_PF, default=1, units="count", stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="RemoveOutliersBetweenIters", type=ParamType.BOOL, category="Calibration inner loop",
        description="Remove outliers between iterations rather than only at the end.",
        applies_to=FF_PF, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="ReFitPeaks", type=ParamType.BOOL, category="Calibration inner loop",
        description="Re-fit peaks after initial pass using refined geometry.",
        applies_to=FF_PF, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="TrimmedMeanFraction", type=ParamType.FLOAT, category="Calibration inner loop",
        description="Fraction retained in trimmed-mean centroid (1.0 = full mean).",
        applies_to=FF_PF, default=1.0, units="fraction", stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="WeightByRadius", type=ParamType.BOOL, category="Calibration inner loop",
        description="Weight fit residuals by ring radius.",
        applies_to=FF_PF, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="WeightByFitSNR", type=ParamType.BOOL, category="Calibration inner loop",
        description="Weight fit residuals by per-peak fit SNR.",
        applies_to=FF_PF, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="WeightByPositionUncertainty", type=ParamType.BOOL, category="Calibration inner loop",
        description="Weight fit residuals by fitted position uncertainty.",
        applies_to=FF_PF, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="AdaptiveEtaBins", type=ParamType.BOOL, category="Calibration inner loop",
        description="Adapt η bin widths to spot density.",
        applies_to=FF_PF, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="L2Objective", type=ParamType.BOOL, category="Calibration inner loop",
        description="Use L2 norm in the optimization objective (default is L1).",
        applies_to=FF_PF, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="ConvergenceThresholdPPM", type=ParamType.FLOAT, category="Calibration inner loop",
        description="Early-stop Δstrain threshold; 0 disables.",
        applies_to=FF_PF, default=0, units="ppm", stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="SkipVerification", type=ParamType.BOOL, category="Calibration inner loop",
        description="Skip the final verification E-step.",
        applies_to=FF_PF, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="RingDiagnosticsCSV", type=ParamType.PATH, category="Calibration inner loop",
        description="Output path for per-ring calibration diagnostics CSV.",
        applies_to=FF_PF, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="ResumeFromCheckpoint", type=ParamType.BOOL, category="Calibration inner loop",
        description="Resume calibration from the last checkpoint file.",
        applies_to=FF_PF, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="SubPixelCardinalWidth", type=ParamType.FLOAT, category="Peak search",
        description="Half-width for cardinal-η sub-pixel splitting.",
        applies_to=ALL, default=5.0, units="deg", stages=S_PEAK, hidden_in_wizard=True,
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # 12. OUTPUT PATHS
    # ═══════════════════════════════════════════════════════════════════════

    ParamSpec(
        name="OutputFolder", type=ParamType.PATH, category="Output",
        description="Intermediate analysis output directory.",
        applies_to=FF_PF, stages=S_FILE,
    ),
    ParamSpec(
        name="ResultFolder", type=ParamType.PATH, category="Output",
        description="Final results directory (Grains.csv, Results/).",
        applies_to=FF_PF, stages=S_FILE,
    ),
    ParamSpec(
        name="MicFileBinary", type=ParamType.PATH, category="Output",
        description="Output binary .mic file with voxel orientations.",
        applies_to=frozenset({NF}), required_for=frozenset({NF}),
        stages=frozenset({Stage.POST_ANALYSIS}),
    ),
    ParamSpec(
        name="MicFileText", type=ParamType.PATH, category="Output",
        description="Text .mic file.",
        applies_to=frozenset({NF}), required_for=frozenset({NF}),
        stages=frozenset({Stage.POST_ANALYSIS}),
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # 13. SEEDED / TRACKED WORKFLOWS
    # ═══════════════════════════════════════════════════════════════════════

    ParamSpec(
        name="GrainsFile", type=ParamType.PATH, category="Seeding",
        description="Input Grains.csv for refinement/tracking.",
        applies_to=FNP, stages=S_REFINE,
        validators=("file_exists",),
    ),
    ParamSpec(
        name="RefinementFileName", type=ParamType.PATH, category="Seeding",
        description="Refinement input file (FF refinement; InFileName in forward-sim).",
        applies_to=FF_PF, stages=S_REFINE,
        validators=("file_exists",),
        notes="MIDAS_ParamParser.c treats InFileName and RefinementFileName as "
              "aliases for one C-level field, but the validator keeps them "
              "separate so forward-sim and refinement users get distinct "
              "error messages. Do not add InFileName to aliases here.",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # 14. IMAGE PROCESSING FILTERS (mostly NF)
    # ═══════════════════════════════════════════════════════════════════════

    ParamSpec(
        name="BlanketSubtraction", type=ParamType.INT, category="Image processing",
        description="Flat DC offset subtracted from all pixels.",
        applies_to=frozenset({NF}), default=0, units="counts", stages=S_IMG,
    ),
    ParamSpec(
        name="MedFiltRadius", type=ParamType.INT, category="Image processing",
        description="Median filter radius.",
        applies_to=frozenset({NF}), default=1, units="pixels", stages=S_IMG,
    ),
    ParamSpec(
        name="GaussFiltRadius", type=ParamType.FLOAT, category="Image processing",
        description="Gaussian filter σ.",
        applies_to=frozenset({NF}), default=4, units="pixels", stages=S_IMG,
    ),
    ParamSpec(
        name="DoLoGFilter", type=ParamType.BOOL, category="Image processing",
        description="Enable Laplacian-of-Gaussian filter.",
        applies_to=frozenset({NF}), default=0, stages=S_IMG,
    ),
    ParamSpec(
        name="LoGMaskRadius", type=ParamType.INT, category="Image processing",
        description="LoG mask radius (should be ≥ 2.5× GaussFiltRadius).",
        applies_to=frozenset({NF}), default=10, units="pixels", stages=S_IMG,
    ),
    ParamSpec(
        name="Deblur", type=ParamType.BOOL, category="Image processing",
        description="Enable deblurring step (forces WriteFinImage=1).",
        applies_to=frozenset({NF}), default=0, stages=S_IMG,
    ),
    ParamSpec(
        name="WriteFinImage", type=ParamType.BOOL, category="Image processing",
        description="Write uncompressed reduced images.",
        applies_to=frozenset({NF}), default=0, stages=S_IMG,
    ),
    ParamSpec(
        name="WriteLegacyBin", type=ParamType.BOOL, category="Image processing",
        description="Write legacy per-frame .bin output.",
        applies_to=frozenset({NF}), default=0, stages=S_IMG, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="SkipImageBinning", type=ParamType.BOOL, category="Image processing",
        description="Skip 2×2 binning in MMapImageInfo.",
        applies_to=frozenset({NF}), default=0, stages=S_IMG, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="PrecomputedSpotsInfo", type=ParamType.BOOL, category="Image processing",
        description="Read existing SpotsInfo.bin rather than regenerate.",
        applies_to=frozenset({NF}), default=0, stages=S_IMG, hidden_in_wizard=True,
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # 15. MULTI-PANEL DETECTOR (advanced, hidden in wizard)
    # ═══════════════════════════════════════════════════════════════════════

    ParamSpec(
        name="NPanelsY", type=ParamType.INT, category="Multi-panel",
        description="Panels in Y direction.",
        applies_to=FF_PF, default=0, stages=S_INDEX, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="NPanelsZ", type=ParamType.INT, category="Multi-panel",
        description="Panels in Z direction.",
        applies_to=FF_PF, default=0, stages=S_INDEX, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="PanelSizeY", type=ParamType.INT, category="Multi-panel",
        description="Pixels per panel in Y.",
        applies_to=FF_PF, default=0, units="pixels", stages=S_INDEX, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="PanelSizeZ", type=ParamType.INT, category="Multi-panel",
        description="Pixels per panel in Z.",
        applies_to=FF_PF, default=0, units="pixels", stages=S_INDEX, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="PanelGapsY", type=ParamType.INT_LIST, category="Multi-panel",
        description="Gap widths between Y-panels (up to 10 on one line).",
        applies_to=FF_PF, units="pixels", stages=S_INDEX, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="PanelGapsZ", type=ParamType.INT_LIST, category="Multi-panel",
        description="Gap widths between Z-panels.",
        applies_to=FF_PF, units="pixels", stages=S_INDEX, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="PanelShiftsFile", type=ParamType.PATH, category="Multi-panel",
        description="Per-panel geometry corrections file.",
        applies_to=FF_PF, stages=S_INDEX, hidden_in_wizard=True,
        validators=("file_exists",),
    ),
    ParamSpec(
        name="BigDetSize", type=ParamType.INT, category="Multi-panel",
        description="Size of assembled detector (multi-detector mode).",
        applies_to=FF_PF, default=0, units="pixels", stages=S_INDEX, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="LsdMean", type=ParamType.FLOAT, category="Multi-panel",
        description="Mean Lsd for multi-distance NF grids / multi-detector mode.",
        applies_to=FF_PF, default=0, units="um", stages=S_INDEX, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="DetParams", type=ParamType.FLOAT_LIST, category="Multi-panel",
        description="Per-detector geometry row (10 values, repeatable up to 10 panels).",
        applies_to=FF_PF, multi_entry=True, stages=S_INDEX, hidden_in_wizard=True,
        notes="Each line encodes one detector's `Lsd, tx, ty, tz, yBC, zBC, "
              "p0, p1, p2, RhoD`. Only used when BigDetSize > 0.",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # 16. NF CALIBRATION-ONLY (hidden in wizard)
    # ═══════════════════════════════════════════════════════════════════════

    ParamSpec(
        name="LsdTol", type=ParamType.FLOAT, category="Calibration",
        description="Lsd optimization search range (NF calibration).",
        applies_to=frozenset({NF}), units="um", stages=S_CALIB, hidden_in_wizard=True,
        validators=("positive",),
    ),
    ParamSpec(
        name="LsdRelativeTol", type=ParamType.FLOAT, category="Calibration",
        description="Relative Lsd accuracy between distances.",
        applies_to=frozenset({NF}), units="um", stages=S_CALIB, hidden_in_wizard=True,
        validators=("positive",),
    ),
    ParamSpec(
        name="BCTol", type=ParamType.FLOAT_LIST, category="Calibration",
        description="Beam center tolerance `[y_tol z_tol]`.",
        applies_to=frozenset({NF}), units="pixels", stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="TiltsTol", type=ParamType.FLOAT, category="Calibration",
        description="Tilt angle tolerance.",
        applies_to=frozenset({NF}), units="deg", stages=S_CALIB, hidden_in_wizard=True,
        validators=("positive",),
    ),
    ParamSpec(
        name="NumIterations", type=ParamType.INT, category="Calibration",
        description="Calibration iterations (MultiPoint).",
        applies_to=frozenset({NF}), default=3, units="count", stages=S_CALIB,
        hidden_in_wizard=True,
        validators=("positive",),
    ),
    ParamSpec(
        name="GridPoints", type=ParamType.INT_LIST, category="Calibration",
        description="Grid points `[nx ny]` to calibrate (MultiPoint).",
        applies_to=frozenset({NF}), stages=S_CALIB, hidden_in_wizard=True,
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # 17. FORWARD SIMULATION (hidden in wizard)
    # ═══════════════════════════════════════════════════════════════════════

    ParamSpec(
        name="InFileName", type=ParamType.PATH, category="Forward simulation",
        description="Input grain orientations file.",
        applies_to=FNP, stages=S_SIM, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="OutFileName", type=ParamType.STR, category="Forward simulation",
        description="Output file stem (produces `{stem}_scanNr_0.zip`).",
        applies_to=FF_PF, stages=S_SIM, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="PeakIntensity", type=ParamType.FLOAT, category="Forward simulation",
        description="Peak amplitude in simulated spots.",
        applies_to=FF_PF, default=2000.0, units="counts", stages=S_SIM, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="MaxOutputIntensity", type=ParamType.FLOAT, category="Forward simulation",
        description="Saturation cap.",
        applies_to=FF_PF, default=65000.0, units="counts", stages=S_SIM, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="GaussWidth", type=ParamType.FLOAT, category="Forward simulation",
        description="Gaussian width of simulated spots.",
        applies_to=FNP, default=0, units="pixels", stages=S_SIM, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="SimNoiseSigma", type=ParamType.FLOAT, category="Forward simulation",
        description="Gaussian noise level.",
        applies_to=FF_PF, default=0, units="fraction", stages=S_SIM, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="WriteSpots", type=ParamType.BOOL, category="Forward simulation",
        description="Write SpotMatrixGen.csv.",
        applies_to=FF_PF, default=1, stages=S_SIM, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="WriteImage", type=ParamType.BOOL, category="Forward simulation",
        description="Write simulated detector images.",
        applies_to=frozenset({FF, NF, PF}), default=1, stages=S_SIM, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="OnlySpotsInfo", type=ParamType.BOOL, category="Forward simulation",
        description="NF simulate: skip image writing, produce SpotsInfo.bin only.",
        applies_to=frozenset({NF}), default=0, stages=S_SIM, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="SimulationBatches", type=ParamType.INT, category="Forward simulation",
        description="NF simulate: memory-partition count for very large simulations.",
        applies_to=frozenset({NF}), default=0, stages=S_SIM, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="IntensitiesFile", type=ParamType.PATH, category="Forward simulation",
        description="Per-grain intensities input file.",
        applies_to=FF_PF, stages=S_SIM, hidden_in_wizard=True,
        validators=("file_exists",),
    ),
    ParamSpec(
        name="IsBinary", type=ParamType.BOOL, category="Forward simulation",
        description="Input grain orientations file is binary (not text).",
        applies_to=FF_PF, default=0, stages=S_SIM, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="LoadNr", type=ParamType.INT, category="Forward simulation",
        description="Grain index to load from a binary input file.",
        applies_to=FF_PF, default=0, stages=S_SIM, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="UpdatedOrientations", type=ParamType.BOOL, category="Forward simulation",
        description="Use refined orientations from a prior analysis pass.",
        applies_to=FF_PF, default=1, stages=S_SIM, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="NFOutput", type=ParamType.BOOL, category="Forward simulation",
        description="Emit output in NF-style stacked format instead of FF.",
        applies_to=FF_PF, default=0, stages=S_SIM, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="NoSaveAll", type=ParamType.BOOL, category="Forward simulation",
        description="Skip writing the all-spots aggregate file.",
        applies_to=FF_PF, default=0, stages=S_SIM, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="EResolution", type=ParamType.FLOAT_LIST, category="Forward simulation",
        description="Energy resolution and λ-sample count `resolution num_lambda_samples`.",
        applies_to=FF_PF, default=[0.0, 1.0], stages=S_SIM, hidden_in_wizard=True,
        notes="Single line with two values; parsed as (dE/E fraction, "
              "integer sample count).",
    ),
    ParamSpec(
        name="OutDirPath", type=ParamType.PATH, category="Forward simulation",
        description="Output directory for forward-simulation products.",
        applies_to=FF_PF, stages=S_SIM, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="InputAllExtraInfoFittingAll", type=ParamType.PATH, category="Seeding",
        description="Pre-computed spots input (used with -provideInputAll 1).",
        applies_to=FF_PF, stages=S_REFINE, hidden_in_wizard=True,
        validators=("file_exists",),
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # 18. POST-PROCESSING
    # ═══════════════════════════════════════════════════════════════════════

    ParamSpec(
        name="MaxAngle", type=ParamType.FLOAT, category="Post-processing",
        description="Misorientation threshold for grain boundary detection.",
        applies_to=frozenset({NF}), units="deg",
        stages=frozenset({Stage.POST_ANALYSIS}), hidden_in_wizard=True,
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # 19. REFINEMENT TOLERANCES (calibration subsystem, hidden in wizard)
    # ═══════════════════════════════════════════════════════════════════════

    ParamSpec(
        name="tolTilts", type=ParamType.FLOAT, category="Calibration",
        description="Fallback for tx/ty/tz tolerances.",
        applies_to=ALL, default=0, units="deg", stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="tolBC", type=ParamType.FLOAT, category="Calibration",
        description="Beam center refinement tolerance.",
        applies_to=ALL, default=0, units="pixels", stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="tolLsd", type=ParamType.FLOAT, category="Calibration",
        description="Sample-detector distance tolerance.",
        applies_to=ALL, default=0, units="um", stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="tolP", type=ParamType.FLOAT, category="Calibration",
        description="General distortion-coefficient fallback.",
        applies_to=FF_PF, default=0, stages=S_CALIB, hidden_in_wizard=True,
        notes="Fallback used when a per-coefficient tolP0..tolP14 is not set "
              "and the coefficient has no periodicity default.",
    ),
    # Per-coefficient distortion tolerances. Phase-angle coefficients
    # (p3, p6, p8, p10, p12, p14) have periodicity-constant defaults and
    # are typically NOT user-refined; the amplitudes default to 0 (fall back
    # to `tolP` via midas_apply_tol_defaults in MIDAS_ParamParser.c).
    ParamSpec(
        name="tolP0", type=ParamType.FLOAT, category="Calibration",
        description="Tolerance for p0 (2-fold amplitude).",
        applies_to=FF_PF, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="tolP1", type=ParamType.FLOAT, category="Calibration",
        description="Tolerance for p1 (4-fold amplitude).",
        applies_to=FF_PF, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="tolP2", type=ParamType.FLOAT, category="Calibration",
        description="Tolerance for p2 (isotropic R²).",
        applies_to=FF_PF, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="tolP3", type=ParamType.FLOAT, category="Calibration",
        description="Tolerance for p3 (4-fold phase; periodicity default 45°).",
        applies_to=FF_PF, default=45, units="deg", stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="tolP4", type=ParamType.FLOAT, category="Calibration",
        description="Tolerance for p4 (isotropic R⁶).",
        applies_to=FF_PF, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="tolP5", type=ParamType.FLOAT, category="Calibration",
        description="Tolerance for p5 (isotropic R⁴).",
        applies_to=FF_PF, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="tolP6", type=ParamType.FLOAT, category="Calibration",
        description="Tolerance for p6 (2-fold phase; periodicity default 90°).",
        applies_to=FF_PF, default=90, units="deg", stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="tolP7", type=ParamType.FLOAT, category="Calibration",
        description="Tolerance for p7 (dipole amplitude).",
        applies_to=FF_PF, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="tolP8", type=ParamType.FLOAT, category="Calibration",
        description="Tolerance for p8 (dipole phase; periodicity default 180°).",
        applies_to=FF_PF, default=180, units="deg", stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="tolP9", type=ParamType.FLOAT, category="Calibration",
        description="Tolerance for p9 (trefoil amplitude).",
        applies_to=FF_PF, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="tolP10", type=ParamType.FLOAT, category="Calibration",
        description="Tolerance for p10 (trefoil phase; periodicity default 180°).",
        applies_to=FF_PF, default=180, units="deg", stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="tolP11", type=ParamType.FLOAT, category="Calibration",
        description="Tolerance for p11 (pentafoil amplitude).",
        applies_to=FF_PF, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="tolP12", type=ParamType.FLOAT, category="Calibration",
        description="Tolerance for p12 (pentafoil phase; periodicity default 180°).",
        applies_to=FF_PF, default=180, units="deg", stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="tolP13", type=ParamType.FLOAT, category="Calibration",
        description="Tolerance for p13 (hexafoil amplitude).",
        applies_to=FF_PF, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="tolP14", type=ParamType.FLOAT, category="Calibration",
        description="Tolerance for p14 (hexafoil phase; periodicity default 180°).",
        applies_to=FF_PF, default=180, units="deg", stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="tolP2Panel", type=ParamType.FLOAT, category="Calibration",
        description="Per-panel p2 (radial distortion) tolerance.",
        applies_to=FF_PF, default=0.0001, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="tolTiltX", type=ParamType.FLOAT, category="Calibration",
        description="Individual tx tolerance (overrides tolTilts fallback).",
        applies_to=FF_PF, default=0, units="deg", stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="tolLsdPanel", type=ParamType.FLOAT, category="Calibration",
        description="Per-panel Lsd tolerance.",
        applies_to=FF_PF, default=100, units="um", stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="tolParallax", type=ParamType.FLOAT, category="Calibration",
        description="Parallax correction tolerance.",
        applies_to=FF_PF, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="FitParallax", type=ParamType.BOOL, category="Calibration",
        description="Refine parallax.",
        applies_to=FF_PF, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="PerPanelLsd", type=ParamType.BOOL, category="Calibration",
        description="Independent Lsd per panel.",
        applies_to=FF_PF, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="PerPanelDistortion", type=ParamType.BOOL, category="Calibration",
        description="Independent distortion per panel.",
        applies_to=FF_PF, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="tolShifts", type=ParamType.FLOAT, category="Calibration",
        description="Panel shift refinement tolerance.",
        applies_to=ALL, default=1.0, units="um", stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="tolRotation", type=ParamType.FLOAT, category="Calibration",
        description="Panel rotation tolerance.",
        applies_to=ALL, default=0, units="deg", stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="tolWavelength", type=ParamType.FLOAT, category="Calibration",
        description="Wavelength refinement tolerance.",
        applies_to=ALL, default=0.001, units="Å", stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="FitWavelength", type=ParamType.BOOL, category="Calibration",
        description="Refine wavelength.",
        applies_to=ALL, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
    ParamSpec(
        name="FixPanelID", type=ParamType.INT, category="Calibration",
        description="Hold this panel fixed (don't refine).",
        applies_to=FF_PF, default=0, stages=S_CALIB, hidden_in_wizard=True,
    ),
]


# ─── Index helpers ───────────────────────────────────────────────────────────


def by_name() -> dict[str, ParamSpec]:
    """Fast lookup including aliases. An alias resolves to the canonical spec."""
    out: dict[str, ParamSpec] = {}
    for p in PARAMS:
        out[p.name] = p
        for a in p.aliases:
            out[a] = p
    return out


def for_path(path: Path) -> list[ParamSpec]:
    """All parameters applicable to a pipeline path."""
    return [p for p in PARAMS if path in p.applies_to]


def required_for(path: Path) -> list[ParamSpec]:
    """Parameters that must be present for the given path."""
    return [p for p in PARAMS if path in p.required_for]


def wizard_visible_for(path: Path) -> list[ParamSpec]:
    """Params to show in the wizard for a given path."""
    return [p for p in PARAMS
            if path in p.applies_to and not p.hidden_in_wizard]
