"""Parameter-file parsing for NF-HEDM fit drivers.

Parses the same plain-text parameter files consumed by the C executables
(``FitOrientationOMP``, ``FitOrientationParameters``,
``FitOrientationParametersMultiPoint``) and returns a single typed
:class:`FitParams` dataclass. Recognises every key the C codes look for,
plus a small set of new keys that toggle the differentiable-surrogate
behaviour and box-constraint regularisation.

Existing C keys (drop-in compatible)
-----------------------------------
``nDistances``, ``DataDirectory``, ``OutputDirectory``, ``ReducedFileName``,
``MicFileBinary``, ``GridFileName``, ``Lsd``, ``BC``, ``tx``, ``ty``,
``tz``, ``Wedge``, ``Wavelength``, ``px``, ``OmegaStart``, ``OmegaStep``,
``StartNr``, ``EndNr``, ``ExcludePoleAngle``, ``MinFracAccept``,
``OrientTol``, ``LsdTol``, ``LsdRelativeTol``, ``BCTol`` (a, b),
``TiltsTol``, ``WedgeTol`` (new — only used if ``RefineWedge 1``),
``OmegaRange``, ``BoxSize``, ``LatticeParameter``/``LatticeConstant``,
``RingsToUse``, ``MaxRingRad``, ``SpaceGroup``, ``SaveNSolutions``,
``MinMisoNSaves``, ``NearestMisorientation``,
``NumIterations``, ``GridSize``, ``GridPoints``, ``NrPixels``,
``NrPixelsY``, ``NrPixelsZ``.

The C key ``Ice9Input`` (a Z-axis flip toggle from the original
detector orientation work) is **deprecated** and silently ignored if
encountered.

New keys (extensions)
---------------------
``RefineWedge``, ``WedgeTol``, ``TikhonovCalibration``,
``TikhonovSigmaLsd``, ``TikhonovSigmaTilts``, ``TikhonovSigmaBC``,
``TikhonovSigmaWedge``, ``GaussianSplatSigmaPx`` (overrides automatic σ
choice for the soft-overlap surrogate).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class FitParams:
    """Configuration for an NF-HEDM fit run.

    Mirrors the union of keys consumed by the three C drivers, plus a
    small set of new fields for the differentiable-surrogate path.

    Distance / detector geometry
        n_distances, Lsd, ybc, zbc, px, n_pixels_y, n_pixels_z
    Sample axes / wedge
        tx, ty, tz, wedge
    Scan
        omega_start, omega_step, start_nr, end_nr,
        omega_ranges, box_sizes
    Diffraction
        wavelength, lattice_constant, space_group, max_ring_rad,
        rings_to_use, exclude_pole_angle
    Filenames
        data_dir, output_dir, reduced_file_name, mic_file_binary,
        grid_file_name (None ⇒ ``grid.txt``)
    Fit tolerances (used as ±tol box bounds for L-BFGS reparam)
        orient_tol, lsd_tol, lsd_rel_tol, bc_tol_a, bc_tol_b,
        tilts_tol, wedge_tol
    Phase 1 screen
        min_frac_accept
    Phase 2 fit
        save_n_solutions, min_miso_n_saves_deg, num_iterations
    Multipoint-only
        grid_size, grid_points (list of (xc, yc, ud, eul1, eul2, eul3))
    Toggles
        ice9_input, nearest_miso, refine_wedge
    Differentiable-surrogate knobs
        gaussian_splat_sigma_px (None ⇒ auto from voxel size)
    Tikhonov
        tikhonov_calibration, tikhonov_sigma_*
    """

    # detector / geometry
    n_distances: int = 1
    Lsd: List[float] = field(default_factory=list)
    ybc: List[float] = field(default_factory=list)
    zbc: List[float] = field(default_factory=list)
    px: float = 0.0
    n_pixels_y: int = 2048
    n_pixels_z: int = 2048

    # sample axes
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    wedge: float = 0.0

    # scan
    omega_start: float = 0.0
    omega_step: float = 0.0
    start_nr: int = 0
    end_nr: int = 0
    omega_ranges: List[Tuple[float, float]] = field(default_factory=list)
    box_sizes: List[Tuple[float, float, float, float]] = field(default_factory=list)

    # diffraction
    wavelength: float = 0.0
    lattice_constant: Tuple[float, float, float, float, float, float] = (
        0.0, 0.0, 0.0, 90.0, 90.0, 90.0,
    )
    space_group: int = 0
    max_ring_rad: float = 0.0
    rings_to_use: List[int] = field(default_factory=list)
    exclude_pole_angle: float = 0.0

    # filenames
    data_dir: str = "."
    output_dir: str = ""
    reduced_file_name: str = ""
    mic_file_binary: str = ""
    mic_file_text: str = ""
    grid_file_name: Optional[str] = None

    # tolerances
    orient_tol: float = 1.0           # degrees
    lsd_tol: float = 1000.0           # microns
    lsd_rel_tol: float = 100.0        # microns
    bc_tol_a: float = 1.0             # pixels
    bc_tol_b: float = 1.0             # pixels
    tilts_tol: float = 0.05           # degrees
    wedge_tol: float = 0.05           # degrees

    # screen
    min_frac_accept: float = 0.6
    min_confidence: float = 0.0

    # fit
    save_n_solutions: int = 1
    min_miso_n_saves_deg: float = 1.0
    num_iterations: int = 1

    # multipoint
    grid_size_um: float = 0.0
    grid_points: List[Tuple[float, float, float, float, float, float]] = field(
        default_factory=list
    )  # (xc, yc, ud, eul1, eul2, eul3)

    # toggles
    nearest_miso: bool = False
    refine_wedge: bool = False

    # surrogate
    gaussian_splat_sigma_px: Optional[float] = None

    # Tikhonov
    tikhonov_calibration: float = 0.0
    tikhonov_sigma_lsd: float = 100.0       # microns
    tikhonov_sigma_tilts: float = 0.05      # degrees
    tikhonov_sigma_bc: float = 1.0          # pixels
    tikhonov_sigma_wedge: float = 0.05      # degrees

    @property
    def out_dir(self) -> str:
        """Effective output directory (``output_dir`` or ``data_dir``)."""
        return self.output_dir or self.data_dir

    @property
    def n_frames_per_distance(self) -> int:
        """Number of files per distance (``end_nr - start_nr + 1``)."""
        return self.end_nr - self.start_nr + 1


# ---------------------------------------------------------------------------
#  paramfile parser
# ---------------------------------------------------------------------------

def parse_paramfile(path: str | Path) -> FitParams:
    """Parse a paramfile into a :class:`FitParams` object.

    Unknown lines are silently ignored, matching the C behaviour: every
    keyword is independently recognised and unrecognised lines just fall
    through. This keeps the parser tolerant of stray comments and of new
    keys added by other workflows.
    """
    p = FitParams()

    # First sweep: nDistances must be known so per-distance lists are sized
    # correctly. The C code does the same two-pass strategy.
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("nDistances"):
                p.n_distances = int(line.split()[1])
                break

    Lsd: List[float] = []
    ybc: List[float] = []
    zbc: List[float] = []

    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            key = tokens[0]
            args = tokens[1:]

            try:
                # ---------- filenames ----------
                if key == "DataDirectory":
                    p.data_dir = args[0]
                elif key == "OutputDirectory":
                    p.output_dir = args[0]
                elif key == "ReducedFileName":
                    p.reduced_file_name = args[0]
                elif key == "MicFileBinary":
                    p.mic_file_binary = args[0]
                elif key == "MicFileText":
                    p.mic_file_text = args[0]
                elif key == "GridFileName":
                    p.grid_file_name = args[0]

                # ---------- per-distance ----------
                elif key == "Lsd":
                    Lsd.append(float(args[0]))
                elif key == "BC":
                    ybc.append(float(args[0]))
                    zbc.append(float(args[1]))

                # ---------- scalar geometry ----------
                elif key == "px":
                    p.px = float(args[0])
                elif key == "tx":
                    p.tx = float(args[0])
                elif key == "ty":
                    p.ty = float(args[0])
                elif key == "tz":
                    p.tz = float(args[0])
                elif key == "Wedge":
                    p.wedge = float(args[0])
                elif key == "Wavelength":
                    p.wavelength = float(args[0])
                elif key == "NrPixels":
                    p.n_pixels_y = p.n_pixels_z = int(args[0])
                elif key == "NrPixelsY":
                    p.n_pixels_y = int(args[0])
                elif key == "NrPixelsZ":
                    p.n_pixels_z = int(args[0])

                # ---------- scan ----------
                elif key == "OmegaStart":
                    p.omega_start = float(args[0])
                elif key == "OmegaStep":
                    p.omega_step = float(args[0])
                elif key == "StartNr":
                    p.start_nr = int(args[0])
                elif key == "EndNr":
                    p.end_nr = int(args[0])
                elif key == "OmegaRange":
                    p.omega_ranges.append((float(args[0]), float(args[1])))
                elif key == "BoxSize":
                    p.box_sizes.append((
                        float(args[0]), float(args[1]),
                        float(args[2]), float(args[3]),
                    ))

                # ---------- diffraction ----------
                elif key in ("LatticeParameter", "LatticeConstant"):
                    p.lattice_constant = tuple(float(x) for x in args[:6])  # type: ignore[assignment]
                elif key == "SpaceGroup":
                    p.space_group = int(args[0])
                elif key == "MaxRingRad":
                    p.max_ring_rad = float(args[0])
                elif key == "RingsToUse":
                    p.rings_to_use.append(int(args[0]))
                elif key == "ExcludePoleAngle":
                    p.exclude_pole_angle = float(args[0])

                # ---------- tolerances ----------
                elif key == "OrientTol":
                    p.orient_tol = float(args[0])
                elif key == "LsdTol":
                    p.lsd_tol = float(args[0])
                elif key == "LsdRelativeTol":
                    p.lsd_rel_tol = float(args[0])
                elif key == "BCTol":
                    p.bc_tol_a = float(args[0])
                    p.bc_tol_b = float(args[1])
                elif key == "TiltsTol":
                    p.tilts_tol = float(args[0])
                elif key == "WedgeTol":
                    p.wedge_tol = float(args[0])

                # ---------- screen / fit ----------
                elif key == "MinFracAccept":
                    p.min_frac_accept = float(args[0])
                elif key == "MinConfidence":
                    p.min_confidence = float(args[0])
                elif key == "SaveNSolutions":
                    p.save_n_solutions = int(args[0])
                elif key == "MinMisoNSaves":
                    p.min_miso_n_saves_deg = float(args[0])
                elif key == "NumIterations":
                    p.num_iterations = int(args[0])

                # ---------- multipoint ----------
                elif key == "GridSize":
                    p.grid_size_um = float(args[0])
                elif key == "GridPoints":
                    # GridPoints _ _ _ _ xc yc _ ud eul1 eul2 eul3 _ _
                    # (matches the C sscanf format exactly)
                    if len(args) >= 11:
                        xc = float(args[4])
                        yc = float(args[5])
                        ud = float(args[7])
                        e1 = float(args[8])
                        e2 = float(args[9])
                        e3 = float(args[10])
                        p.grid_points.append((xc, yc, ud, e1, e2, e3))

                # ---------- toggles ----------
                elif key == "Ice9Input":
                    # Deprecated: detector Z-flip toggle from the
                    # original C path. Silently ignored.
                    continue
                elif key == "NearestMisorientation":
                    p.nearest_miso = bool(int(args[0]))
                elif key == "RefineWedge":
                    p.refine_wedge = bool(int(args[0]))

                # ---------- surrogate / Tikhonov ----------
                elif key == "GaussianSplatSigmaPx":
                    p.gaussian_splat_sigma_px = float(args[0])
                elif key == "TikhonovCalibration":
                    p.tikhonov_calibration = float(args[0])
                elif key == "TikhonovSigmaLsd":
                    p.tikhonov_sigma_lsd = float(args[0])
                elif key == "TikhonovSigmaTilts":
                    p.tikhonov_sigma_tilts = float(args[0])
                elif key == "TikhonovSigmaBC":
                    p.tikhonov_sigma_bc = float(args[0])
                elif key == "TikhonovSigmaWedge":
                    p.tikhonov_sigma_wedge = float(args[0])

                # nDistances handled in first pass; ignore other keys.
            except (IndexError, ValueError):
                # Tolerate malformed lines the same way the C sscanf path
                # does — silent skip rather than abort.
                continue

    p.Lsd = Lsd
    p.ybc = ybc
    p.zbc = zbc

    if len(Lsd) != p.n_distances:
        raise ValueError(
            f"paramfile declares nDistances={p.n_distances} but supplies "
            f"{len(Lsd)} Lsd entries"
        )
    if len(ybc) != p.n_distances:
        raise ValueError(
            f"paramfile declares nDistances={p.n_distances} but supplies "
            f"{len(ybc)} BC entries"
        )

    return p
