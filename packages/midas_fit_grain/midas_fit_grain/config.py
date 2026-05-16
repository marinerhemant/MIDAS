"""Refinement configuration.

``FitConfig`` is the single source of truth for everything ``refine_block``
needs to know: geometry, ring list, omega/box ranges, refinement margins, fit
mode, solver choice, loss kind, and backend hints. Constructed either
explicitly or via :py:meth:`FitConfig.from_param_file` which parses the
``paramstest.txt`` written by ``FitSetupParamsAllZarr`` (or by ``ff_MIDAS.py``).

The set of keys read here mirrors what ``FitPosOrStrainsOMP.c`` reads via
``midas_parse_params`` (see ``FF_HEDM/src/FitPosOrStrainsOMP.c:2101-2160``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple

import torch

SolverName = Literal["lbfgs", "adam", "lm", "nelder_mead", "lm_batched"]
ModeName = Literal["iterative", "all_at_once"]
LossKind = Literal["pixel", "angular", "internal_angle"]


@dataclass
class FitConfig:
    """Refinement parameters + solver/loss/mode/backend hints."""

    # --- Geometry (lab frame, micrometers / radians-or-degrees as noted) ---
    Lsd: float = 0.0                       # sample-to-detector distance, um
    px: float = 0.0                        # detector pixel size, um
    Wavelength: float = 0.0                # X-ray wavelength, Å
    Wedge: float = 0.0                     # tilt angle, degrees
    chi: float = 0.0                       # secondary wedge, degrees
    RhoD: float = 0.0                      # max ring radius, um
    Rsample: float = 0.0                   # sample radius, um
    Hbeam: float = 0.0                     # beam height, um

    # --- Crystal ---
    LatticeConstant: Tuple[float, float, float, float, float, float] = (
        0.0, 0.0, 0.0, 90.0, 90.0, 90.0,
    )
    SpaceGroup: int = 0

    # --- Rings ---
    RingNumbers: list[int] = field(default_factory=list)
    RingRadii: list[float] = field(default_factory=list)   # parallel to RingNumbers
    RingsToReject: list[int] = field(default_factory=list)

    # --- Omega / box ranges (parallel) ---
    OmegaRanges: list[Tuple[float, float]] = field(default_factory=list)
    BoxSizes: list[Tuple[float, float, float, float]] = field(default_factory=list)

    # --- Match margins ---
    MarginRadius: float = 0.0       # pixel radius for spot matching
    MarginRadial: float = 0.0
    MarginEta: float = 0.0
    MarginOme: float = 0.0
    EtaBinSize: float = 2.0
    OmeBinSize: float = 2.0
    MinEta: float = 6.0             # excluded pole angle
    MargABC: float = 0.0
    MargABG: float = 0.0

    # --- Fit-time switches (mirror C cfg.* fields) ---
    FitAllAtOnce: int = 0
    DoDynamicReassignment: int = 0
    TopLayer: int = 0
    TakeGrainMax: int = 0

    # --- Paths ---
    OutputFolder: str = "."         # FitBest.bin lives here
    ResultFolder: str = "."         # OrientPosFit.bin / Key.bin live here
    InputFileName: str = "Spots.bin"
    SpotsFileName: str = "Spots.bin"
    IDsFileName: str = "SpotsToIndex.csv"
    RefinementFileName: str = "InputAllExtraInfoFittingAll.csv"
    GrainsFile: str = ""
    ExtraInfoBinFileName: str = "ExtraInfo.bin"
    ProcessKeyFileName: str = "ProcessKey.bin"
    KeyFileName: str = "Key.bin"
    OrientPosFitFileName: str = "OrientPosFit.bin"
    FitBestFileName: str = "FitBest.bin"

    # --- Detector / multi-panel (passed through to forward model) ---
    nDetParams: int = 1
    DetParams: list[Tuple[float, ...]] = field(default_factory=list)
    BigDetSize: float = 0.0

    # --- Multi-detector pinwheel additions (Phase D2) ---
    # ``DetParamsByID[det_id] = {lsd, y_bc, z_bc, tx, ty, tz, p_distortion}``.
    # Empty for single-detector. Populated from ``DetParams`` rows in
    # the merged paramstest emitted by midas-ff-pipeline's
    # cross_det_merge stage. Used by the refiner's per-spot panel-aware
    # forward model (Phase D2).
    DetParamsByID: dict = field(default_factory=dict)
    # Per-(det, ring) ring radius in lab um. Each panel has its own Lsd,
    # so ring radii differ slightly across panels.
    RingRadiiPerDet: dict = field(default_factory=dict)
    # Per-panel η coverage, list of ``(ring, eta_lo_deg, eta_hi_deg)``.
    EtaCoverage: dict = field(default_factory=dict)

    # --- Refinement-package specific (NOT in the C param parser) ---
    solver: SolverName = "lbfgs"
    mode: ModeName = "iterative"
    loss: LossKind = "pixel"
    max_iter: int = 5000
    ftol: float = 1e-5
    xtol: float = 1e-5
    rematch_radius_px: float = 2.0          # used between iterative stages
    phase_steps: Tuple[int, int, int, int] = (4, 4, 4, 4)  # pos / orient / strain / joint
    weight_by_position_uncertainty: int = 0
    weight_by_fit_rmse: int = 0
    debug_mode: int = 0

    # --- Backend hints ---
    device: Optional[str | torch.device] = None
    dtype: Optional[str | torch.dtype] = None
    compile: "bool | str" = False           # passed to HEDMForwardModel(compile=)
                                            # CUDA-only; ignored on CPU/MPS

    # --- Scan-aware (pf-HEDM) extensions ---
    # Default-off ⇒ FF behavior unchanged on every existing test.
    # ``scan_positions_path`` is the path to ``positions.csv`` (1-D Y
    # values, µm). ``scan_pos_tol_um`` enables the per-voxel scan-position
    # filter in the observation builder when > 0 (matches the kernel
    # extension we added to midas-index.compute.matching at P5a).
    # ``position_mode`` controls position refinement: "fixed" matches the
    # C IndexerScanningOMP behavior (voxel position is locked to the scan
    # grid) and "voxel_bounded" is new functionality where positions can
    # refine inside ``voxel_center ± beam_size/2`` along Y, jointly with
    # orientation / strain via ``mode="all_at_once"``. See plan §1e + §7b.
    scan_positions_path: str = ""           # path to positions.csv; "" ⇒ FF
    scan_pos_tol_um: float = 0.0            # 0 ⇒ filter disabled
    friedel_symmetric_scan_filter: bool = True
    beam_size_um: float = 0.0               # bound for voxel_bounded mode
    position_mode: str = "fixed"            # "fixed" | "voxel_bounded"

    # --- Bounded refinement (sigmoid reparameterization, torch-native) ---
    # When ``use_bounds`` is True, ``refine_grain`` reparameterizes euler
    # and lattice as ``x = lb + (ub-lb) * sigmoid(theta)`` and optimizes
    # the unbounded ``theta`` with the chosen solver. The optimizer can
    # never escape ``[lb, ub]`` and the chain rule preserves full
    # autograd portability across CPU/CUDA/MPS — no scipy / NLopt round-trip.
    # Bounds are constructed around the seed: euler_lb = seed - half, etc.
    use_bounds: bool = False
    bound_euler_deg: float = 5.0            # ±half-width on each Euler component
    bound_lat_abc_pct: float = 0.01         # ±fraction on a, b, c
    bound_lat_angle_deg: float = 2.0        # ±half-width on α, β, γ

    # --- Convenience ---
    @property
    def n_rings(self) -> int:
        return len(self.RingNumbers)

    def ring_radius(self, ring_nr: int) -> float:
        for n, r in zip(self.RingNumbers, self.RingRadii):
            if n == ring_nr:
                return r
        return 0.0

    @classmethod
    def from_param_file(
        cls,
        path: str | Path,
        *,
        solver: SolverName = "lbfgs",
        mode: Optional[ModeName] = None,
        loss: LossKind = "pixel",
        device: Optional[str | torch.device] = None,
        dtype: Optional[str | torch.dtype] = None,
        **overrides,
    ) -> "FitConfig":
        """Parse a ``paramstest.txt`` and apply optional refinement overrides.

        ``mode`` defaults to ``iterative`` unless the param file carries
        ``FitAllAtOnce 1``, matching the C default.
        """
        cfg = cls(solver=solver, loss=loss, device=device, dtype=dtype)
        ring_radii_user: list[float] = []

        with open(path, "r") as fp:
            for raw in fp:
                line = raw.rstrip("\r\n").strip()
                if not line or line.startswith("#") or line.startswith("%"):
                    continue
                # Be permissive: handle "Key1 v;" + multi-stmt "Key1 v; Key2 v"
                # by splitting at semicolons first.
                for stmt in line.split(";"):
                    s = stmt.strip()
                    if not s:
                        continue
                    tokens = s.split()
                    key = tokens[0]
                    args = [t.rstrip(";") for t in tokens[1:]]
                    _apply_param(cfg, key, args, ring_radii_user)

        for n, r in zip(cfg.RingNumbers, ring_radii_user):
            # store parallel; ring lookup uses ring_radius()
            cfg.RingRadii.append(r)

        if mode is not None:
            cfg.mode = mode
        else:
            cfg.mode = "all_at_once" if cfg.FitAllAtOnce == 1 else "iterative"

        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise AttributeError(f"FitConfig has no field '{k}'")
            setattr(cfg, k, v)

        return cfg


# ---- internal helpers ----

_FLOAT_KEYS = {
    "px", "Wavelength", "Distance", "Lsd", "Wedge", "chi",
    "Rsample", "Hbeam", "RhoD", "MaxRingRad",
    "MarginOme", "MarginRadius", "MarginRad", "MarginRadial", "MarginEta",
    "EtaBinSize", "OmeBinSize", "MinEta", "ExcludePoleAngle",
    "MargABC", "MargABG", "BigDetSize",
}
_FLOAT_ALIASES = {
    "Distance": "Lsd",
    "MaxRingRad": "RhoD",
    "ExcludePoleAngle": "MinEta",
    "MarginRad": "MarginRadius",
}
_INT_KEYS = {
    "SpaceGroup", "FitAllAtOnce", "DoDynamicReassignment",
    "TopLayer", "TakeGrainMax",
}
_STR_KEYS = {
    "OutputFolder", "ResultFolder", "SpotsFileName", "InputFileName",
    "IDsFileName", "RefinementFileName", "GrainsFile",
}


def _apply_param(cfg: FitConfig, key: str, args: list[str],
                 ring_radii_user: list[float]) -> None:
    # --- Multi-detector pinwheel keys ---
    if key == "DetParams":
        # DetParams det_id Lsd y_bc z_bc tx ty tz p0..p10
        if len(args) >= 7:
            det_id = int(float(args[0]))
            cfg.DetParamsByID[det_id] = {
                "lsd": float(args[1]),
                "y_bc": float(args[2]),
                "z_bc": float(args[3]),
                "tx": float(args[4]),
                "ty": float(args[5]),
                "tz": float(args[6]),
                "p_distortion": [float(v) for v in args[7:7 + 11]],
            }
        return
    if key.startswith("RingRadii_Det"):
        try:
            det_id = int(key[len("RingRadii_Det"):])
        except ValueError:
            return
        if len(args) >= 2:
            cfg.RingRadiiPerDet.setdefault(det_id, {})[
                int(float(args[0]))
            ] = float(args[1])
        return
    if key.startswith("EtaCoverage_Det"):
        try:
            det_id = int(key[len("EtaCoverage_Det"):])
        except ValueError:
            return
        if len(args) >= 3:
            cfg.EtaCoverage.setdefault(det_id, []).append((
                int(float(args[0])),
                float(args[1]),
                float(args[2]),
            ))
        return

    if key in ("RingNumbers",):
        cfg.RingNumbers.append(int(args[0]))
        return
    if key in ("RingRadii",):
        ring_radii_user.append(float(args[0]))
        return
    if key in ("RingsToExclude", "RingsToExcludeFraction"):
        cfg.RingsToReject.append(int(args[0]))
        return
    if key == "OmegaRange":
        cfg.OmegaRanges.append((float(args[0]), float(args[1])))
        return
    if key == "BoxSize":
        cfg.BoxSizes.append((float(args[0]), float(args[1]),
                             float(args[2]), float(args[3])))
        return
    if key in ("LatticeParameter", "LatticeConstant"):
        if len(args) >= 6:
            cfg.LatticeConstant = (float(args[0]), float(args[1]),
                                   float(args[2]), float(args[3]),
                                   float(args[4]), float(args[5]))
        else:
            a = float(args[0])
            cfg.LatticeConstant = (a, a, a, 90.0, 90.0, 90.0)
        return
    if key in _FLOAT_KEYS:
        attr = _FLOAT_ALIASES.get(key, key)
        setattr(cfg, attr, float(args[0]))
        return
    if key in _INT_KEYS:
        setattr(cfg, key, int(args[0]))
        return
    if key in _STR_KEYS:
        setattr(cfg, key, args[0])
        return
    # Unknown key: let it pass silently — paramstest.txt carries many keys
    # used only by other binaries.
    if key in {
        "BeamSize", "StepsizePos", "StepsizeOrient", "StepSizeOrient",
        "MinMatchesToAcceptFrac", "Completeness", "UseFriedelPairs",
        "RingThresh", "Width", "tx", "ty", "tz",
        "p0", "p1", "p2", "p3", "p4", "p5",
    }:
        return
    warnings.warn(
        f"midas_fit_grain: unrecognized paramstest.txt key '{key}'", stacklevel=3,
    )
