"""Multi-stage progressive calibration — matches AutoCalibrateZarr's flow.

Problem: single-call ``run_calibration()`` refines only the p-coefficients
that start non-zero in the config. Leaving p5..p14 at zero means MIDASCalibrant
never fits them, producing 50-600 µε residual on detectors that actually
have non-trivial higher-order distortion.

Solution: a two-stage flow that matches ``utils/AutoCalibrateZarr.py``:

    Stage 1 — geometry only.
        tolP = 0 (distortion locked), p0..p14 all zero. Refines tilts +
        Lsd + BC only. Establishes the rigid geometry without letting
        distortion absorb tilt signal.

    Stage 2 — full model.
        Seed every active p-coefficient at 1e-4 (amplitude) or 45°
        (phase). Carry the refined Lsd/BC/tilts from Stage 1 as seeds.
        Refines all active p's simultaneously.

Active p's are controlled by ``fit_p_models`` — a comma-separated set
from {"tilt", "spherical", "dipole", "trefoil", "pentafoil5",
"hexafoil6", "all"}:

    tilt        → ty, tz (stage 1)
    spherical   → p0, p1, p2, p3
    dipole      → p7, p8 (p8 is phase in degrees)
    trefoil     → p9, p10
    pentafoil5  → p11, p12
    hexafoil6   → p13, p14
    all         → all of the above (dipole + trefoil + pentafoil5 + hexafoil6)

Paper-quality calibration on Pilatus / Varex uses ``fit_p_models='all'``.

This module is pure Python — it wraps ``run_calibration()`` and doesn't
touch C. Stage 3 (TPS residual correction map) and Stage 4 (evaluate-
only pass) from AutoCalibrateZarr are v0.2 additions; not required to
match MIDAS's paper-quality MeanStrain numbers.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from ._config import CalibrationConfig
from .calibrate import CalibrationResult, run_calibration
from .geometry import DetectorGeometry

__all__ = [
    "DEFAULT_FIT_P_MODELS",
    "ProgressiveResult",
    "calibrate_progressive",
]

DEFAULT_FIT_P_MODELS = ("tilt", "spherical", "dipole", "trefoil",
                        "pentafoil5", "hexafoil6")


@dataclass
class ProgressiveResult:
    """Output of :func:`calibrate_progressive`.

    ``final`` holds the last stage's ``CalibrationResult`` (the one
    users care about). ``stages`` is the full history; each entry is a
    ``(name, CalibrationResult)`` tuple, ordered by execution.
    """

    final: CalibrationResult
    stages: list[tuple[str, CalibrationResult]] = field(default_factory=list)

    @property
    def geometry(self) -> DetectorGeometry:
        return self.final.geometry

    @property
    def pseudo_strain(self) -> float:
        return self.final.pseudo_strain


def calibrate_progressive(
    config: CalibrationConfig,
    image: str | Path,
    *,
    work_dir: str | Path,
    fit_p_models: Iterable[str] | str = "all",
    n_iterations_stage1: int = 10,
    n_iterations_stage2: int = 30,
    n_cpus: int = 4,
    bin_dir: str | Path | None = None,
) -> ProgressiveResult:
    """Two-stage progressive calibration matching AutoCalibrateZarr.

    Parameters
    ----------
    config : CalibrationConfig
        User inputs. ``tx``, ``ty``, ``tz``, ``lsd``, ``ybc``, ``zbc`` are
        used as Stage 1 seeds; p-coefficients are ignored (Stage 1 sets
        them to zero, Stage 2 re-seeds from the model set below).
    image : path-like
        Numbered calibrant image (e.g. ``CeO2_00001.tif``).
    work_dir : path-like
        Stage 1 runs in ``work_dir / "stage1"``, Stage 2 in
        ``work_dir / "stage2"``. Both Parameters.txt and output CSVs are
        written into those subdirs.
    fit_p_models : iterable[str] | str, default "all"
        Distortion modes to enable in Stage 2. Pass ``"all"`` for
        dipole + trefoil + pentafoil5 + hexafoil6. Accepts a comma-
        separated string (``"tilt,spherical,dipole"``) or an iterable.
    n_iterations_stage1, n_iterations_stage2 : int
        Inner MIDASCalibrant iteration counts. AutoCalibrateZarr defaults
        are 10 and 30.
    n_cpus : int
        Parallelism for MIDASCalibrant.
    bin_dir : path-like, optional
        Forwarded to :func:`run_calibration`.
    """
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    modes = _normalize_models(fit_p_models)
    image_path = Path(image).resolve()

    # ------------------------------------------------------------------
    # Stage 1 — geometry only.
    # ------------------------------------------------------------------
    s1_dir = work / "stage1"
    s1_image = _stage_inputs(image_path, config, s1_dir)
    s1_cfg = _stage1_config(config)
    s1_result = run_calibration(
        s1_cfg, s1_image,
        work_dir=s1_dir, n_cpus=n_cpus, bin_dir=bin_dir,
        n_iterations=n_iterations_stage1,
    )

    # ------------------------------------------------------------------
    # Stage 2 — full model, seeded from Stage 1 geometry + default p's.
    # ------------------------------------------------------------------
    s2_dir = work / "stage2"
    s2_image = _stage_inputs(image_path, config, s2_dir)
    # Also carry Stage 1's panel shifts into Stage 2 so MIDAS converges
    # from a near-optimal panel state (critical for Pilatus mosaic fits).
    s1_panel_shifts = s1_dir / "panelshiftsCalibrant.txt"
    if s1_panel_shifts.exists():
        import shutil as _sh
        _sh.copy(s1_panel_shifts, s2_dir / "panelshiftsCalibrant.txt")
    s2_cfg = _stage2_config(config, s1_result.geometry, modes)
    s2_result = run_calibration(
        s2_cfg, s2_image,
        work_dir=s2_dir, n_cpus=n_cpus, bin_dir=bin_dir,
        n_iterations=n_iterations_stage2,
    )

    return ProgressiveResult(
        final=s2_result,
        stages=[("stage1_geometry", s1_result),
                ("stage2_distortion", s2_result)],
    )


# ---------------------------------------------------------------------------
# Per-stage input staging — copies image + dark + mask into stage work dir.
# ---------------------------------------------------------------------------

def _stage_inputs(image_path: Path, config: CalibrationConfig, stage_dir: Path) -> Path:
    """Copy image + dark + mask into ``stage_dir``; return the staged image path.

    MIDASCalibrant reads ``<Folder>/<FileStem>_<NN>.<Ext>``, and ``Folder``
    in the generated Parameters.txt is the stage work_dir — so the image
    itself plus any Dark / MaskFile referenced in the config must live
    there. We copy rather than symlink to keep each stage self-contained
    (so users can re-run a single stage without leaving dangling links).
    """
    import shutil as _sh
    stage_dir.mkdir(parents=True, exist_ok=True)

    staged_image = stage_dir / image_path.name
    if not staged_image.exists():
        _sh.copy(image_path, staged_image)

    # Dark / Mask — relative paths inherited from the config. If absolute,
    # leave them alone; if relative, stage them too.
    src_dir = image_path.parent
    for attr in ("dark_file", "mask_file"):
        rel = getattr(config, attr, "")
        if not rel or Path(rel).is_absolute():
            continue
        src = src_dir / rel
        dst = stage_dir / rel
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            _sh.copy(src, dst)

    return staged_image


# ---------------------------------------------------------------------------
# Stage config builders
# ---------------------------------------------------------------------------

def _stage1_config(user_cfg: CalibrationConfig) -> CalibrationConfig:
    """Geometry-only config: tilts + Lsd + BC, all p's at zero."""
    cfg = copy.deepcopy(user_cfg)
    for i in range(15):
        setattr(cfg, f"p{i}", 0.0)
    # Lock distortion: tolP=0, tolP4=0. Keep whatever extra_params the
    # user supplied but override the tolerance knobs. Strip any panel
    # knobs — Stage 1 runs as a monolithic fit (matches AutoCalibrateZarr).
    extra = dict(cfg.extra_params or {})
    extra.update({"tolP": 0, "tolP4": 0})
    # Zero out any p-seed the user might have set via extra_params —
    # Stage 1 must not inherit them.
    for i in range(15):
        extra.pop(f"p{i}", None)
    # Strip panel / per-panel refinement keys (monolithic fit only).
    for k in ("NPanelsY", "NPanelsZ", "PanelSizeY", "PanelSizeZ",
              "PanelGapsY", "PanelGapsZ", "FixPanelID",
              "tolShifts", "tolRotation", "PerPanelLsd",
              "PerPanelDistortion", "PanelShiftsFile"):
        extra.pop(k, None)
    # Sensible AutoCalibrateZarr defaults.
    _apply_autocal_defaults(extra)
    cfg.extra_params = extra
    return cfg


def _stage2_config(
    user_cfg: CalibrationConfig,
    stage1_geometry: DetectorGeometry,
    modes: frozenset[str],
) -> CalibrationConfig:
    """Full-model config: seeded from Stage 1 + mode-dependent p-seeds."""
    cfg = copy.deepcopy(user_cfg)

    # Carry refined rigid geometry + tilts from Stage 1.
    cfg.lsd = stage1_geometry.lsd
    cfg.ybc = stage1_geometry.ybc
    cfg.zbc = stage1_geometry.zbc
    cfg.tx = stage1_geometry.tx
    cfg.ty = stage1_geometry.ty
    cfg.tz = stage1_geometry.tz

    # CalibrationConfig doesn't expose p0..p14 as structured fields, so
    # seeds must go into ``extra_params`` (which write_params_file emits
    # verbatim). User overrides from their original extra_params take
    # precedence — only seed a p if the user hasn't set it.
    extra = dict(cfg.extra_params or {})
    seeds = _p_seeds(modes)
    for i in range(15):
        extra.setdefault(f"p{i}", seeds[i])

    # Unlock distortion — tolP / tolP4 non-zero (AutoCalibrateZarr defaults).
    extra.setdefault("tolP", 2e-3)
    extra.setdefault("tolP4", 1e-4)
    _apply_autocal_defaults(extra)
    cfg.extra_params = extra
    return cfg


def _apply_autocal_defaults(extra: dict[str, Any]) -> None:
    """Fill in calibration knobs AutoCalibrateZarr emits but aren't in
    CalibrationConfig's structured fields. Preserves user overrides via
    ``setdefault`` — only sets what the user hasn't already decided.
    """
    extra.setdefault("RBinWidth", 4)
    extra.setdefault("DoubletSeparation", 25)
    extra.setdefault("MinIndicesForFit", 5)
    extra.setdefault("TrimmedMeanFraction", 0.2)
    extra.setdefault("RemoveOutliersBetweenIters", 1)
    extra.setdefault("Wedge", 0)

    # Per-p tolerances — CRUCIAL for p-seeds to actually refine. Phases
    # need DEGREE-scale steps (45-180°), amplitudes need small REL steps
    # (~1e-3). Without these, the optimizer can't move phases at all.
    # Values mirror AutoCalibrateZarr's defaults.
    extra.setdefault("tolP1", 2e-3)
    extra.setdefault("tolP2", 2e-3)
    extra.setdefault("tolP3", 45)      # phase (degrees)
    extra.setdefault("tolP4", 2e-3)
    extra.setdefault("tolP5", 2e-3)
    extra.setdefault("tolP6", 90)      # phase
    extra.setdefault("tolP7", 1e-3)
    extra.setdefault("tolP8", 180)     # phase
    extra.setdefault("tolP9", 1e-3)
    extra.setdefault("tolP10", 180)    # phase
    extra.setdefault("tolP11", 1e-3)
    extra.setdefault("tolP12", 180)    # phase
    extra.setdefault("tolP13", 1e-3)
    extra.setdefault("tolP14", 180)    # phase


# ---------------------------------------------------------------------------
# Model-set normalisation + per-p seed rules.
# ---------------------------------------------------------------------------

_MODEL_NAMES = frozenset({"tilt", "spherical", "dipole", "trefoil",
                          "pentafoil5", "hexafoil6", "all"})


def _normalize_models(modes: Iterable[str] | str) -> frozenset[str]:
    if isinstance(modes, str):
        parts = [m.strip().lower() for m in modes.split(",") if m.strip()]
    else:
        parts = [m.strip().lower() for m in modes if m.strip()]
    if not parts:
        raise ValueError("fit_p_models is empty")
    unknown = set(parts) - _MODEL_NAMES
    if unknown:
        raise ValueError(
            f"unknown fit_p_models entries: {sorted(unknown)}. "
            f"Valid: {sorted(_MODEL_NAMES)}"
        )
    expanded = set(parts)
    if "all" in expanded:
        expanded |= {"tilt", "spherical", "dipole", "trefoil",
                     "pentafoil5", "hexafoil6"}
    return frozenset(expanded)


def _p_seeds(modes: frozenset[str]) -> dict[int, float]:
    """Per-index seed values for p0..p14 based on active modes.

    Matches AutoCalibrateZarr's seeding at lines 2048-2055: amplitudes at
    1e-4, phases at 45° for newly-unlocked harmonic terms. p0..p4 (spherical)
    seed at 1e-4 when spherical is active. p3 is actually a phase for the
    p1 cos(4η + p3) term — seeded accordingly.
    """
    seeds: dict[int, float] = {i: 0.0 for i in range(15)}
    if "spherical" in modes:
        seeds[0] = 1e-4      # p0 amplitude
        seeds[1] = 1e-4      # p1 amplitude
        seeds[2] = 1e-4      # p2 amplitude
        seeds[3] = 0.0       # p3 phase — starts at 0, optimizer will adjust
        seeds[4] = 1e-4      # p4 R⁶ amplitude
        seeds[5] = 1e-4      # p5 R⁴ amplitude
        seeds[6] = 0.0       # p6 phase for p0·cos(2η+p6)
    if "dipole" in modes:
        seeds[7] = 1e-4
        seeds[8] = 45.0
    if "trefoil" in modes:
        seeds[9] = 1e-4
        seeds[10] = 45.0
    if "pentafoil5" in modes:
        seeds[11] = 1e-4
        seeds[12] = 45.0
    if "hexafoil6" in modes:
        seeds[13] = 1e-4
        seeds[14] = 45.0
    return seeds
