"""Lightweight grain-based geometry refinement — recover ``tx`` (and ``Wedge``)
that powder calibration is blind to.

Powder rings are azimuthally symmetric, so a rotation of the detector about the
beam (``tx``) leaves them invariant — calibrant data cannot constrain ``tx``.
Single-crystal grain spots can: empirically (verified on real Ni FF data) ``tx``
is a ~1:1 rotation of the azimuth η with negligible effect on the radius R. So:

  * ``tx`` lives on the **observation** side. We re-run MIDAS's detector
    correction (``midas_calibrate.geometry_torch.pixel_to_REta_torch``) on the
    RAW spot pixels with a *trial* ``tx`` — η of the corrected spot moves with
    ``tx``, differentiably. (SpotMatrix DetectorHor/DetectorVert are raw;
    YLab/ZLab are DetCor'd with the pipeline's ``tx=0``.)
  * ``Wedge`` (rotation-axis tilt) lives on the **forward** side — it changes
    predicted ω/η in the diffraction model.
  * The forward model runs in *ideal* (tilt-removed) space (``apply_tilts=
    False``), matching the DetCor'd observations.

Because ``tx`` only moves η, the loss must be **η-sensitive**: ``kind="angular"``
(the full 3D (2θ, η, ω) residual) is the default. A radial/pixel loss is
structurally blind to ``tx``. ``tx`` identifiability rests on the ω-coupling
across *multiple* grains breaking the ``tx`` ↔ per-grain-orientation degeneracy;
refine several grains together.
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

LOG = logging.getLogger(__name__)

#: Geometry scalars that act directly on the *predicted* side, or on the
#: observed side by a rotation we can apply to the stored YLab/ZLab. These are
#: refinable without recomputing the observations from raw pixels.
_DIRECT = frozenset({"tx", "Wedge", "Lsd"})

#: Geometry scalars that were folded into ``SpotMatrix`` YLab/ZLab when the
#: pipeline detector-corrected them. Thawing any of these switches the residual
#: to the RAW-PIXEL path, which recomputes the observed (Y,Z) from
#: ``det_hor``/``det_vert`` at the trial geometry. Validated on 20-ID Varex
#: data: re-deriving at the known geometry reproduces the stored eta to
#: 4e-5 deg and the stored position to a 2.3 um median -- the residue being the
#: Stage-4 residual spline, which the analytic transform does not carry.
_NEEDS_RAW = frozenset({"BC_y", "ty", "tz"})

#: Distortion coefficients. Also raw-path, also validated by the same check.
#: Frozen by default and best left so: a powder calibrant with thousands of
#: ring points constrains them far better than a few thousand grain spots.
_DISTORTION = frozenset(["iso_R2", "iso_R4", "iso_R6"]
                        + [f"a{i}" for i in range(1, 7)]
                        + [f"phi{i}" for i in range(1, 7)])

#: Everything the caller may thaw.
REFINABLE = _DIRECT | _NEEDS_RAW | _DISTORTION

#: Still refused, with the reason.
_NO_LEVER = {
    "BC_z": "A vertical beam-centre shift is degenerate with a global shift "
            "of the grain Z positions, which is exactly the coordinate FF "
            "determines worst. Refine BC_y and leave BC_z at the powder value.",
}

import midas_peakfit as mp
from midas_peakfit import Parameter
from midas_calibrate.params import CalibrationParams as V1Params
from midas_calibrate.geometry_torch import pixel_to_REta_torch
from midas_diffract import HEDMForwardModel
from midas_diffract.forward import HEDMGeometry
from midas_diffract.hkls import hkls_for_forward_model
from midas_hkls import Lattice, SpaceGroup
from torch.func import functional_call

from .grain_observations import (
    load_phase2_grains_and_spots,
    load_ring_two_theta,
    build_observations_and_matches,
)
from .spec import build_joint_spec

_DEG2RAD = math.pi / 180.0


# ───────────────────────────────────────────────────────────────────── result
@dataclass
class GrainGeomRefineResult:
    refined: Dict[str, float]                 # refined geometry scalars (tx, Wedge, …)
    cost_init: float
    cost_final: float
    rc: str
    n_grains: int
    n_spots_matched: int
    paramstest_out: Optional[Path] = None
    unpacked: Dict[str, torch.Tensor] = field(default_factory=dict)
    #: Refined names that finished ON a bound. A bound is not a measurement:
    #: it means the fit ran out of room, which on this objective has always
    #: meant the parameter was not constrained by the data.
    at_bounds: List[str] = field(default_factory=list)
    #: Conditioning complaints raised before the fit (too few grains for what
    #: was asked). Advisory; the fit still runs.
    conditioning: List[str] = field(default_factory=list)

    @property
    def trustworthy(self) -> bool:
        """False when anything hit a bound. Cheap gate for scripted callers."""
        return not self.at_bounds


# ────────────────────────────────────────────────────────── paramstest parsing
#: Keys that mean "omega of the first frame". MIDAS FF parameter files —
#: including everything ``ff_paramstest_from_auto_result`` writes — say
#: ``OmegaStart``; only some older hand-written files say ``OmegaFirstFile``.
_OMEGA_START_ALIASES = ("OmegaFirstFile", "OmegaStart")

#: Keys that mean "how many frames in the sweep". NOT ``NrFilesPerSweep``,
#: which counts *files* and is legitimately 1 on a one-file-per-sweep scan.
_N_FRAMES_ALIASES = ("NrFrames", "nFrames")


def _load_residual_map(v1, paramstest: Path, n_y: int, n_z: int, *, device, dtype):
    """Load the Stage-4 residual-correction map named by the paramstest.

    Returns ``None`` when the file names no map. A named-but-unusable map is an
    error rather than a silent skip: dropping it leaves a systematic in the
    residual exactly where a tilt or beam-centre fit would absorb it.
    """
    path = str(getattr(v1, "ResidualCorrectionMap", "")
               or (getattr(v1, "extra", {}) or {}).get("ResidualCorrectionMap", "")
               or "").strip()
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = Path(paramstest).parent / p
    if not p.exists():
        raise FileNotFoundError(
            f"{paramstest} names ResidualCorrectionMap {path!r}, which does not "
            "exist. The pipeline applied that map when it wrote YLab/ZLab, so "
            "refining without it leaves its signature in the residual.")
    arr = np.fromfile(p, dtype=np.float64)
    if arr.size != n_y * n_z:
        raise ValueError(
            f"residual map {p} has {arr.size} elements, expected "
            f"{n_y * n_z} (NrPixelsZ x NrPixelsY = {n_z} x {n_y}).")
    LOG.info("applying residual-correction map %s (%d x %d)", p, n_z, n_y)
    return torch.from_numpy(arr.reshape(n_z, n_y)).to(device=device, dtype=dtype)


def _read_hedm_keys(paramstest: Path) -> dict:
    """Pull FF acquisition keys the calibration ``CalibrationParams`` doesn't
    carry (omega scan, detector size, eta gate) from the raw paramstest text.

    Two of these have bitten before and are now resolved by alias and checked,
    because a wrong value here does not raise — it silently moves every
    predicted spot out of the matching window, and the fit then "converges"
    on almost no matched spots:

    * **omega of the first frame** — read from ``OmegaStart`` as well as
      ``OmegaFirstFile``. A file that says ``OmegaStart 180`` used to default
      this to 0.0, putting the whole predicted pattern 180 deg from the data.
    * **frame count** — read from ``NrFrames``, else derived from
      ``OmegaRange``/``OmegaStep``. It used to be taken from
      ``NrFilesPerSweep``, which counts *files*: on a one-file-per-sweep scan
      that is 1, and the model then predicted a single-frame sweep against a
      full 360 deg dataset.
    """
    keys = {"OmegaFirstFile": None, "OmegaStep": 0.0, "NrFilesPerSweep": None,
            "NrPixelsY": 2048, "NrPixelsZ": 2048, "MinEta": 6.0}
    raw: dict = {}
    for line in Path(paramstest).read_text().splitlines():
        line = line.split("#", 1)[0].strip().rstrip(";").strip()
        if not line:
            continue
        t = line.split()
        if len(t) < 2:
            continue
        raw[t[0]] = t[1:]
        if t[0] in keys:
            try:
                keys[t[0]] = float(t[1]) if ("." in t[1] or t[0] in
                                             ("OmegaFirstFile", "OmegaStep", "MinEta")) else int(t[1])
            except ValueError:
                pass

    def _f(name):
        try:
            return float(raw[name][0])
        except (KeyError, IndexError, ValueError):
            return None

    # omega of the first frame, by alias
    if keys["OmegaFirstFile"] is None:
        for nm in _OMEGA_START_ALIASES:
            v = _f(nm)
            if v is not None:
                keys["OmegaFirstFile"] = v
                break
    if keys["OmegaFirstFile"] is None:
        raise ValueError(
            f"{paramstest} sets none of {_OMEGA_START_ALIASES} — the omega of "
            "the first frame is unknown. Defaulting it to 0 silently shifts "
            "every predicted spot off the data and the fit matches almost "
            "nothing while still reporting success.")

    # frame count: explicit, else derived from the omega scan
    n_frames = None
    for nm in _N_FRAMES_ALIASES:
        v = _f(nm)
        if v is not None:
            n_frames = int(round(v))
            break
    if n_frames is None and keys["OmegaStep"]:
        rng = raw.get("OmegaRange")
        if rng and len(rng) >= 2:
            try:
                span = abs(float(rng[1]) - float(rng[0]))
                n_frames = int(round(span / abs(float(keys["OmegaStep"])))) + 1
            except ValueError:
                n_frames = None
    if n_frames is None:
        n_frames = keys["NrFilesPerSweep"]
    if not n_frames or n_frames < 10:
        raise ValueError(
            f"{paramstest} gives a sweep of {n_frames} frame(s). Set "
            f"{_N_FRAMES_ALIASES[0]}, or give OmegaRange + OmegaStep so it can "
            "be derived. (NrFilesPerSweep counts FILES, not frames, and is 1 "
            "on a one-file-per-sweep scan — using it here predicts a "
            "single-frame sweep against a full rotation and matches nothing.)")
    keys["NrFilesPerSweep"] = int(n_frames)
    return keys


def _build_forward_model(v1: V1Params, hedm: dict, grains: dict,
                         *, two_theta_max_deg: float, device, dtype):
    """Ideal-space (apply_tilts=False) HEDM forward model for the sample phase."""
    sg = SpaceGroup.from_number(grains["sg"] or v1.SpaceGroup)
    lat_vals = grains["lattice"] or tuple(v1.LatticeConstant)
    lattice = Lattice(*[float(x) for x in lat_vals[:6]])
    hkls_cart, thetas, hkls_int = hkls_for_forward_model(
        sg, lattice, wavelength_A=float(v1.Wavelength),
        two_theta_max_deg=two_theta_max_deg, expand_equivalents=True,
    )
    geom = HEDMGeometry(
        Lsd=float(v1.Lsd), y_BC=float(v1.BC_y), z_BC=float(v1.BC_z),
        px=float(v1.pxY), omega_start=float(hedm["OmegaFirstFile"]),
        omega_step=float(hedm["OmegaStep"]), n_frames=int(hedm["NrFilesPerSweep"]),
        n_pixels_y=int(hedm["NrPixelsY"]), n_pixels_z=int(hedm["NrPixelsZ"]),
        min_eta=float(hedm["MinEta"]), wavelength=float(v1.Wavelength),
        tx=0.0, ty=float(v1.ty), tz=float(v1.tz), wedge=0.0,
        flip_y=True, apply_tilts=False, multi_mode="layered",
    )
    model = HEDMForwardModel(hkls_cart, thetas, geom, hkls_int=hkls_int.float(),
                             device=device)
    return model


# ─────────────────────────────────────────────────────── tx-aware residual
def _angular_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Smallest signed angle a-b, wrapped to (-π, π]."""
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def _g_unit_pred(tth, eta, om):
    """Unit scattering vector in the sample frame from (2θ, η, ω) [rad]."""
    ct, st = torch.cos(tth / 2), torch.sin(tth / 2)
    ce, se = torch.cos(eta), torch.sin(eta)
    gx = -st
    gy = ct * (-se)
    gz = ct * ce
    co, so = torch.cos(-om), torch.sin(-om)
    return torch.stack([co * gx - so * gy, so * gx + co * gy, gz], dim=-1)


def _per_grain_internal_angle(model, observations, matches, eulers, positions,
                              lattices):
    """Mean observed-vs-predicted g-vector angle (rad) per grain at the given
    pose. Lower = better-fitting grain. Empty/unmatched grains get +inf so they
    sort last. Used to pick the best grains for tx refinement."""
    out = np.full(len(observations), np.inf, dtype=np.float64)
    for g in range(len(observations)):
        mt = matches[g]
        if int(mt.k_idx.shape[0]) == 0 or not bool(mt.mask.any()):
            continue
        spots = functional_call(
            model, {}, args=(eulers[g].view(1, 1, 3), positions[g].view(1, 1, 3)),
            kwargs={"lattice_params": lattices[g].view(1, 6)})

        def _flat(t):
            while t.dim() > 2 and t.shape[0] == 1:
                t = t.squeeze(0)
            return t.reshape(-1)
        M = spots.omega.shape[-1]
        flat = mt.k_idx * M + mt.m_idx
        gp = _g_unit_pred(_flat(spots.two_theta).gather(0, flat),
                          _flat(spots.eta).gather(0, flat),
                          _flat(spots.omega).gather(0, flat))
        go = observations[g].g_unit_lab()
        ang = torch.acos(((gp * go).sum(-1).abs()).clamp(0.0, 1.0 - 1e-12))
        m = mt.mask
        out[g] = float(ang[m].mean())
    return out


#: tx is separated from each grain's own orientation only by the omega-coupling
#: across differently-oriented grains, so it needs several of them.
_MIN_GRAINS_TX = 5
#: The distortion is a detector-wide field. A handful of grains samples a few
#: azimuths, which cannot pin fifteen harmonics; the powder calibrant, with
#: thousands of ring points at every azimuth, is where these belong.
_MIN_GRAINS_DISTORTION = 50


def _conditioning_warnings(refine_params: Sequence[str], n_grains: int) -> List[str]:
    """Complain before spending the time, when the ask outruns the data."""
    out: List[str] = []
    thawed = set(refine_params)
    if "tx" in thawed and n_grains < _MIN_GRAINS_TX:
        out.append(
            f"refining tx from {n_grains} grain(s): tx is distinguished from a "
            "grain's own orientation only by omega-coupling ACROSS grains, so "
            f"fewer than {_MIN_GRAINS_TX} makes it poorly determined.")
    d = sorted(thawed & _DISTORTION)
    if d and n_grains < _MIN_GRAINS_DISTORTION:
        out.append(
            f"refining {len(d)} distortion coefficient(s) from {n_grains} "
            "grain(s). The distortion is a detector-wide field and these "
            "grains sample only a few azimuths; expect them to run to their "
            "bounds and absorb error. Fit distortion on the powder calibrant "
            "(midas-calibrate-v2 --mode ff) instead.")
    if len(thawed) > max(1, n_grains):
        out.append(
            f"{len(thawed)} free parameters against {n_grains} grain(s) is "
            "under-determined.")
    return out


def _bounds_warnings(spec, unpacked, refine_params: Sequence[str],
                     tol_frac: float = 1e-3) -> List[str]:
    """Name any refined parameter that finished sitting on a bound.

    This has bitten three times on real data -- Wedge at +5.0 from a misread
    omega key, iso_R4 and iso_R6 at +0.05 from six grains -- and each time the
    run reported rc=0 and looked like a result. A value pressed against its
    limit is the optimiser saying it wanted to keep going, which means the data
    was not holding it.
    """
    out: List[str] = []
    for nm in refine_params:
        par = spec.parameters.get(nm)
        b = getattr(par, "bounds", None) if par is not None else None
        if not b or nm not in unpacked:
            continue
        lo, hi = float(b[0]), float(b[1])
        span = hi - lo
        if span <= 0:
            continue
        try:
            v = float(unpacked[nm])
        except (TypeError, ValueError):
            continue
        tol = tol_frac * span
        if abs(v - lo) <= tol or abs(v - hi) <= tol:
            edge = "lower" if abs(v - lo) <= tol else "upper"
            out.append(
                f"{nm} = {v:g} is ON its {edge} bound ({lo:g}, {hi:g}). That is "
                "not a measurement: the fit ran out of room, which means this "
                "parameter is not constrained by the data. Do not use it.")
    return out


def _p_coeffs_from_named_torch(named: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Assemble the v1 ``p0..p14`` vector from v2-named coefficients, keeping
    the autograd graph intact.

    ``midas_distortion.v2_coeffs_from_named`` cannot be used on the refinement
    path: it fills a numpy array with ``float(v)``, which detaches every tensor
    it is given. A thawed distortion coefficient then receives **zero
    gradient**, so the optimiser leaves it at its initial value while the
    result still reports it as refined — refined in name only. Observed on real
    ruby data: ``--refine ...,iso_R2,iso_R4,...`` moved Lsd/BC_y/tilts and left
    all fifteen harmonics exactly where they started.

    Index tables come from ``midas_distortion`` so the ordering has one source
    of truth; only the assembly is re-done here, in torch.
    """
    from midas_distortion.core import _NAME_TO_V2IDX, _PERM_V2_TO_V1
    from midas_distortion import P_COEF_NAMES

    ref = next((v for v in named.values() if torch.is_tensor(v)), None)
    dtype = ref.dtype if ref is not None else torch.float64
    device = ref.device if ref is not None else None
    zero = torch.zeros((), dtype=dtype, device=device)

    v2: List[torch.Tensor] = [zero] * len(P_COEF_NAMES)
    for nm, val in named.items():
        idx = _NAME_TO_V2IDX.get(nm)
        if idx is None or val is None:
            continue
        v2[idx] = val if torch.is_tensor(val) else torch.as_tensor(
            val, dtype=dtype, device=device)
    return torch.stack([v2[_PERM_V2_TO_V1[i]].reshape(()) for i in range(15)])


def make_residual(
    model: HEDMForwardModel,
    observations,
    matches,
    raw_yz: List[Tuple[torch.Tensor, torch.Tensor]],
    *,
    fixed_geo: dict,
    kind: str = "angular",
    observed_from_raw: bool = False,
    fixed_v2: Optional[dict] = None,
):
    """Build the LM residual closure — the FitMultipleGrains objective.

    Minimizes the **on-detector (Y,Z) position difference (µm)** between predicted
    and observed spots (``CalcAngleErrors``'s ``Error[0]=mean|diffLen|`` in
    ``FitMultipleGrains.c``). Predicted (Y,Z) come from the forward model in
    ideal space (grain pose); observed (Y,Z) are the SpotMatrix ``YLab``/``ZLab``
    — already correctly DetCor'd at ``tx=0`` — **rotated about the beam by the
    trial ``tx``** (tx is a pure on-detector rotation; R/2θ and ω are invariant).
    The spot association (``matches``) only pairs spots; the loss is the (Y,Z)
    distance. ``Wedge`` is injected into the forward model via ``functional_call``.

    NB: ``raw_yz`` / ``kind`` are accepted for signature stability but unused —
    re-deriving observed (R,η) from raw pixels gave a flipped-η / broken-2θ
    convention mismatch, so we use the pipeline's own YLab/ZLab instead.
    """
    from midas_calibrate.geometry_torch import build_tilt_matrix_torch

    Lsd_fixed = fixed_geo["Lsd"]
    n_g = len(observations)
    _v2_names = ["iso_R2", "iso_R4", "iso_R6"] + \
                [x for i in range(1, 7) for x in (f"a{i}", f"phi{i}")]

    def _observed_from_raw(g, unpacked, tx, Lsd):
        """Recompute observed (Y,Z) from RAW pixels at the trial geometry.

        Used when BC_y / ty / tz / distortion are thawed: those were folded
        into the stored YLab/ZLab, so they only become levers if we redo the
        detector correction ourselves. ``tx`` enters here too, so the caller
        must NOT also rotate the result.

        Delegates to ``midas_transforms``' ``apply_tilt_distortion`` — the same
        function the pipeline's own transforms stage uses — so this reproduces
        the stored YLab/ZLab exactly, **including the Stage-4 residual spline**.
        Rolling the transform by hand here instead left a 2.3 um median offset
        (the spline), which would have biased any tilt or beam-centre fit.
        """
        from midas_transforms.fit_setup.transform import apply_tilt_distortion
        hor, ver = raw_yz[g]
        if any(nm in unpacked for nm in _v2_names):
            named = {nm: unpacked[nm] if nm in unpacked else fixed_v2[nm]
                     for nm in _v2_names}
            p_arr = _p_coeffs_from_named_torch(named)
        else:
            p_arr = fixed_geo["p_coeffs"]
        return apply_tilt_distortion(
            hor, ver,
            Lsd=Lsd,
            BC_y=unpacked.get("BC_y", fixed_geo["BC_y"]),
            BC_z=unpacked.get("BC_z", fixed_geo["BC_z"]),
            tx=tx,
            ty=unpacked.get("ty", fixed_geo["ty"]),
            tz=unpacked.get("tz", fixed_geo["tz"]),
            p_coeffs=p_arr,
            px=fixed_geo["px"], rho_d=fixed_geo["RhoD"],
            residual_corr_map=fixed_geo.get("residual_corr_map"),
        )

    def residual(unpacked: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Lsd is thawable (it scales R_pred); falls back to the frozen value.
        Lsd = unpacked.get("Lsd", Lsd_fixed)
        tx = unpacked.get("tx", torch.zeros((), dtype=torch.float64))
        z = torch.zeros_like(tx)
        # 2-D rotation (about the beam) the trial tx applies to the on-detector
        # (Y,Z); ty/tz already baked into YLab/ZLab, tx applied incrementally.
        T = build_tilt_matrix_torch(tx, z, z)
        R11, R12, R21, R22 = T[1, 1], T[1, 2], T[2, 1], T[2, 2]
        overrides = {}
        if "Wedge" in unpacked:
            overrides["wedge"] = unpacked["Wedge"].reshape(()).to(model.wedge.dtype)
        eulers = unpacked["grain_euler"]
        positions = unpacked["grain_pos"]
        lattices = unpacked["grain_lattice"]
        pieces: List[torch.Tensor] = []
        for g in range(n_g):
            mt = matches[g]
            S = int(mt.k_idx.shape[0])
            if S == 0:
                continue
            spots = functional_call(
                model, overrides,
                args=(eulers[g].view(1, 1, 3), positions[g].view(1, 1, 3)),
                kwargs={"lattice_params": lattices[g].view(1, 6)},
            )

            def _flat(t):
                while t.dim() > 2 and t.shape[0] == 1:
                    t = t.squeeze(0)
                return t.reshape(-1)
            M = spots.omega.shape[-1]
            flat_idx = mt.k_idx * M + mt.m_idx
            # Predicted detector (Y,Z) in µm: R = Lsd·tan(2θ); η = atan2(-Y, Z)
            # ⇒ Y = -R·sin η, Z = R·cos η (matches the YLab/ZLab convention).
            pick_2th = _flat(spots.two_theta).gather(0, flat_idx)
            pick_eta = _flat(spots.eta).gather(0, flat_idx)
            R_pred = Lsd * torch.tan(pick_2th)
            Y_pred = -R_pred * torch.sin(pick_eta)
            Z_pred = R_pred * torch.cos(pick_eta)
            if observed_from_raw:
                # tx is applied inside the detector correction here, so do NOT
                # also rotate — that would double-count it.
                Yo_t, Zo_t = _observed_from_raw(g, unpacked, tx, Lsd)
            else:
                # Observed detector (Y,Z) (µm), rotated about the beam by tx.
                Yo = observations[g].y_lab
                Zo = observations[g].z_lab
                Yo_t = R11 * Yo + R12 * Zo
                Zo_t = R21 * Yo + R22 * Zo
            r = torch.stack([Y_pred - Yo_t, Z_pred - Zo_t], dim=-1)
            r = r * mt.mask.to(r.dtype).unsqueeze(-1)
            pieces.append(r.flatten())
        if not pieces:
            return torch.zeros(0, dtype=torch.float64)
        return torch.cat(pieces)

    return residual


# ─────────────────────────────────────────────────────────── main entry point
def refine_geometry_from_grains(
    paramstest: Path | str,
    layer_dir: Path | str,
    *,
    refine_params: Sequence[str] = ("tx", "Wedge"),
    kind: str = "angular",
    max_grains: int = 50,
    max_iter: int = 50,
    two_theta_max_deg: float = 20.0,
    refine_grain_strain: bool = True,
    refine_grain_orientation: bool = False,
    refine_grain_position: bool = False,
    fix_values: Optional[Dict[str, object]] = None,
    with_powder: bool = False,
    select: str = "internal_angle",
    out_paramstest: Optional[Path | str] = None,
    device: str = "cpu",
    dtype=torch.float64,
) -> GrainGeomRefineResult:
    """Refine ``refine_params`` (default ``tx``, ``Wedge``) from reconstructed
    grain spots, holding all other geometry fixed.

    Parameters
    ----------
    paramstest : the paramstest the pipeline ran with (``tx≈0``, full geometry).
    layer_dir  : pipeline layer dir holding ``Grains.csv`` + ``SpotMatrix.csv``
                 (and ``hkls.csv``).
    refine_params : geometry blocks to thaw. ``tx`` is refined on the
                 observation side (DetCor); ``Wedge`` on the forward side.
    kind       : ``"angular"`` (3D, η-sensitive, default) or ``"internal_angle"``.
    refine_grain_strain : free per-grain lattice too (recommended — absorbs
                 strain so it doesn't leak into geometry).
    with_powder : full-joint path (powder + grains); not yet wired (raises).
    out_paramstest : if given, write the corrected paramstest for the re-run.
    """
    if with_powder:
        raise NotImplementedError(
            "with_powder=True (full joint) is layered on this same entry point; "
            "use midas_joint_ff_calibrate.runners.run_real_phase3_joint for now.")

    layer_dir = Path(layer_dir)
    dev = torch.device(device)
    v1 = V1Params.from_file(str(paramstest))
    hedm = _read_hedm_keys(Path(paramstest))

    # The forward model needs the omega-scan + detector-size acquisition keys
    # (OmegaStep, NrFilesPerSweep, NrPixelsY/Z). The STRIPPED per-layer
    # paramstest the c-omp pipeline feeds its refiner DROPS these, so reading
    # them here silently defaults OmegaStep→0 (a degenerate, zero-width omega
    # scan) and NrPixels→2048, which invalidates EVERY predicted spot and
    # yields "matched spots=0". Pass the full FF parameter file (the one
    # ff_paramstest_from_auto_result / fit-setup wrote, with OmegaStep,
    # NrPixelsY/Z, etc.), not the layer's stripped paramstest.txt.
    if float(hedm.get("OmegaStep", 0.0)) == 0.0:
        raise ValueError(
            f"{paramstest} has no (non-zero) OmegaStep — it looks like the "
            "stripped per-layer paramstest the c-omp refiner consumes, which "
            "omits the omega-scan/detector acquisition keys the forward model "
            "needs. With OmegaStep=0 every predicted spot is invalid and tx "
            "cannot be refined. Pass the FULL FF parameter file (with "
            "OmegaStep, NrFilesPerSweep, NrPixelsY/Z) instead."
        )

    # Grains + spots.
    (grain_eulers, grain_pos, grain_lat, spots_per_grain, grains, _smatrix) = \
        load_phase2_grains_and_spots(layer_dir)
    ring_tt = load_ring_two_theta(layer_dir / "hkls.csv")

    model = _build_forward_model(v1, hedm, grains, two_theta_max_deg=two_theta_max_deg,
                                 device=dev, dtype=dtype)

    # Grain selection. ``internal_angle`` keeps the best-FITTING grains (smallest
    # mean observed-vs-predicted g-vector angle at the init pose) — far more
    # robust for tx refinement than confidence, which admits poorly-fit grains
    # whose ~tens-of-degrees residuals swamp tx's sub-degree signal.
    n_avail = len(grains["confidence"])
    if select == "internal_angle":
        pool = np.argsort(-grains["confidence"])[:min(n_avail, max(max_grains * 10, 200))]
        pool_spots = [spots_per_grain[i] for i in pool]
        pool_obs, pool_matches = build_observations_and_matches(
            model, pool_spots, grain_eulers[pool], grain_pos[pool],
            grain_lat[pool], grains["radius"][pool], ring_tt)
        ia = _per_grain_internal_angle(
            model, pool_obs, pool_matches,
            torch.from_numpy(grain_eulers[pool]).to(dtype),
            torch.from_numpy(grain_pos[pool]).to(dtype),
            torch.from_numpy(grain_lat[pool]).to(dtype))
        keep = pool[np.argsort(ia)[:max(1, min(max_grains, len(pool)))]]
    else:
        keep = np.argsort(-grains["confidence"])[:max(1, min(max_grains, n_avail))]

    grain_eulers = grain_eulers[keep]; grain_pos = grain_pos[keep]
    grain_lat = grain_lat[keep]
    spots = [spots_per_grain[i] for i in keep]
    radius = grains["radius"][keep]
    observations, matches = build_observations_and_matches(
        model, spots, grain_eulers, grain_pos, grain_lat, radius, ring_tt)

    # Per-grain RAW pixels (DetectorHor/Vert), aligned with each grain's obs.
    raw_yz: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for bag in spots:
        if "spot_id" not in bag or len(bag["spot_id"]) == 0:
            raw_yz.append((torch.zeros(0, dtype=dtype), torch.zeros(0, dtype=dtype)))
        else:
            raw_yz.append((torch.tensor(bag["det_hor"], dtype=dtype),
                           torch.tensor(bag["det_vert"], dtype=dtype)))

    # Spec: geometry params (tx, Wedge, …) + grain blocks. Freeze ALL geometry,
    # then thaw only refine_params; grain orient/pos/(strain) refined.
    spec = mp.ParameterSpec()
    # Distortion: v2-native param files carry iso_R2/a1/phi1… (the indexer/
    # refiner zero p0..p14, and V1Params stashes unknown keys in .extra). Build
    # the canonical v2 vector from v2 names OR legacy p0..p14, then map to the
    # v1-slot order pixel_to_REta_torch expects (it shims back to v2 internally).
    from midas_distortion import v2_coeffs_from_named, v2_to_v1_coeffs, P_COEF_NAMES
    _named = {nm: float(v1.extra[nm]) for nm in P_COEF_NAMES if nm in getattr(v1, "extra", {})}
    _named.update({f"p{i}": float(getattr(v1, f"p{i}")) for i in range(15)})
    _v2vec = v2_coeffs_from_named(_named)
    p_arr = torch.tensor(v2_to_v1_coeffs(_v2vec), dtype=dtype)
    spec.add(Parameter("tx", init=torch.tensor(float(v1.tx), dtype=dtype),
                       refined=False, bounds=(-5.0, 5.0)))
    spec.add(Parameter("Wedge", init=torch.tensor(float(getattr(v1, "Wedge", 0.0) or 0.0),
                       dtype=dtype), refined=False, bounds=(-5.0, 5.0)))
    # Lsd scales the predicted radius (R = Lsd·tan 2θ), so it is a genuine
    # lever on this residual and may be thawed. Frozen by default: the powder
    # calibrant constrains it far better than a few hundred grain spots do,
    # and thawing it lets Lsd trade against the grain radial positions.
    spec.add(Parameter("Lsd", init=torch.tensor(float(v1.Lsd), dtype=dtype),
                       refined=False,
                       bounds=(0.5 * float(v1.Lsd), 2.0 * float(v1.Lsd))))
    # Raw-pixel-path geometry: only levers once the observations are recomputed
    # from det_hor/det_vert (see _NEEDS_RAW). Frozen unless asked for.
    spec.add(Parameter("BC_y", init=torch.tensor(float(v1.BC_y), dtype=dtype),
                       refined=False,
                       bounds=(float(v1.BC_y) - 50.0, float(v1.BC_y) + 50.0)))
    spec.add(Parameter("ty", init=torch.tensor(float(v1.ty), dtype=dtype),
                       refined=False, bounds=(-5.0, 5.0)))
    spec.add(Parameter("tz", init=torch.tensor(float(v1.tz), dtype=dtype),
                       refined=False, bounds=(-5.0, 5.0)))
    _v2_init = {nm: float(_named.get(nm, 0.0)) for nm in _DISTORTION}
    for nm in sorted(_DISTORTION):
        lo, hi = (-180.0, 180.0) if nm.startswith("phi") else (-0.05, 0.05)
        spec.add(Parameter(nm, init=torch.tensor(_v2_init[nm], dtype=dtype),
                           refined=False, bounds=(lo, hi)))
    # Grain pose is held FIXED at the (good) MIDAS values for the tx step: a
    # free grain orientation rotates the predicted (Y,Z) pattern about the beam
    # exactly as tx rotates the observed one, so co-refining it re-absorbs tx
    # (the divergence we saw). With the pose fixed, the (Y,Z) position loss has a
    # clean minimum at the true tx (validated by a tx-cost scan). Strain/pose are
    # refined separately downstream (process-grains), not here.
    spec = build_joint_spec(
        powder_spec=spec,
        grain_eulers_init=torch.from_numpy(grain_eulers).to(dtype),
        grain_positions_init=torch.from_numpy(grain_pos).to(dtype),
        grain_lattices_init=torch.from_numpy(grain_lat).to(dtype),
        refine_grain_orientation=refine_grain_orientation,
        refine_grain_position=refine_grain_position,
        refine_grain_strain=False,
    )
    spec.parameters["grain_euler"].bounds = (-2 * math.pi, 2 * math.pi)
    spec.parameters["grain_pos"].bounds = (-2000.0, 2000.0)
    for nm in refine_params:
        if nm in _NO_LEVER:
            raise ValueError(
                f"refine_params has {nm!r}. {_NO_LEVER[nm]} Refinable here: "
                f"{', '.join(sorted(REFINABLE))} (+ the grain blocks via "
                "refine_grain_orientation / _position / _strain).")
        if nm not in spec.parameters:
            raise KeyError(
                f"refine_params has {nm!r} but spec has no such parameter. "
                f"Refinable: {', '.join(sorted(REFINABLE))}.")
        spec.parameters[nm].refined = True

    # Pin parameters to externally KNOWN values (a LaB6 lattice, a focused-beam
    # grain position). Distinct from freezing, which just keeps whatever the
    # parameter file happened to say.
    for nm, val in (fix_values or {}).items():
        if nm in refine_params:
            raise ValueError(
                f"{nm!r} is in both refine_params and fix_values — it cannot "
                "be pinned to a known value and refined at the same time.")
        if nm not in spec.parameters:
            raise KeyError(
                f"fix_values has {nm!r}; known parameters are "
                f"{', '.join(sorted(spec.parameters))}.")
        cur = spec.parameters[nm].init
        t = torch.as_tensor(val, dtype=dtype)
        if t.ndim == 1 and cur.ndim == 2 and t.shape[0] == cur.shape[1]:
            t = t.unsqueeze(0).expand_as(cur).clone()   # one row → all grains
        if t.shape != cur.shape:
            raise ValueError(
                f"fix_values[{nm!r}] has shape {tuple(t.shape)}, expected "
                f"{tuple(cur.shape)}.")
        spec.parameters[nm].init = t
        spec.parameters[nm].refined = False
        LOG.info("pinned %s to a supplied known value", nm)

    conditioning = _conditioning_warnings(refine_params, len(grain_eulers))
    for msg in conditioning:
        LOG.warning("%s", msg)

    observed_from_raw = bool(
        (set(refine_params) & (_NEEDS_RAW | _DISTORTION)))
    if observed_from_raw:
        LOG.info("recomputing observations from raw pixels (thawed: %s)",
                 ", ".join(sorted(set(refine_params) & (_NEEDS_RAW | _DISTORTION))))
    if "tx" in refine_params and refine_grain_orientation:
        LOG.warning(
            "refining tx with refine_grain_orientation=True: a free grain "
            "orientation rotates the predicted pattern about the beam exactly "
            "as tx rotates the observed one, so tx is re-absorbed and comes "
            "back ~0. Hold the pose fixed for the tx step.")

    fixed_geo = dict(
        Lsd=torch.tensor(float(v1.Lsd), dtype=dtype),
        BC_y=torch.tensor(float(v1.BC_y), dtype=dtype),
        BC_z=torch.tensor(float(v1.BC_z), dtype=dtype),
        ty=torch.tensor(float(v1.ty), dtype=dtype),
        tz=torch.tensor(float(v1.tz), dtype=dtype),
        px=torch.tensor(float(v1.pxY), dtype=dtype),
        RhoD=torch.tensor(float(v1.RhoD if v1.RhoD > 0 else v1.MaxRingRad), dtype=dtype),
        p_coeffs=p_arr,
    )
    # Stage-4 residual spline: applied, never refined. The pipeline folds it
    # into the stored YLab/ZLab, so the raw-pixel path must apply it too or it
    # sits in the residual as a systematic (2.3 um median on 20-ID Varex) and
    # biases any tilt / beam-centre fit.
    fixed_geo["residual_corr_map"] = _load_residual_map(
        v1, paramstest, int(hedm["NrPixelsY"]), int(hedm["NrPixelsZ"]),
        device=dev, dtype=dtype)
    residual = make_residual(model, observations, matches, raw_yz,
                             fixed_geo=fixed_geo, kind=kind,
                             observed_from_raw=observed_from_raw,
                             fixed_v2={nm: torch.tensor(_v2_init[nm], dtype=dtype)
                                       for nm in _DISTORTION})

    unpacked0 = {n: spec.parameters[n].init_tensor() for n in spec.parameters}
    cost_init = float((residual(unpacked0) ** 2).sum().item())
    unpacked, cost, rc = mp.lm_minimise(
        spec, residual,
        config=mp.GenericLMConfig(max_iter=max_iter, ftol_rel=1e-10, xtol_rel=1e-10),
        fallback_span=2.0,
    )
    n_matched = sum(int(m.mask.sum()) for m in matches)

    refined = {nm: float(unpacked[nm]) for nm in refine_params}
    bound_msgs = _bounds_warnings(spec, unpacked, refine_params)
    for msg in bound_msgs:
        LOG.warning("%s", msg)
    at_bounds = [m.split(" =", 1)[0] for m in bound_msgs]
    out_path = None
    if out_paramstest is not None:
        # Edit the ORIGINAL param text in place — replace/append only the
        # refined keys — so the full v2 distortion, lattice, and acquisition
        # keys carry through verbatim. (Round-tripping through V1Params drops
        # non-v1 keys like LatticeParameter → zero lattice → downstream hkl
        # failure.)
        import re as _re
        out_path = Path(out_paramstest)
        txt = Path(paramstest).read_text()
        # Which refined scalars are RELATIVE to the geometry the input
        # reconstruction already used, and so must be COMPOSED rather than
        # overwritten when writing a paramstest for a FRESH run?
        #
        # The rule follows _build_model: a scalar seeded there from ``v1`` is
        # absolute (the fit replaces it); a scalar hardcoded to 0.0 is a
        # CORRECTION applied on top of whatever the observations already carry.
        #
        #   Lsd, ty, tz, BC, distortion : seeded from v1        -> ABSOLUTE
        #   Wedge                       : geom built wedge=0.0  -> RELATIVE
        #                                 (observed omega already carries the
        #                                 pipeline's wedge correction)
        #   tx                          : DEPENDS ON THE PATH
        #       observed_from_raw=False -> geom built tx=0.0 and the trial tx
        #           ROTATES the stored YLab/ZLab, which already carry the
        #           pipeline's tx            -> RELATIVE
        #       observed_from_raw=True  -> observations are re-derived from raw
        #           pixels and tx is applied inside the detector correction
        #           ("do NOT also rotate")   -> ABSOLUTE
        #
        # Without this, iterating the tool silently DISCARDS the previous pass:
        # measured on 20-ID Au (5 grains, MinNrPx-4 spot list)
        #     pass 1 on a tx=0 recon        -> -0.158497
        #     pass 2 on the -0.1585 recon   -> -0.087265   (the residual)
        #     composed total                -> -0.245762
        # against -0.2455 from an independent ring/eta systematics fit. Writing
        # -0.087265 back would apply a THIRD of the true roll — a second pass
        # strictly worse than the first, with no error and no log line.
        _relative = {"Wedge"} | (set() if observed_from_raw else {"tx"})
        for nm in refine_params:
            value = float(unpacked[nm])
            if nm in _relative:
                prior = float(getattr(v1, nm, 0.0) or 0.0)
                if prior:
                    LOG.info(
                        "grain-tx: %s is a correction on top of the input "
                        "reconstruction — composing %.6g (already applied) + "
                        "%.6g (fitted) = %.6g", nm, prior, value, prior + value)
                value = prior + value
            line = f"{nm} {value:.10g}"
            pat = rf"(?m)^{nm}\b.*$"
            if _re.search(pat, txt):
                txt = _re.sub(pat, line, txt)
            else:
                txt += ("" if txt.endswith("\n") else "\n") + line + "\n"
        out_path.write_text(txt)

    return GrainGeomRefineResult(
        refined=refined, cost_init=cost_init, cost_final=float(cost), rc=str(rc),
        n_grains=len(observations), n_spots_matched=n_matched,
        paramstest_out=out_path, unpacked=unpacked,
        at_bounds=at_bounds, conditioning=conditioning,
    )


__all__ = ["refine_geometry_from_grains", "make_residual", "GrainGeomRefineResult"]
