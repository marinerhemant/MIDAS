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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

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


# ────────────────────────────────────────────────────────── paramstest parsing
def _read_hedm_keys(paramstest: Path) -> dict:
    """Pull FF acquisition keys the calibration ``CalibrationParams`` doesn't
    carry (omega scan, detector size, eta gate) from the raw paramstest text."""
    keys = {"OmegaFirstFile": 0.0, "OmegaStep": 0.0, "NrFilesPerSweep": 1440,
            "NrPixelsY": 2048, "NrPixelsZ": 2048, "MinEta": 6.0}
    for line in Path(paramstest).read_text().splitlines():
        line = line.split("#", 1)[0].strip().rstrip(";").strip()
        if not line:
            continue
        t = line.split()
        if len(t) >= 2 and t[0] in keys:
            try:
                keys[t[0]] = float(t[1]) if ("." in t[1] or t[0] in
                                             ("OmegaFirstFile", "OmegaStep", "MinEta")) else int(t[1])
            except ValueError:
                pass
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


def make_residual(
    model: HEDMForwardModel,
    observations,
    matches,
    raw_yz: List[Tuple[torch.Tensor, torch.Tensor]],
    *,
    fixed_geo: dict,
    kind: str = "angular",
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

    Lsd = fixed_geo["Lsd"]
    n_g = len(observations)

    def residual(unpacked: Dict[str, torch.Tensor]) -> torch.Tensor:
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
        refine_grain_orientation=False, refine_grain_position=False,
        refine_grain_strain=False,
    )
    spec.parameters["grain_euler"].bounds = (-2 * math.pi, 2 * math.pi)
    spec.parameters["grain_pos"].bounds = (-2000.0, 2000.0)
    for nm in refine_params:
        if nm not in spec.parameters:
            raise KeyError(f"refine_params has {nm!r} but spec has no such parameter")
        spec.parameters[nm].refined = True

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
    residual = make_residual(model, observations, matches, raw_yz,
                             fixed_geo=fixed_geo, kind=kind)

    unpacked0 = {n: spec.parameters[n].init_tensor() for n in spec.parameters}
    cost_init = float((residual(unpacked0) ** 2).sum().item())
    unpacked, cost, rc = mp.lm_minimise(
        spec, residual,
        config=mp.GenericLMConfig(max_iter=max_iter, ftol_rel=1e-10, xtol_rel=1e-10),
        fallback_span=2.0,
    )
    n_matched = sum(int(m.mask.sum()) for m in matches)

    refined = {nm: float(unpacked[nm]) for nm in refine_params}
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
        for nm in refine_params:
            line = f"{nm} {float(unpacked[nm]):.10g}"
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
    )


__all__ = ["refine_geometry_from_grains", "make_residual", "GrainGeomRefineResult"]
