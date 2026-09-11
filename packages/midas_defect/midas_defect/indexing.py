"""The middle of the chain: orientation → predicted spots → assignment → audit.

`ingest` gets you a spot list. `seed_index` gets you an orientation. `completeness`
audits the result. Between them sits the step that says **which observed spot is
which reflection**, and until now that lived in a driver script.

Nothing here re-implements a forward model or a matcher. It wires up:

===============================================  ==========================================
`midas_diffract.hkls_for_forward_model`          the reflection universe, from the real
                                                 space group, expanded to Laue equivalents
`midas_diffract.HEDMForwardModel`                prediction — pixel-exact against the
                                                 canonical C reference simulators
`midas_diffract.SpotAssigner`                    matching, in **split** (2θ, η, ω) tolerances
`midas_defect.seed_index`                        the orientation, if you do not have one
`midas_defect.completeness`                      the four-way audit
===============================================  ==========================================

Two conventions this module is responsible for getting right
------------------------------------------------------------
**Tilts.** `HEDMGeometry` ignores detector tilts in FF mode by default, because FF/pf
workflows apply a DetCor correction at peak-finding time and applying them twice is
worse than not at all. Spots from `midas_defect.ingest` are **raw centroids with no
DetCor**, so this module sets ``apply_tilts=True`` whenever any tilt is non-zero, and
says so in the result. Getting this wrong is silent.

**Split tolerances.** A single scalar match radius mixes a radial pixel error with an
angular error set by the ω step; splitting them once took a real solution from 7 to 12
reflections without loosening the cell. :func:`assign_spots` therefore requires
per-channel tolerances and refuses a lone scalar.
"""
from __future__ import annotations
import dataclasses
from midas_hkls import Lattice

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .completeness import audit_completeness, window_from_residuals
from .geometry import Geometry

__all__ = ["IndexResult", "ConventionScan", "mat_to_euler", "build_forward_model",
           "observed_coords", "predict_spots", "assign_spots",
           "resolve_conventions", "index_from_cloud"]


# ---------------------------------------------------------------------------
# orientation plumbing
# ---------------------------------------------------------------------------

def mat_to_euler(U: np.ndarray) -> np.ndarray:
    """Crystal→sample rotation matrix → ZXZ Bunge Euler angles (radians).

    Inverse of ``HEDMForwardModel.euler2mat``, which delegates to
    ``midas_stress.euler_to_orient_mat`` — so this is the inverse of the
    MIDAS-canonical convention, and ``test_indexing.py`` pins the round trip
    against it rather than against a formula.
    """
    U = np.asarray(U, float).reshape(3, 3)
    Phi = math.acos(max(-1.0, min(1.0, U[2, 2])))
    if abs(math.sin(Phi)) < 1e-10:                      # gimbal: phi1+phi2 only
        return np.array([math.atan2(U[0, 1], U[0, 0]), Phi, 0.0])
    return np.array([math.atan2(U[0, 2], -U[1, 2]), Phi,
                     math.atan2(U[2, 0], U[2, 1])])


# ---------------------------------------------------------------------------
# geometry / model construction
# ---------------------------------------------------------------------------

def build_forward_model(crystal, geom: Geometry, *,
                        d_min: Optional[float] = None,
                        two_theta_max_deg: Optional[float] = None,
                        min_eta_deg: float = 0.0,
                        apply_tilts: Optional[bool] = None,
                        device: str = "cpu"):
    """Build a `HEDMForwardModel` for this crystal and geometry.

    ``apply_tilts`` defaults to **True whenever any tilt is non-zero**, which is
    correct for raw `ingest` centroids. Pass False explicitly if your spot
    positions were already DetCor-corrected at peak-finding time.

    Returns ``(model, hkls_int, apply_tilts_used)``.
    """
    from midas_diffract import HEDMGeometry
    from midas_diffract.forward import HEDMForwardModel
    from midas_diffract.hkls import hkls_for_forward_model

    if d_min is None and two_theta_max_deg is None:
        raise ValueError("supply d_min or two_theta_max_deg — the reflection "
                         "universe must be bounded deliberately, not by a default")

    tilted = any(abs(v) > 0 for v in (geom.tx_deg, geom.ty_deg, geom.tz_deg))
    if apply_tilts is None:
        apply_tilts = tilted

    hkls_cart, thetas, hkls_int = hkls_for_forward_model(
        crystal.space_group, crystal.lattice,
        wavelength_A=geom.wavelength_A,
        two_theta_max_deg=two_theta_max_deg, d_min=d_min)

    hgeom = HEDMGeometry(
        Lsd=geom.lsd_um, y_BC=geom.bcy_px, z_BC=geom.bcz_px, px=geom.px_um,
        omega_start=geom.omega_first_deg, omega_step=geom.omega_step_deg,
        n_frames=geom.n_frames, n_pixels_y=geom.n_pix_y, n_pixels_z=geom.n_pix_z,
        min_eta=min_eta_deg, wavelength=geom.wavelength_A,
        tx=geom.tx_deg, ty=geom.ty_deg, tz=geom.tz_deg,
        wedge=geom.wedge_deg, apply_tilts=bool(apply_tilts))

    model = HEDMForwardModel(hkls_cart, thetas, hgeom, hkls_int=hkls_int,
                             device=torch.device(device))
    return model, hkls_int.detach().cpu().numpy().astype(int), bool(apply_tilts)


def observed_coords(row, col, omega_deg, geom: Geometry) -> torch.Tensor:
    """Observed spots → ``(N, 3)`` of (2θ, η, ω) in **radians**.

    Matches the forward model's convention, verified against it: with
    ``Y = -(col - y_BC)·px`` and ``Z = (row - z_BC)·px``, η = atan2(-Y, Z), which
    is the inverse of the ``Y = -R sin η, Z = R cos η`` the transform emits.
    """
    row = np.asarray(row, float); col = np.asarray(col, float)
    Y = -(col - geom.bcy_px) * geom.px_um
    Z = (row - geom.bcz_px) * geom.px_um
    R = np.hypot(Y, Z)
    two_theta = np.arctan2(R, geom.lsd_um)
    eta = np.arctan2(-Y, Z)
    return torch.tensor(np.stack([two_theta, eta,
                                  np.radians(np.asarray(omega_deg, float))], 1))


def predict_spots(model, U: np.ndarray, *, position=(0.0, 0.0, 0.0)):
    """Predict every reflection for one orientation. Returns `SpotDescriptors`."""
    eul = torch.tensor(mat_to_euler(U)).reshape(1, 1, 3)
    pos = torch.tensor(np.asarray(position, float)).reshape(1, 3)
    return model(eul, pos)


# ---------------------------------------------------------------------------
# assignment
# ---------------------------------------------------------------------------

def assign_spots(spots, obs_xyw: torch.Tensor, *,
                 max_two_theta_rad: float,
                 max_eta_rad: float,
                 max_omega_rad: float,
                 one_to_one: bool = True):
    """Match predicted to observed with **split** per-channel tolerances.

    Deliberately has no single-scalar mode: the three channels carry physically
    different errors (radial pixel, azimuthal pixel, ω step) and one number
    cannot express them. Delegates to `midas_diffract.losses.SpotAssigner`,
    which also wraps the periodic channels.

    Returns ``(pred_flat_index, obs_index, n_matched)``.
    """
    from midas_diffract.losses import SpotAssigner
    for nm, v in (("max_two_theta_rad", max_two_theta_rad),
                  ("max_eta_rad", max_eta_rad), ("max_omega_rad", max_omega_rad)):
        if v is None or v <= 0:
            raise ValueError(f"{nm} must be a positive tolerance")

    pred = torch.stack([spots.two_theta, spots.eta, spots.omega], dim=-1)
    assigner = SpotAssigner(obs_xyw.to(pred.dtype))
    _, _, pred_idx = assigner.assign(
        pred, spots.valid,
        max_two_theta=max_two_theta_rad, max_eta=max_eta_rad,
        max_omega=max_omega_rad, one_to_one=one_to_one)

    # recover which observation each kept prediction took
    flat = pred.reshape(-1, 3)[pred_idx]
    obs = obs_xyw.to(pred.dtype)
    def _wrap(d):
        return (d + math.pi) % (2 * math.pi) - math.pi
    d = torch.stack([(flat[:, 0:1] - obs[None, :, 0]).abs() / max_two_theta_rad,
                     _wrap(flat[:, 1:2] - obs[None, :, 1]).abs() / max_eta_rad,
                     _wrap(flat[:, 2:3] - obs[None, :, 2]).abs() / max_omega_rad],
                    dim=-1).norm(dim=-1)
    obs_idx = d.argmin(dim=1)
    return pred_idx.cpu().numpy(), obs_idx.cpu().numpy(), int(len(pred_idx))


# ---------------------------------------------------------------------------
# conventions, and the end-to-end call
# ---------------------------------------------------------------------------

@dataclass
class ConventionScan:
    """Which ω sign the data prefers, and by how much — the evidence, not a verdict."""
    best_omega_sign: int
    table: List[Dict]
    min_assigned: int = 8             # a ratio between tiny counts is not a verdict

    @property
    def decisive(self) -> bool:
        """True when the winner more than doubles the runner-up AND reaches ``min_assigned``.

        The ratio alone was not enough. On a held-out La3Ni2O7 position (2026-09-10) the
        bright-core seeds were gasket and anvil spots: the scan returned -1 at 4 : 1,
        "decisive" by the ratio, and wrong -- the crystal's own reflections gave +1 at
        38 : 0. Feed the scan seedable spots (non-powder, non-stationary), and treat a
        winner below the floor as no verdict.
        """
        n = sorted((r["n_assigned"] for r in self.table), reverse=True)
        return len(n) > 1 and n[0] >= self.min_assigned and n[0] >= 2 * max(n[1], 1)

    def __str__(self) -> str:
        body = ", ".join(f"omega {r['omega_sign']:+d}: {r['n_assigned']}"
                         for r in self.table)
        return (f"omega sign {self.best_omega_sign:+d} ({body})"
                f"{'' if self.decisive else '  -- NOT DECISIVE, do not adopt blindly'}")


@dataclass
class IndexResult:
    """One orientation, its assignment, and the audit that judges it."""
    U: np.ndarray
    a: float
    c: float
    n_assigned: int
    assigned_hkl: List[Tuple[int, int, int]]
    audit: object
    convention: ConventionScan
    apply_tilts: bool
    median_residual_px: float
    tolerances: Dict[str, float] = field(default_factory=dict)
    # The cell the PREDICTION actually used. `a`/`c` above come from the seed
    # refinement and are kept for compatibility; these come from the converged
    # fit to the domain's own reflections. `b` did not exist before, so an
    # orthorhombic cell could not be reported at all.
    b: Optional[float] = None
    #: ``(a, b, c)`` the forward model was actually built from, or None if the
    #: cell never converged and the nominal crystal was used.
    cell_converged: Optional[Tuple[float, float, float]] = None
    #: How many reflections the converged fit used.
    n_cell_reflections: int = 0

    def __str__(self) -> str:
        return (f"{self.n_assigned} reflections assigned | {self.audit} | "
                f"a={self.a:.4f} c={self.c:.4f} | residual "
                f"{self.median_residual_px:.2f} px | {self.convention}")


def resolve_conventions(cloud_q, intensity, row, col, frame, crystal,
                        geom: Geometry, *, d_min: float,
                        tolerances: Dict[str, float],
                        omega_signs: Sequence[int] = (+1, -1),
                        n_bright: int = 30,
                        device: str = "cpu") -> ConventionScan:
    """Scan the ω sign, returning the score for every option, not just the winner.

    A tie is information: :attr:`ConventionScan.decisive` reports whether the
    winner is actually separated from the runner-up. Adopting a convention that
    won by one reflection is how a mirrored reciprocal space survives.
    """
    from .seed_index import find_seed_orientation

    rows = []
    for sign in omega_signs:
        qs = cloud_q(sign)
        res = find_seed_orientation(qs[:, 0], qs[:, 1], qs[:, 2], intensity,
                                    crystal=crystal, n_bright=n_bright,
                                    tol_q_rel=0.02, tol_angle_deg=3.0,
                                    refine_lattice=True)
        model, _, _ = build_forward_model(crystal, geom, d_min=d_min, device=device)
        spots = predict_spots(model, np.asarray(res.U))
        obs = observed_coords(row, col,
                              geom.omega_first_deg + geom.omega_step_deg * np.asarray(frame),
                              geom)
        _, _, n = assign_spots(spots, obs,
                               max_two_theta_rad=tolerances["two_theta"],
                               max_eta_rad=tolerances["eta"],
                               max_omega_rad=tolerances["omega"])
        rows.append({"omega_sign": int(sign), "n_assigned": n,
                     "a": float(res.a), "c": float(res.c)})
    best = max(rows, key=lambda r: r["n_assigned"])
    return ConventionScan(best_omega_sign=best["omega_sign"], table=rows)


def index_from_cloud(q_sample: np.ndarray, intensity: np.ndarray,
                     row: np.ndarray, col: np.ndarray, frame: np.ndarray,
                     crystal, geom: Geometry, mask: np.ndarray, *,
                     d_min: float,
                     max_two_theta_rad: float,
                     max_eta_rad: float,
                     max_omega_rad: float,
                     n_bright: int = 30,
                     apply_tilts: Optional[bool] = None,
                     convention: Optional[ConventionScan] = None,
                     sigma_rtn: Optional[tuple] = None,
                     tol_sigma: Optional[float] = None,
                     device: str = "cpu") -> IndexResult:
    """Cloud + spot list → orientation, assignment and completeness audit.

    The one call a per-raster-point loop should make. Pass ``convention`` from
    :func:`resolve_conventions` if the ω sign is not already settled; this
    function does **not** silently choose one for you.

    ``sigma_rtn`` (radial, transverse, normal residual budget, 1/Å) and ``tol_sigma`` reach the cell
    convergence. Left None, `refine_to_convergence` runs on its defaults, which are La3Ni2O7's measured
    budget — measure yours (`window_from_residuals`) and pass it.

    Every tolerance is required and none has a default — see the module
    docstring for why a single scalar will not do.
    """
    from .seed_index import find_seed_orientation

    res = find_seed_orientation(q_sample[:, 0], q_sample[:, 1], q_sample[:, 2],
                                intensity, crystal=crystal, n_bright=n_bright,
                                tol_q_rel=0.02, tol_angle_deg=3.0,
                                refine_lattice=True)
    U = np.asarray(res.U)

    # The refined cell used to be DISCARDED here: `find_seed_orientation` fits
    # (U, a, c) against the seed's matched pairs and this function then built the
    # forward model from the NOMINAL `crystal`, so every prediction -- and hence
    # the completeness audit -- used the input cell. Worse, the seed fits against
    # a handful of bright cores only, and nothing re-fitted the cell to the
    # larger set that `assign_spots` goes on to match.
    #
    # Converge the cell in q-space first, on the domain's OWN reflections, then
    # predict from that. `refine_to_convergence` re-matches on each iteration and
    # refits on the converged set, so the reported cell is the fit to the
    # reflections it actually used. The full seed cell is passed (not just a/c):
    # below ~2 % splitting it makes no difference, above ~3 % a tetragonal seed
    # claims nothing at all -- see `rows.refine_to_convergence`.
    # Pass the crystal's OWN space group. refine_to_convergence defaults to 139
    # (La3Ni2O7, I-centring), and omitting it here applied I4/mmm extinctions to the
    # cell convergence of EVERY other crystal indexed through this function -- the
    # F-vs-I centring error that once halved S5 indexing. Found 2026-09-10.
    from .rows import refine_to_convergence
    lat0 = crystal.lattice
    conv = refine_to_convergence(
        q_sample, U, a0=lat0.a, b0=lat0.b, c0=lat0.c,
        alpha0=lat0.alpha, beta0=lat0.beta, gamma0=lat0.gamma,
        min_reflections=max(5, min(8, len(q_sample)//4)), space_group_number=int(crystal.space_group.number),
        **{k: v for k, v in (("sigma_rtn", sigma_rtn), ("tol_sigma", tol_sigma)) if v is not None})
    if conv is not None:
        U = np.asarray(conv.lat.U) if hasattr(conv.lat, "U") else U
        crystal = dataclasses.replace(
            crystal, lattice=Lattice(a=conv.lat.a, b=conv.lat.b, c=conv.lat.c,
                                     alpha=conv.lat.alpha, beta=conv.lat.beta,
                                     gamma=conv.lat.gamma))
        refined_cell = (conv.lat.a, conv.lat.b, conv.lat.c)
        n_converged = int(conv.claim.sum())
    else:
        refined_cell, n_converged = None, 0

    model, hkls_int, tilts_used = build_forward_model(
        crystal, geom, d_min=d_min, apply_tilts=apply_tilts, device=device)
    spots = predict_spots(model, U)

    omega_deg = geom.omega_first_deg + geom.omega_step_deg * np.asarray(frame, float)
    obs = observed_coords(row, col, omega_deg, geom)
    pred_idx, obs_idx, n = assign_spots(
        spots, obs, max_two_theta_rad=max_two_theta_rad,
        max_eta_rad=max_eta_rad, max_omega_rad=max_omega_rad)

    M = hkls_int.shape[0]
    y = spots.y_pixel.reshape(-1).detach().cpu().numpy()
    z = spots.z_pixel.reshape(-1).detach().cpu().numpy()
    fr = spots.frame_nr.reshape(-1).detach().cpu().numpy()
    ok = spots.valid.reshape(-1).detach().cpu().numpy() > 0.5

    hkl_flat = np.tile(hkls_int, (y.size // M, 1))
    pred_ome = geom.omega_first_deg + geom.omega_step_deg * fr

    assigned = [tuple(int(v) for v in hkl_flat[i]) for i in pred_idx]
    r_px = [float(np.hypot(row[o] - z[p], col[o] - y[p]))
            for p, o in zip(pred_idx, obs_idx)]
    r_om = [float(abs(omega_deg[o] - pred_ome[p]))
            for p, o in zip(pred_idx, obs_idx)]
    if not r_px:
        raise RuntimeError(
            "no reflection was assigned. Check the omega sign (resolve_conventions), "
            "the tilt convention, and that the tolerances are not tighter than the "
            "geometry supports.")
    w_px, w_om = window_from_residuals(r_px, r_om, 90.0)

    audit = audit_completeness(
        predicted_hkl=hkl_flat[ok], predicted_row=z[ok], predicted_col=y[ok],
        predicted_omega_deg=pred_ome[ok],
        observed_row=np.asarray(row, float), observed_col=np.asarray(col, float),
        observed_omega_deg=omega_deg, assigned_hkl=assigned, mask=mask,
        window_px=w_px, window_omega_deg=w_om, observed_intensity=intensity)

    return IndexResult(
        U=U, a=float(refined_cell[0]) if refined_cell else float(res.a),
        c=float(refined_cell[2]) if refined_cell else float(res.c),
        b=float(refined_cell[1]) if refined_cell else None,
        cell_converged=refined_cell, n_cell_reflections=n_converged,
        n_assigned=n, assigned_hkl=assigned,
        audit=audit,
        convention=convention or ConventionScan(1, [{"omega_sign": 1,
                                                     "n_assigned": n}]),
        apply_tilts=tilts_used, median_residual_px=float(np.median(r_px)),
        tolerances={"two_theta": max_two_theta_rad, "eta": max_eta_rad,
                    "omega": max_omega_rad, "window_px": w_px,
                    "window_omega_deg": w_om})
