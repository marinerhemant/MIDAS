"""Rods measured on the FRAME STACK — the complement to `rod_detect`.

:mod:`midas_defect.rod_detect` finds rods as lines in a q-space voxel cloud.
That is the right tool once a cloud exists. It cannot answer the questions
below, which need the frames themselves:

- how does intensity run **along** a known rod, against continuous L?
- is the rod real, or is it a powder ring, a detector artefact, or the walk
  dragging a bright pixel?
- how **wide** is the rod transverse to itself, and how much of that width is
  just the instrument?

Walk the rod where it is straight
---------------------------------
A c\\* rod is a straight line in reciprocal space. The map from q to a flat
detector goes through the Ewald sphere and is **nonlinear**, so that straight
rod is a *curve* on the detector. Walking a straight line in detector space
drifts off the rod, worst at large \\|L\\| — which is exactly where the profile is
usually being read. So parameterise it where it is straight,
``G_sample(L) = U · B · (h, k, L)`` with L continuous, solve ω for each L, and
project. Two consequences, both improvements: the sampled path is the true
curved image of the rod, and each L is read from **its own frame** — the one
where that part of the rod is in diffraction condition — rather than from a max
projection that piles 36 frames of background under every point.

Quote sigma, never a ratio
--------------------------
Reading each point from its own background-subtracted frame gives data centred
on **zero**. A "ratio to control" then divides by noise about zero and is
meaningless (in the source analysis it returned −1550×). :func:`rod_significance`
reports ``(rod − median(control)) / sigma(control)`` with a robust sigma
(1.4826 × MAD), which is the correct scale for zero-centred data.

The control has to be able to fail
----------------------------------
The matched control is the identical walk at ``(h + ½, k + ½, L)``: not a
lattice rod, but the same \\|q\\| range, the same curvature, the same ω range and
the same detector regions. Anything the geometry does to the rod it does to the
control.

An earlier control — the azimuthal median at the same radius — returned exactly
zero for every row, because a polar-median-subtracted stack *is* an azimuthal
median per (2θ, sector), so its azimuthal median is zero by construction. **A
control that cannot fail is not a control.** :func:`rod_significance` refuses a
control whose scatter is identically zero for this reason.

Not-observable is not not-there
-------------------------------
Three distinct things make a point drop out, none of which is "no intensity":
no ω solution inside the delivered rocking range; the projection lands off the
detector; the transverse slice is mostly masked. All three are recorded
separately and returned as NaN, never as zero. A silent gap looks exactly like
a real minimum in the rod, which is the quantity being measured.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, TYPE_CHECKING, Tuple

import math
import numpy as np

if TYPE_CHECKING:
    from .geometry import Geometry

__all__ = [
    "RodPath", "LauePath", "RodProfile", "WidthResult",
    "rod_path", "matched_control_path", "profile_along",
    "rod_path_geometry", "matched_control_path_geometry",
    "rod_path_laue", "matched_control_path_laue", "profile_along_laue",
    "laue_predict_hkl", "rod_collisions",
    "rod_significance", "ring_L_marks", "transverse_width",
    "centred_L_nodes", "diffuse_to_bragg",
]

#: Reasons a point on the rod is not observable. Never conflate with zero.
DROP_NO_OMEGA = "no_omega_solution"
DROP_OFF_DETECTOR = "off_detector"
DROP_MASKED = "masked"
DROP_NO_ENERGY = "no_energy_solution"

#: hc in keV*Angstrom (E = hc / lambda). Local to this module: the rest of
#: midas_defect.geometry is wavelength-in, never energy-out, because every
#: existing consumer is monochromatic.
_HC_KEV_A = 12.39842


@dataclass
class RodPath:
    """The detector image of a reciprocal-space rod, point by point."""
    L: np.ndarray                    # continuous L actually realised
    row: np.ndarray
    col: np.ndarray
    omega_deg: np.ndarray
    q_mag: np.ndarray
    hk: Tuple[float, float]
    dropped: Dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.L)

    def __str__(self) -> str:
        d = ", ".join(f"{k} {v}" for k, v in sorted(self.dropped.items())) or "none"
        return (f"rod (h,k) = {self.hk}: {len(self)} points, "
                f"L {self.L.min():+.2f}..{self.L.max():+.2f}; dropped: {d}")


def rod_path(U: np.ndarray, B: np.ndarray, h: float, k: float,
             L_values: Sequence[float], *,
             wavelength_A: float, lsd_um: float, pixel_um: float,
             bc_row: float, bc_col: float, n_rows: int, n_cols: int,
             omega_lo_deg: float, omega_hi_deg: float,
             omega_sign: int = 1,
             q_convention: str = "1/d") -> RodPath:
    """Walk ``(h, k, L)`` through q-space, solving ω, and project to pixels.

    ``B`` must be in the same convention as ``q_convention``: ``"1/d"`` for a
    plain ``diag(1/a, 1/b, 1/c)``, or ``"2pi/d"`` for one scaled by 2π (the
    convention the rest of :mod:`midas_defect` uses). The Bragg condition is
    written for whichever is supplied. **The default is ``"1/d"`` although the rest
    of :mod:`midas_defect` is 2π/d.** A 2π-scaled ``B`` left on the default does not
    raise: the Bragg condition then fails for most ``L`` and the points land in
    ``RodPath.dropped`` instead. An unexpectedly short path is the symptom.

    **The detector is FLAT.** Points are projected onto a plane normal to the beam
    at ``lsd_um``, with ``Y = -(col - bc_col)·px`` and ``Z = +(row - bc_row)·px``: no
    tilt, no ``tx``, no radial distortion. On a tilted detector a predicted point is
    off by up to about ``lsd_um·tan(tilt)/pixel_um`` px -- ~14 px at 0.4° and
    Lsd ≈ 350 mm -- so every rod walked this way carries that error.

    **``omega_sign`` changes only the REPORTED ω.** The rotation that places a point
    on the detector is always the physical angle ``w`` solving the Bragg condition;
    ``omega_sign`` enters only ``omega = omega_sign·w``, which is both the value
    returned and the value tested against ``[omega_lo_deg, omega_hi_deg]``. It turns
    a physical rotation into a motor reading. It does **not** mirror reciprocal space:
    if the sample-frame vectors behind ``U`` were built with the opposite rotation
    sense, fix that where they were built. With a window symmetric about zero,
    ``+1`` and ``-1`` give identical pixels and negated ω (pinned by a test).

    Points that cannot be observed are **counted, not fabricated** — see
    ``RodPath.dropped``.
    """
    if q_convention not in ("1/d", "2pi/d"):
        raise ValueError("q_convention must be '1/d' or '2pi/d'")
    # Bragg: G_lab_x = -lambda|G|^2/2 in 1/d units; with q = 2pi/d the same
    # condition is q_lab_x = -|q|^2 lambda / (4 pi).
    bragg_scale = (wavelength_A / 2.0 if q_convention == "1/d"
                   else wavelength_A / (4.0 * math.pi))
    U = np.asarray(U, float)
    B = np.asarray(B, float)
    out_L, out_r, out_c, out_w, out_q = [], [], [], [], []
    dropped = {DROP_NO_OMEGA: 0, DROP_OFF_DETECTOR: 0}

    for Lc in np.asarray(L_values, float):
        g_s = U @ (B @ np.array([h, k, Lc], float))
        gn = float(np.linalg.norm(g_s))
        if gn < 1e-12:
            dropped[DROP_NO_OMEGA] += 1
            continue
        D = -bragg_scale * gn * gn
        A_, Bc = g_s[0], -g_s[1]
        R = math.hypot(A_, Bc)
        if R < 1e-12 or abs(D) > R:
            dropped[DROP_NO_OMEGA] += 1
            continue
        base = math.atan2(Bc, A_)
        dw = math.acos(max(-1.0, min(1.0, D / R)))
        chosen = None
        for w in (base + dw, base - dw):
            ome = (omega_sign * math.degrees(w) + 180.0) % 360.0 - 180.0
            if omega_lo_deg <= ome <= omega_hi_deg:
                chosen = (w, ome)
                break
        if chosen is None:
            dropped[DROP_NO_OMEGA] += 1
            continue
        w, ome = chosen
        cw, sw = math.cos(w), math.sin(w)
        g_lab = np.array([cw * g_s[0] - sw * g_s[1],
                          sw * g_s[0] + cw * g_s[1], g_s[2]])
        k_i = (1.0 / wavelength_A if q_convention == "1/d"
               else 2.0 * math.pi / wavelength_A)
        kf = np.array([k_i, 0.0, 0.0]) + g_lab
        if kf[0] <= 0:
            dropped[DROP_OFF_DETECTOR] += 1
            continue
        t = lsd_um / kf[0]
        # signs match midas_defect.geometry.pixel_to_qlab: Y = -(col-BCy)*px,
        # Z = +(row-BCz)*px
        r_p = bc_row + kf[2] * t / pixel_um
        c_p = bc_col - kf[1] * t / pixel_um
        if not (0 <= r_p < n_rows and 0 <= c_p < n_cols):
            dropped[DROP_OFF_DETECTOR] += 1
            continue
        out_L.append(Lc); out_r.append(r_p); out_c.append(c_p)
        out_w.append(ome); out_q.append(gn)

    return RodPath(L=np.asarray(out_L), row=np.asarray(out_r),
                   col=np.asarray(out_c), omega_deg=np.asarray(out_w),
                   q_mag=np.asarray(out_q), hk=(h, k), dropped=dropped)


def matched_control_path(U, B, h: float, k: float, L_values, **kw) -> RodPath:
    """The identical walk at ``(h + ½, k + ½, L)`` — the matched control.

    Not a reciprocal-lattice rod, but the same |q| range, curvature, ω range and
    detector regions. If intensity is elevated only at integer (h, k), the rod
    is real.
    """
    return rod_path(U, B, h + 0.5, k + 0.5, L_values, **kw)


@dataclass
class LauePath:
    """The detector image of a reciprocal-space rod, walked by ENERGY not ω.

    :func:`rod_path` walks a rod at fixed wavelength, solving for the ω that
    puts each L on the Ewald sphere. A stationary (Laue) crystal has no ω to
    solve for -- instead each L has, at most, ONE wavelength that satisfies
    Bragg for that fixed orientation, from the same equation with the roles
    reversed. ``LauePath`` is the energy-sweep analogue of :class:`RodPath`:
    same idea (walk the rod where it is straight, in (h, k, L), and project
    each point through its OWN diffraction condition), different free
    variable, so it is a new dataclass rather than a strained reuse of
    ``RodPath.omega_deg``.
    """
    L: np.ndarray
    row: np.ndarray
    col: np.ndarray
    energy_keV: np.ndarray
    q_mag: np.ndarray
    hk: Tuple[float, float]
    dropped: Dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.L)

    def __str__(self) -> str:
        d = ", ".join(f"{k} {v}" for k, v in sorted(self.dropped.items())) or "none"
        return (f"Laue rod (h,k) = {self.hk}: {len(self)} points, "
                f"L {self.L.min():+.2f}..{self.L.max():+.2f}; dropped: {d}")


def _laue_solve_point(g_s: np.ndarray, *,
                      bragg_scale: float, lsd_um: float, pixel_um: float,
                      bc_row: float, bc_col: float, n_rows: int, n_cols: int,
                      E_lo_keV: float, E_hi_keV: float,
                      q_convention: str) -> Tuple[float, float, float, float, Optional[str]]:
    """One stationary-crystal Bragg solve: sample-frame vector -> (row, col, energy, |g|, reason).

    ``reason`` is ``None`` on success, else :data:`DROP_NO_ENERGY` or
    :data:`DROP_OFF_DETECTOR` -- the geometric-infeasibility vs
    detector-footprint distinction :func:`rod_path_laue` already tracks (see
    its docstring for why a backscattering ``kf_x <= 0`` counts as the
    former, not the latter). On failure the numeric fields are NaN, never a
    fabricated value.

    Shared by :func:`rod_path_laue` (sweeps one (h, k) row over continuous L)
    and :func:`laue_predict_hkl` (evaluates an arbitrary list of hkl) so the
    energy-solve algebra exists in exactly one place.
    """
    nan4 = (float("nan"),) * 4
    gn = float(np.linalg.norm(g_s))
    if gn < 1e-12 or g_s[0] >= 0:
        return (*nan4, DROP_NO_ENERGY)
    wavelength_A = -g_s[0] / (bragg_scale * gn * gn)
    energy = _HC_KEV_A / wavelength_A
    if not (E_lo_keV <= energy <= E_hi_keV):
        return (*nan4, DROP_NO_ENERGY)
    k_i = (1.0 / wavelength_A if q_convention == "1/d"
           else 2.0 * math.pi / wavelength_A)
    kf = np.array([k_i, 0.0, 0.0]) + g_s  # omega = 0: g_lab == g_s
    if kf[0] <= 0:
        return (*nan4, DROP_NO_ENERGY)
    t = lsd_um / kf[0]
    r_p = bc_row + kf[2] * t / pixel_um
    c_p = bc_col - kf[1] * t / pixel_um
    if not (0 <= r_p < n_rows and 0 <= c_p < n_cols):
        return (r_p, c_p, energy, gn, DROP_OFF_DETECTOR)
    return (r_p, c_p, energy, gn, None)


def laue_predict_hkl(U: np.ndarray, B: np.ndarray, hkl: np.ndarray, *,
                     lsd_um: float, pixel_um: float,
                     bc_row: float, bc_col: float, n_rows: int, n_cols: int,
                     E_lo_keV: float, E_hi_keV: float,
                     q_convention: str = "1/d") -> dict:
    """Stationary-crystal Laue prediction for an arbitrary list of hkl.

    The non-swept counterpart of :func:`rod_path_laue`: instead of one fixed
    (h, k) row over continuous L, this takes an ``(N, 3)`` array of discrete
    reflections (e.g. every F-centred hkl in a resolution box) and predicts
    which ones are observable. Built for :func:`rod_collisions` -- checking
    whether some OTHER reflection's predicted spot lands on top of a rod's --
    but is a standalone utility (a one-shot Laue pattern predictor) in its
    own right.

    Returns a dict of equal-length arrays for the accessible subset only:
    ``hkl`` (N, 3), ``row``, ``col``, ``energy_keV``, ``q_mag``. Uses the
    SAME per-point solve as :func:`rod_path_laue` (see :func:`_laue_solve_point`),
    not a separate re-derivation.
    """
    if q_convention not in ("1/d", "2pi/d"):
        raise ValueError("q_convention must be '1/d' or '2pi/d'")
    bragg_scale = (0.5 if q_convention == "1/d" else 1.0 / (4.0 * math.pi))
    U = np.asarray(U, float)
    B = np.asarray(B, float)
    hkl = np.asarray(hkl, float)
    if hkl.ndim != 2 or hkl.shape[1] != 3:
        raise ValueError(f"hkl must be (N, 3), got {hkl.shape}")

    kw = dict(bragg_scale=bragg_scale, lsd_um=lsd_um, pixel_um=pixel_um,
              bc_row=bc_row, bc_col=bc_col, n_rows=n_rows, n_cols=n_cols,
              E_lo_keV=E_lo_keV, E_hi_keV=E_hi_keV, q_convention=q_convention)
    out_hkl, out_r, out_c, out_e, out_q = [], [], [], [], []
    for row in hkl:
        g_s = U @ (B @ row)
        r_p, c_p, energy, gn, reason = _laue_solve_point(g_s, **kw)
        if reason is not None:
            continue
        out_hkl.append(row); out_r.append(r_p); out_c.append(c_p)
        out_e.append(energy); out_q.append(gn)

    return {
        "hkl": np.asarray(out_hkl).reshape(-1, 3),
        "row": np.asarray(out_r), "col": np.asarray(out_c),
        "energy_keV": np.asarray(out_e), "q_mag": np.asarray(out_q),
    }


def rod_collisions(path: LauePath, U: np.ndarray, B: np.ndarray,
                   hkl_candidates: np.ndarray, *,
                   tol_px: float = 5.0,
                   exclude_own_row: bool = True,
                   **laue_predict_kw) -> dict:
    """Which points on a Laue rod share a pixel with a DIFFERENT reflection.

    A Pilatus (or any energy-integrating detector) cannot separate two
    reflections that diffract at different energies but land on the same
    pixel -- the harmonic-collision problem this whole module chain exists to
    answer. For every point on ``path``, this checks ``hkl_candidates``
    (typically every F-centred hkl in a resolution box, from the SAME
    orientation and cell) for the nearest OTHER accessible reflection, and
    reports a collision if one lands within ``tol_px``.

    Candidates whose (h, k) matches ``path.hk`` exactly are the rod's own
    integer members, not contamination -- excluded whenever
    ``exclude_own_row`` (default) regardless of which L, otherwise a
    candidate box that happens to include the rod's own reflections would
    have them flag each other (two genuine points on the SAME rod are not a
    collision between DIFFERENT reflections, even when their pixels happen
    to be close).

    Returns a dict with per-path-point arrays ``collided`` (bool),
    ``nearest_hkl`` (N, 3; NaN row where no candidate was ever evaluated),
    and ``nearest_dist_px`` (inf where none). Does not know about any OTHER
    phase (e.g. diamond anvils) -- pass their own hkl list (predicted with
    THEIR OWN U, B) through the same candidates array to include them.
    """
    if len(path) == 0:
        return {"collided": np.zeros(0, bool),
                "nearest_hkl": np.zeros((0, 3)),
                "nearest_dist_px": np.zeros(0)}

    pred = laue_predict_hkl(U, B, hkl_candidates, **laue_predict_kw)
    if len(pred["row"]) == 0:
        return {"collided": np.zeros(len(path), bool),
                "nearest_hkl": np.full((len(path), 3), np.nan),
                "nearest_dist_px": np.full(len(path), np.inf)}

    own_h, own_k = path.hk
    if exclude_own_row:
        keep_mask = ~(np.isclose(pred["hkl"][:, 0], own_h)
                      & np.isclose(pred["hkl"][:, 1], own_k))
    else:
        keep_mask = np.ones(len(pred["row"]), bool)

    collided = np.zeros(len(path), bool)
    nearest_hkl = np.full((len(path), 3), np.nan)
    nearest_dist = np.full(len(path), np.inf)

    for i in range(len(path)):
        keep = keep_mask
        if not keep.any():
            continue
        d = np.hypot(pred["row"][keep] - path.row[i], pred["col"][keep] - path.col[i])
        j = int(np.argmin(d))
        nearest_dist[i] = d[j]
        nearest_hkl[i] = pred["hkl"][keep][j]
        collided[i] = d[j] <= tol_px

    return {"collided": collided, "nearest_hkl": nearest_hkl, "nearest_dist_px": nearest_dist}


def rod_path_laue(U: np.ndarray, B: np.ndarray, h: float, k: float,
                  L_values: Sequence[float], *,
                  lsd_um: float, pixel_um: float,
                  bc_row: float, bc_col: float, n_rows: int, n_cols: int,
                  E_lo_keV: float, E_hi_keV: float,
                  q_convention: str = "1/d") -> LauePath:
    """Walk ``(h, k, L)`` through q-space, solving ENERGY, and project to pixels.

    The stationary-crystal (Laue) analogue of :func:`rod_path`: ω is fixed at
    0 (the crystal does not rotate), and the free variable per point is the
    diffracting energy. For a fixed orientation, a reciprocal-lattice
    direction has at most one energy that satisfies Bragg -- unlike
    :func:`rod_path`'s two ω roots -- because there is no second sense of
    rotation to try.

    Same conventions and same caveats as :func:`rod_path`, which this
    duplicates on purpose rather than sharing a code path with (the omega
    solve and the energy solve are different enough algebraically that
    forcing one implementation to cover both would obscure both): ``B`` in
    ``q_convention`` ("1/d" default, or "2pi/d"); **the detector is FLAT**
    (no tilt, no distortion -- see :func:`rod_path`'s docstring for the
    ``~lsd*tan(tilt)/pixel`` error this costs); row/col signs match
    :func:`midas_defect.geometry.pixel_to_qlab`.

    ``E_lo_keV``/``E_hi_keV`` is the usable energy band (source spectrum and/or
    detector response), the direct analogue of ``rod_path``'s
    ``[omega_lo_deg, omega_hi_deg]`` window.

    Points that cannot be observed are counted under
    :data:`DROP_NO_ENERGY` (no forward-scattering solution, OR a solution
    outside the energy band -- ``rod_path`` does not separate its two
    equivalent failure modes either) and :data:`DROP_OFF_DETECTOR`, never
    fabricated.
    """
    if q_convention not in ("1/d", "2pi/d"):
        raise ValueError("q_convention must be '1/d' or '2pi/d'")
    bragg_scale = (0.5 if q_convention == "1/d" else 1.0 / (4.0 * math.pi))
    U = np.asarray(U, float)
    B = np.asarray(B, float)
    out_L, out_r, out_c, out_e, out_q = [], [], [], [], []
    dropped = {DROP_NO_ENERGY: 0, DROP_OFF_DETECTOR: 0}
    kw = dict(bragg_scale=bragg_scale, lsd_um=lsd_um, pixel_um=pixel_um,
              bc_row=bc_row, bc_col=bc_col, n_rows=n_rows, n_cols=n_cols,
              E_lo_keV=E_lo_keV, E_hi_keV=E_hi_keV, q_convention=q_convention)

    for Lc in np.asarray(L_values, float):
        g_s = U @ (B @ np.array([h, k, Lc], float))
        r_p, c_p, energy, gn, reason = _laue_solve_point(g_s, **kw)
        if reason is not None:
            dropped[reason] += 1
            continue
        out_L.append(Lc); out_r.append(r_p); out_c.append(c_p)
        out_e.append(energy); out_q.append(gn)

    return LauePath(L=np.asarray(out_L), row=np.asarray(out_r),
                    col=np.asarray(out_c), energy_keV=np.asarray(out_e),
                    q_mag=np.asarray(out_q), hk=(h, k), dropped=dropped)


def matched_control_path_laue(U, B, h: float, k: float, L_values, **kw) -> LauePath:
    """:func:`matched_control_path`'s energy-sweep analogue -- see its docstring
    for what "matched" means and why the control must be able to fail."""
    return rod_path_laue(U, B, h + 0.5, k + 0.5, L_values, **kw)


def profile_along_laue(image: np.ndarray, mask: np.ndarray, path: LauePath, *,
                       half_width_px: float = 6.0,
                       min_valid_fraction: float = 0.5,
                       reducer: str = "mean") -> RodProfile:
    """Transverse-slice intensity along ``path``, from ONE Laue image.

    The single-frame analogue of :func:`profile_along`: a Laue exposure is
    stationary, so there is no per-point frame to select (every point on the
    rod is read from the same image) -- the only difference from
    ``profile_along`` beyond that. Returns the SAME :class:`RodProfile` used
    by the rotation-series path, so :func:`rod_significance` and
    :func:`transverse_width` are unmodified and reused, not duplicated.

    ``RodProfile.omega_deg`` is filled with NaN: a Laue profile has no omega
    per point, and nothing downstream (:func:`rod_significance`,
    :func:`transverse_width`) reads that field. The path's own
    ``energy_keV`` is not carried through -- keep ``path`` if you need it.
    """
    image = np.asarray(image)
    mask = np.asarray(mask, bool)
    if image.ndim != 2:
        raise ValueError(f"image must be (rows, cols), got {image.shape}")
    if reducer not in ("mean", "max"):
        raise ValueError("reducer must be 'mean' or 'max'")
    n_r, n_c = image.shape

    if len(path) < 3:
        raise ValueError("path too short to define a tangent")
    dr = np.gradient(path.row)
    dc = np.gradient(path.col)
    norm = np.hypot(dr, dc)
    norm[norm == 0] = 1.0
    pr, pc = -dc / norm, dr / norm
    offs = np.arange(-half_width_px, half_width_px + 1e-9, 1.0)

    inten = np.full(len(path), np.nan)
    nvalid = np.zeros(len(path), int)
    dropped = dict(path.dropped)
    dropped.setdefault(DROP_MASKED, 0)

    for i in range(len(path)):
        rr = np.round(path.row[i] + offs * pr[i]).astype(int)
        cc = np.round(path.col[i] + offs * pc[i]).astype(int)
        inside = (rr >= 0) & (rr < n_r) & (cc >= 0) & (cc < n_c)
        if not inside.any():
            dropped[DROP_OFF_DETECTOR] = dropped.get(DROP_OFF_DETECTOR, 0) + 1
            continue
        rr, cc = rr[inside], cc[inside]
        good = ~mask[rr, cc]
        nvalid[i] = int(good.sum())
        if good.sum() < min_valid_fraction * len(offs):
            dropped[DROP_MASKED] += 1
            continue
        vals = image[rr[good], cc[good]]
        inten[i] = float(vals.mean() if reducer == "mean" else vals.max())

    return RodProfile(L=path.L, intensity=inten, n_valid_px=nvalid,
                      omega_deg=np.full(len(path), np.nan), dropped=dropped)


def rod_path_geometry(U: np.ndarray, B: np.ndarray, h: float, k: float,
                      L_values: Sequence[float], geom: "Geometry", *,
                      omega_sign: int = 1,
                      omega_lo_deg: Optional[float] = None,
                      omega_hi_deg: Optional[float] = None) -> RodPath:
    """Tilt- and distortion-aware analogue of :func:`rod_path`.

    :func:`rod_path` projects onto a flat, untilted detector plane -- a real,
    documented limitation (see its own docstring: off by ``~lsd*tan(tilt)/px``,
    about 14 px at a 0.4° tilt and Lsd ≈ 350 mm). No detector is ever actually
    untilted or undistorted, so this walks the identical ``(h, k, L)`` in
    q-space and solves the identical Bragg condition, but projects through
    :func:`midas_defect.geometry.qlab_to_pixel` -- the iterative inverse of
    :func:`midas_defect.geometry.pixel_to_qlab`, which honours
    ``geom.tx_deg/ty_deg/tz_deg`` and the 15-coefficient radial distortion
    (``geom.p_coeffs``). These are the same primitives (plus
    :func:`midas_defect.geometry.ewald_crossing_omegas` and
    :func:`midas_defect.geometry.qsample_to_qlab`) that
    :func:`midas_defect.raster.predict_reflections` already uses for tilted,
    distorted detector prediction -- not a new geometric model, a second
    consumer of the validated one.

    ``B`` must be in the ``"2pi/d"`` convention -- the only one
    :mod:`midas_defect.geometry`'s primitives use (see :func:`rod_path`'s own
    convention note for what that means). There is no ``q_convention``
    parameter here.

    **Reduces exactly to** :func:`rod_path` **at zero tilt and zero
    distortion.** Both solve the identical Bragg condition (same ``A, B, C, R,
    phi`` algebra); at ``tx=ty=tz=0`` and ``p_coeffs`` all zero,
    ``qlab_to_pixel``'s seed -- the exact flat-detector inverse -- already
    satisfies its own convergence check in zero iterations, and is algebraically
    the same projection :func:`rod_path` computes by hand. Pinned by
    ``test_rod_path_geometry_matches_flat_rod_path_at_zero_tilt`` in
    ``tests/test_rod_profile.py``: do not treat that test as redundant with
    :func:`rod_path`'s own tests -- it is what makes this function a safe
    addition rather than a second, silently-diverging implementation.

    ``omega_lo_deg``/``omega_hi_deg`` default to
    ``geom.omega_first_deg - 0.5*geom.omega_step_deg`` .. ``geom.omega_first_deg
    + (geom.n_frames - 0.5)*geom.omega_step_deg`` -- half a frame beyond the
    first/last nominal ω, the same convention :func:`midas_defect.raster.
    predict_reflections` already uses to turn a ``Geometry`` into an ω window.
    Unlike :func:`rod_path`, the search is not restricted to a single 360° turn
    (real for a sweep spanning more than one revolution); pass explicit bounds
    to restrict it.

    One ``qlab_to_pixel`` call, batched over every kept ``L`` -- not one call
    per point. Measured (``dev/bench_rod_path_geometry.py``, real 2604
    geometry, 1601-point L grid): the very first ``qlab_to_pixel`` call in
    a process pays a one-time ~0.5-0.6 s import/backend-init cost (lazy import
    of ``midas_transforms.fit_setup.transform``); every call after that is
    ~1-2 ms **independent of how many points are in the batch** (26 points and
    1040 points both land near 1 ms warm), so this only costs roughly 2-3x
    :func:`rod_path`'s own per-rod time once warmed up -- negligible next to
    the per-``L`` Python Bragg-condition loop both functions share. Pinned
    qualitatively (not by absolute wall-clock, which would be flaky across
    machines) by
    ``test_rod_path_geometry_cost_does_not_scale_with_point_count`` in
    ``tests/test_rod_profile.py``. If a point in the batch fails to converge
    (``RuntimeError`` from ``qlab_to_pixel``), that one batch falls back to a
    per-point solve so one marginal point cannot drop the whole rod.
    """
    import torch
    from .geometry import ewald_crossing_omegas, qlab_to_pixel, qsample_to_qlab

    U = np.asarray(U, float)
    B = np.asarray(B, float)
    if omega_lo_deg is None:
        omega_lo_deg = geom.omega_first_deg - 0.5 * geom.omega_step_deg
    if omega_hi_deg is None:
        omega_hi_deg = geom.omega_first_deg + (geom.n_frames - 0.5) * geom.omega_step_deg
    if omega_lo_deg > omega_hi_deg:
        omega_lo_deg, omega_hi_deg = omega_hi_deg, omega_lo_deg
    omega_lo, omega_hi = math.radians(omega_lo_deg), math.radians(omega_hi_deg)

    kept_L, kept_w, kept_qmag, qlab_batch = [], [], [], []
    dropped = {DROP_NO_OMEGA: 0, DROP_OFF_DETECTOR: 0}
    wedge_rad = math.radians(geom.wedge_deg)

    for Lc in np.asarray(L_values, float):
        g_s = U @ (B @ np.array([h, k, Lc], float))
        qmag = float(np.linalg.norm(g_s))
        if qmag < 1e-12:
            dropped[DROP_NO_OMEGA] += 1
            continue
        chosen = None
        for w in ewald_crossing_omegas(g_s, geom.wavelength_A, wedge_rad):
            for n in (-1, 0, 1):
                ww = w + 2.0 * math.pi * n
                w_reported = omega_sign * ww
                if omega_lo <= w_reported <= omega_hi:
                    chosen = ww
                    break
            if chosen is not None:
                break
        if chosen is None:
            dropped[DROP_NO_OMEGA] += 1
            continue
        q_lab = qsample_to_qlab(torch.as_tensor(g_s, dtype=torch.float64), chosen, wedge_rad)
        kept_L.append(Lc); kept_w.append(omega_sign * chosen)
        kept_qmag.append(qmag); qlab_batch.append(q_lab)

    if not qlab_batch:
        return RodPath(L=np.empty(0), row=np.empty(0), col=np.empty(0),
                       omega_deg=np.empty(0), q_mag=np.empty(0), hk=(h, k),
                       dropped=dropped)

    qlab_t = torch.stack(qlab_batch)
    try:
        rows_t, cols_t = qlab_to_pixel(qlab_t, geom, device="cpu")
    except RuntimeError:
        rows_t = torch.full((len(qlab_batch),), float("nan"), dtype=torch.float64)
        cols_t = torch.full((len(qlab_batch),), float("nan"), dtype=torch.float64)
        for i, qb in enumerate(qlab_batch):
            try:
                r, c = qlab_to_pixel(qb.reshape(1, 3), geom, device="cpu")
                rows_t[i], cols_t[i] = r[0], c[0]
            except RuntimeError:
                pass
    rows, cols = rows_t.numpy(), cols_t.numpy()

    out_L, out_r, out_c, out_w, out_q = [], [], [], [], []
    for Lc, w, qm, r, c in zip(kept_L, kept_w, kept_qmag, rows, cols):
        if not (np.isfinite(r) and np.isfinite(c)
                and 0 <= r < geom.n_pix_z and 0 <= c < geom.n_pix_y):
            dropped[DROP_OFF_DETECTOR] += 1
            continue
        out_L.append(Lc); out_r.append(r); out_c.append(c)
        out_w.append(math.degrees(w)); out_q.append(qm)

    return RodPath(L=np.array(out_L), row=np.array(out_r), col=np.array(out_c),
                   omega_deg=np.array(out_w), q_mag=np.array(out_q),
                   hk=(h, k), dropped=dropped)


def matched_control_path_geometry(U: np.ndarray, B: np.ndarray, h: float, k: float,
                                  L_values: Sequence[float], geom: "Geometry",
                                  **kw) -> RodPath:
    """:func:`matched_control_path`'s tilt/distortion-aware analogue.

    The identical walk at ``(h + ½, k + ½, L)`` through the full ``geom``. See
    :func:`matched_control_path` for what "matched" means and why the control
    must be able to fail.
    """
    return rod_path_geometry(U, B, h + 0.5, k + 0.5, L_values, geom, **kw)


@dataclass
class RodProfile:
    """Intensity along a rod, with non-observable points kept as NaN."""
    L: np.ndarray
    intensity: np.ndarray            # NaN where not observable
    n_valid_px: np.ndarray
    omega_deg: np.ndarray
    dropped: Dict[str, int]

    @property
    def observed(self) -> np.ndarray:
        return np.isfinite(self.intensity)

    def __str__(self) -> str:
        n = int(self.observed.sum())
        return (f"rod profile: {n} of {len(self.L)} points observable "
                f"({100*n/max(len(self.L),1):.0f} %); dropped "
                + ", ".join(f"{k} {v}" for k, v in sorted(self.dropped.items())))


def profile_along(stack: np.ndarray, mask: np.ndarray, path: RodPath, *,
                  omega_first_deg: float, omega_step_deg: float,
                  half_width_px: float = 6.0,
                  min_valid_fraction: float = 0.5,
                  reducer: str = "mean") -> RodProfile:
    """Transverse-slice intensity along ``path``, each point from its own frame.

    ``reducer`` is ``"mean"`` or ``"max"``. Prefer the mean: a max over a slice
    is positively biased, and a max over frames is worse — the maximum of many
    noisy samples sits well above the median of one.

    A slice more than ``1 - min_valid_fraction`` masked yields **NaN**, and is
    counted under ``masked``. It is never returned as zero.
    """
    stack = np.asarray(stack)
    mask = np.asarray(mask, bool)
    if stack.ndim != 3:
        raise ValueError(f"stack must be (n_frames, rows, cols), got {stack.shape}")
    if reducer not in ("mean", "max"):
        raise ValueError("reducer must be 'mean' or 'max'")
    n_f, n_r, n_c = stack.shape

    # local perpendicular from the path's own tangent
    if len(path) < 3:
        raise ValueError("path too short to define a tangent")
    dr = np.gradient(path.row)
    dc = np.gradient(path.col)
    norm = np.hypot(dr, dc)
    norm[norm == 0] = 1.0
    pr, pc = -dc / norm, dr / norm
    offs = np.arange(-half_width_px, half_width_px + 1e-9, 1.0)

    inten = np.full(len(path), np.nan)
    nvalid = np.zeros(len(path), int)
    dropped = dict(path.dropped)
    dropped.setdefault(DROP_MASKED, 0)

    for i in range(len(path)):
        kf = int(round((path.omega_deg[i] - omega_first_deg) / omega_step_deg))
        if not (0 <= kf < n_f):
            dropped[DROP_NO_OMEGA] = dropped.get(DROP_NO_OMEGA, 0) + 1
            continue
        rr = np.round(path.row[i] + offs * pr[i]).astype(int)
        cc = np.round(path.col[i] + offs * pc[i]).astype(int)
        inside = (rr >= 0) & (rr < n_r) & (cc >= 0) & (cc < n_c)
        if not inside.any():
            dropped[DROP_OFF_DETECTOR] = dropped.get(DROP_OFF_DETECTOR, 0) + 1
            continue
        rr, cc = rr[inside], cc[inside]
        good = ~mask[rr, cc]
        nvalid[i] = int(good.sum())
        if good.sum() < min_valid_fraction * len(offs):
            dropped[DROP_MASKED] += 1
            continue
        vals = stack[kf, rr[good], cc[good]]
        inten[i] = float(vals.mean() if reducer == "mean" else vals.max())

    return RodProfile(L=path.L, intensity=inten, n_valid_px=nvalid,
                      omega_deg=path.omega_deg, dropped=dropped)


def rod_significance(rod: RodProfile, control: RodProfile) -> dict:
    """Rod intensity in SIGMA above a matched control. Never a ratio.

    Refuses a control with zero scatter — that is the signature of a control
    that cannot fail (e.g. taking the azimuthal median of data that has already
    had its azimuthal median subtracted).
    """
    r = rod.intensity[np.isfinite(rod.intensity)]
    c = control.intensity[np.isfinite(control.intensity)]
    if r.size == 0 or c.size == 0:
        raise ValueError("rod or control has no observable points")
    med = float(np.median(c))
    sigma = float(1.4826 * np.median(np.abs(c - med)))
    if sigma <= 0:
        raise ValueError(
            "the control has zero scatter, so it cannot fail. This is what "
            "happens when the control is the azimuthal median of data whose "
            "azimuthal median has already been subtracted. Use a control that "
            "samples the same geometry but not the lattice — "
            "matched_control_path().")
    return {"median_sigma": float((np.median(r) - med) / sigma),
            "mean_sigma": float((r.mean() - med) / sigma),
            "control_median": med, "control_sigma": sigma,
            "n_rod": int(r.size), "n_control": int(c.size)}


def ring_L_marks(path: RodPath, ring_radii_px: Sequence[float], *,
                 bc_row: float, bc_col: float,
                 tolerance_px: float = 3.0) -> np.ndarray:
    """L values at which the rod crosses a powder ring — mark, do not delete.

    A ring puts a bump at one |q|, and therefore at one L, imitating exactly the
    modulation that would encode fault statistics. Get ``ring_radii_px`` from
    the RAW data (``midas_defect.ingest.detect_powder_rings``), not from a
    background-subtracted stack where the rings have already been removed.
    """
    if len(path) == 0 or len(ring_radii_px) == 0:
        return np.empty(0)
    rad = np.hypot(path.row - bc_row, path.col - bc_col)
    hit = np.zeros(len(path), bool)
    for r0 in ring_radii_px:
        hit |= np.abs(rad - r0) <= tolerance_px
    return path.L[hit]


@dataclass
class WidthResult:
    """Transverse rod width against the Bragg width on the SAME rod."""
    diffuse_fwhm_inv_A: float
    bragg_fwhm_inv_A: float
    excess_fwhm_inv_A: Optional[float]      # None when resolution-limited
    coherence_length_A: Optional[float]     # None when resolution-limited
    lower_bound_A: Optional[float]          # set when resolution-limited
    resolution_limited: bool

    def __str__(self) -> str:
        if self.resolution_limited:
            return (f"RESOLUTION-LIMITED: diffuse FWHM "
                    f"{self.diffuse_fwhm_inv_A:.5f} <= Bragg "
                    f"{self.bragg_fwhm_inv_A:.5f} 1/A. Lateral coherence "
                    f"> {self.lower_bound_A:.0f} A (a LOWER BOUND, not a value)")
        return (f"diffuse {self.diffuse_fwhm_inv_A:.5f} vs Bragg "
                f"{self.bragg_fwhm_inv_A:.5f} 1/A -> excess "
                f"{self.excess_fwhm_inv_A:.5f} -> lateral coherence "
                f"~{self.coherence_length_A:.0f} A")


def transverse_width(diffuse_fwhm_inv_A: float, bragg_fwhm_inv_A: float
                     ) -> WidthResult:
    """Deconvolve the instrument from a rod's transverse width.

    The Bragg peaks **on the same rod** carry exactly the resolution function —
    same beam, optics, mosaic and detector PSF — so::

        excess² = width(diffuse)² − width(Bragg)²

    If the two are equal the rod is **resolution-limited** and the honest output
    is a lower bound on the lateral coherence length, not a number. That is a
    real possible result, not a failure, and this function returns it as one
    rather than reporting a spuriously large length.

    With |G| = 1/d in 1/Å, a lateral domain of size D gives FWHM ~ 1/D, so
    D ~ 1/ΔG. No Scherrer constant is applied — the number is an
    order-of-magnitude coherence length.
    """
    if diffuse_fwhm_inv_A <= 0 or bragg_fwhm_inv_A <= 0:
        raise ValueError("widths must be positive")
    if diffuse_fwhm_inv_A <= bragg_fwhm_inv_A:
        return WidthResult(diffuse_fwhm_inv_A, bragg_fwhm_inv_A, None, None,
                           lower_bound_A=1.0 / bragg_fwhm_inv_A,
                           resolution_limited=True)
    excess = math.sqrt(diffuse_fwhm_inv_A ** 2 - bragg_fwhm_inv_A ** 2)
    return WidthResult(diffuse_fwhm_inv_A, bragg_fwhm_inv_A, excess,
                       1.0 / excess, None, False)


# ── The (h,k) dependence of a rod: what encodes an in-plane fault vector ────
#
# Ported 2026-09-10 from the La3Ni2O7 project's step23_hk_map.py. For planar
# disorder with an in-plane displacement R between faulted blocks the diffuse
# intensity on the (h,k) rod carries a factor 1 - cos(2 pi (h,k).R): it vanishes
# where (h,k).R is an integer and peaks where it is a half-integer, so WHICH rods
# carry diffuse intensity measures R. A Ruddlesden-Popper offset R = (1/2, 1/2)
# predicts rods where h + k is ODD and none where it is EVEN.

def centred_L_nodes(h: int, k: int, L_min: float, L_max: float, centring: str = "P") -> np.ndarray:
    """Integer L at which the (h, k) rod has an ALLOWED Bragg node, by lattice centring.

    ``"P"`` every L; ``"I"`` h + k + L even; ``"F"`` h, k, L all the same parity
    (none at all if h and k differ in parity); ``"C"`` h + k even (then every L).
    The between-node exclusion of :func:`diffuse_to_bragg` must come from THIS,
    not from a fixed integer grid: under I-centring even and odd (h + k) rows have
    their nodes at L of opposite parity.
    """
    Ls = np.arange(int(math.ceil(L_min)), int(math.floor(L_max)) + 1)
    c = centring.upper()
    if c == "P":
        keep = np.ones(len(Ls), bool)
    elif c == "I":
        keep = (h + k + Ls) % 2 == 0
    elif c == "F":
        keep = ((h % 2) == (k % 2)) & ((Ls % 2) == (h % 2))
    elif c == "C":
        keep = np.full(len(Ls), (h + k) % 2 == 0)
    else:
        raise ValueError(f"centring must be one of P, I, F, C; got {centring!r}")
    return Ls[keep].astype(float)


def diffuse_to_bragg(rod: RodProfile, control: RodProfile, bragg_L, *, near_bragg: float = 0.35,
                     bragg_core: float = 0.12, exclude=None, min_points: int = 60,
                     min_bragg_points: int = 10, min_peak_sigma: float = 10.0) -> dict:
    """Diffuse level between the Bragg nodes, over the node height, on ONE rod.

    **The normalisation is the whole point.** Raw diffuse intensity also scales
    with the parent structure factor ``|F(h,k)|^2``, so a row with weak Bragg
    peaks has a weak rod for a trivial reason, and ranking rods by raw level
    recovers "rods where the reflections are bright" -- not a selection rule.
    Compare this ratio across (h, k): flat means no rule, a switch with the
    parity of (h, k) . R is a fault vector.

    ``rod`` and ``control`` come from :func:`profile_along` on :func:`rod_path`
    and :func:`matched_control_path`. ``bragg_L`` are the allowed nodes
    (:func:`centred_L_nodes`). ``exclude`` is an optional boolean mask aligned
    with ``rod.L`` for ring crossings. Points within ``near_bragg`` of a node are
    kept out of the diffuse level; points within ``bragg_core`` of one give the
    peak (95th percentile). The control's median and robust sigma give the
    significance of the diffuse level.

    Returns a dict with ``usable`` and ``reason``, and when usable ``diffuse``,
    ``peak``, ``ratio``, ``sigma``, ``n_diffuse``, ``n_bragg``.

    **A row with no normaliser is not evidence.** If no allowed node reaches the
    Ewald sphere inside the measured omega range the ratio is noise over noise;
    the row comes back ``usable=False`` and must be reported as unusable, never
    as a zero. A node counts only if its peak stands ``min_peak_sigma`` control
    sigmas above THIS rod's own between-node level -- not above the control: a
    real diffuse floor already clears the control, which is the quantity being
    measured, so comparing the node to the control would pass a row with no node
    at all. (The original required 1e3 counts; this form transfers between
    exposures.)
    """
    L = np.asarray(rod.L, float)
    I = np.asarray(rod.intensity, float)
    nodes = np.asarray(bragg_L, float).ravel()
    out = dict(usable=False, reason="")
    if nodes.size == 0:
        out["reason"] = "no allowed Bragg node on this rod"
        return out
    ctl = np.asarray(control.intensity, float)
    ctl = ctl[np.isfinite(ctl)]
    if ctl.size < min_points:
        out["reason"] = f"control has {ctl.size} observable points (< {min_points})"
        return out
    cb = float(np.median(ctl))
    cs = 1.4826 * float(np.median(np.abs(ctl - cb)))
    if not np.isfinite(cs) or cs <= 0:
        out["reason"] = "control noise is zero or undefined"
        return out
    dist = np.min(np.abs(L[:, None] - nodes[None, :]), axis=1)
    ok = np.isfinite(I)
    if exclude is not None:
        ok &= ~np.asarray(exclude, bool)
    dif = ok & (dist >= near_bragg)
    brg = ok & (dist < bragg_core)
    if int(dif.sum()) < min_points:
        out["reason"] = f"{int(dif.sum())} observable between-node points (< {min_points})"
        return out
    if int(brg.sum()) < min_bragg_points:
        out["reason"] = f"{int(brg.sum())} observable points at Bragg nodes (< {min_bragg_points})"
        return out
    peak = float(np.percentile(I[brg], 95))
    d = float(np.median(I[dif]))
    if (peak - d) / cs < min_peak_sigma:
        out["reason"] = ("no Bragg node reaches the Ewald sphere in this omega range -- no "
                         "normaliser; not evidence of anything")
        return out
    out.update(usable=True, diffuse=d, peak=peak, ratio=d / peak, sigma=(d - cb) / cs,
               n_diffuse=int(dif.sum()), n_bragg=int(brg.sum()))
    return out
