"""The physical forward path for an ID03 mu x chi x obpitch scan, from inverted maps.

:mod:`midas_dfxm.forward_maps` regenerates frames with an EMPIRICAL separable Gaussian. This module
does the same job through the package's own physics: a one-voxel-thick "slab" with one voxel per
detector pixel, a local deformation gradient built from the tilt and d-spacing maps, and the
reciprocal-space resolution acceptance of :class:`~midas_dfxm.resolution.ResolutionFunction`
evaluated at every (mu, chi, obpitch) setting with the goniometer rotations of
:class:`~midas_dfxm.conventions.GoniometerSetting`.

What it is and is not
---------------------
* One reflection, one layer, samx/samy/samz never moved: the maps are a 2-D PROJECTION. Depth is
  unresolved (ledger KL-008), so a "voxel" here is a pixel's column, and the slab is not a 3-D
  microstructure. The projection onto the detector is taken as the identity (one voxel per pixel);
  the row foreshortening ``1/sin(2 theta_B)`` is a display scale, not part of this model.
* Per-pixel intrinsic mosaic is zero in this model: a voxel diffracts exactly when its lattice
  meets the acceptance. Every pixel therefore has the SAME rocking width, set by the resolution
  widths you pass. At 1.16 points per FWHM in mu the intrinsic and instrument parts cannot be
  separated from the data (README rule 12 / ENVELOPE section 6), so the default widths are the
  grain-median measured widths, which already contain the intrinsic part.
* The sign chain tilt -> lattice rotation is fixed ALGEBRAICALLY (a voxel diffracts at the setting
  that undoes its lattice rotation) and checked by recovering the input maps from the rendered
  curves (:func:`recovered_centres`). That is a code-consistency check, not a measurement of the
  beamline's motor signs: ID03 ``mu`` rotates about -y and ``chi`` about +x (survey of this
  campaign), taken as given.

Frames: MIDAS lab frame, x = beam, y = outboard, z = up; vertical scattering plane
(``ScatteringGeometry()`` default, ESRF). Angles in degrees, wavelength in angstroms; reciprocal
vectors carry the 2 pi convention.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .conventions import GoniometerSetting, ScatteringGeometry
from .field import DeformationField
from .resolution import ResolutionFunction

__all__ = ["ThinSlab", "thin_slab_from_maps", "resolution_for_plane", "render_weights",
           "recovered_centres", "fit_amplitudes", "marginal_sigmas", "match_marginal_widths",
           "physical_references", "physical_score", "physical_totals"]

DT = torch.float64


@dataclass
class ThinSlab:
    """A one-voxel-thick field and the references it was built around."""

    field: DeformationField
    hkl: tuple
    two_theta0_deg: float
    wavelength_A: float
    mu_ref_deg: float
    chi_ref_deg: float
    geometry: ScatteringGeometry

    @property
    def q_mag(self) -> float:
        """``|Q|`` of the reference reflection, ``2 pi / d``, 1/A."""
        return 4.0 * math.pi * math.sin(math.radians(0.5 * self.two_theta0_deg)) / self.wavelength_A


def _rodrigues_align(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Rotation matrix taking unit vector ``a`` onto unit vector ``b``."""
    a = a / torch.linalg.vector_norm(a); b = b / torch.linalg.vector_norm(b)
    v = torch.linalg.cross(a, b)
    c = torch.dot(a, b)
    if torch.linalg.vector_norm(v) < 1e-14:
        return torch.eye(3, dtype=a.dtype)
    K = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=a.dtype)
    return torch.eye(3, dtype=a.dtype) + K + K @ K * (1.0 / (1.0 + c))


def _rot(axis: tuple, deg: torch.Tensor) -> torch.Tensor:
    """Right-handed rotation about a unit ``axis`` by ``deg`` degrees, batched over ``deg`` ``(N,)``."""
    ax = torch.as_tensor(axis, dtype=DT)
    ax = ax / torch.linalg.vector_norm(ax)
    t = torch.deg2rad(deg)
    c, s = torch.cos(t), torch.sin(t)
    K = torch.tensor([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]], dtype=DT)
    I = torch.eye(3, dtype=DT)
    return I[None] + s[:, None, None] * K[None] + (1 - c)[:, None, None] * (K @ K)[None]


def thin_slab_from_maps(tilt_mu_deg, tilt_chi_deg, strain, *, two_theta0_deg: float,
                        wavelength_A: float, hkl=(0, 0, 2), mu_ref_deg: float = 0.0,
                        chi_ref_deg: float = 0.0, pitch_um: float = 1.0, shape=None,
                        geometry=None) -> ThinSlab:
    """One voxel per map pixel, rotated by the tilt maps and stretched by the strain map.

    ``tilt_mu_deg`` and ``tilt_chi_deg`` are the per-pixel centres of mass of the mu and chi
    marginals minus their grain medians (what :func:`~midas_dfxm.multiplane.strain_and_tilt`
    returns); ``strain`` is the relative ``delta d / d`` (same function). A voxel diffracts when the
    goniometer undoes its lattice rotation, so it is rotated by ``+tilt_mu`` about +y (outboard,
    because ID03 ``mu`` turns about -y) and by ``-tilt_chi`` about +x (the beam). The stretch is the
    isotropic ``F = (1 + strain) I``: ``Q' = F^-T Q0`` shrinks ``|Q|`` by ``1 + strain``, i.e.
    ``strain = delta d / d``. One reflection cannot tell this from a composition change.
    """
    geom = geometry or ScatteringGeometry()
    tmu = torch.as_tensor(np.asarray(tilt_mu_deg).ravel(), dtype=DT)
    tch = torch.as_tensor(np.asarray(tilt_chi_deg).ravel(), dtype=DT)
    eps = torch.as_tensor(np.asarray(strain).ravel(), dtype=DT)
    N = tmu.numel()
    Ry = _rot(geom.frame.outboard, tmu)                      # lattice rotation about +y
    Rx = _rot(geom.frame.beam, -tch)                         # and about +x
    R = Rx @ Ry
    F = R * (1.0 + eps)[:, None, None]
    # reference orientation: the reflection's G0 lies on the Bragg Q direction at two_theta0
    k_in = geom.beam_direction(dtype=DT)
    k_out = geom.k_out(two_theta0_deg, dtype=DT)
    q_dir = (k_out - k_in) / torch.linalg.vector_norm(k_out - k_in)
    d002 = wavelength_A / (2.0 * math.sin(math.radians(0.5 * two_theta0_deg)))
    h, k, l = hkl
    # c-axis-only reflection: set c so that d(hkl) = d002 * 2 / l for 00l, a from the ideal hcp ratio
    c = d002 * l if (h == 0 and k == 0) else d002 * 2.0
    a = c / 1.624
    from .field import reciprocal_basis
    lat = torch.tensor([a, a, c, 90.0, 90.0, 120.0], dtype=DT)
    G_crystal = reciprocal_basis(lat) @ torch.tensor(hkl, dtype=DT)
    OM0 = _rodrigues_align(G_crystal, q_dir)
    # slab positions on a unit grid, only so the field has coordinates (not used by the model)
    nx = int(shape[1]) if shape is not None else N
    ny = int(shape[0]) if shape is not None else 1
    xx, yy = torch.meshgrid(torch.arange(nx, dtype=DT) * pitch_um,
                            torch.arange(ny, dtype=DT) * pitch_um, indexing="xy")
    pos = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros(nx * ny, dtype=DT)], 1)[:N]
    field = DeformationField(positions=pos, F=F, reference_orientation=OM0, lattice_params=lat,
                             shape=(nx, ny, 1) if shape is not None else None)
    return ThinSlab(field=field, hkl=tuple(hkl), two_theta0_deg=float(two_theta0_deg),
                    wavelength_A=float(wavelength_A), mu_ref_deg=float(mu_ref_deg),
                    chi_ref_deg=float(chi_ref_deg), geometry=geom)


def resolution_for_plane(slab: ThinSlab, ob_deg: float, *, sigma_mu_deg: float, sigma_chi_deg: float,
                         sigma_ob_deg: float, flat_top_ob_deg=None, couple=True) -> ResolutionFunction:
    """Acceptance centred on the Q the objective accepts at ``obpitch = ob_deg`` (= 2 theta).

    Widths are the measured Gaussian sigmas of the three scan axes, turned into reciprocal-space
    widths by the rate at which each motor moves ``Q`` through the acceptance:

    * rock (``mu``, about the plane normal): ``|Q| * sigma_mu``;
    * roll (``chi``, about the beam): ``|Q| cos(theta) * sigma_chi``;
    * longitudinal (``obpitch``): ``(2 pi / lambda) cos(theta) * sigma_ob``, the component of the
      accepted-vector shift ``(2 pi / lambda) d k_out`` along ``Q``.

    ``flat_top_ob_deg=(width, softness)`` (degrees of obpitch) replaces the Gaussian longitudinal acceptance by an
    erf-edged box of that width and edge softness, converted with the same rate. ``sigma_ob_deg`` is then ignored.
    The observed obpitch response of the Mg-4Al ID03 scan is a flat top, not a Gaussian (see
    :mod:`midas_dfxm.forward_profile`).
    """
    geom = slab.geometry
    k_in = geom.beam_direction(dtype=DT)
    k_out = geom.k_out(ob_deg, dtype=DT)
    q_nom = (2.0 * math.pi / slab.wavelength_A) * (k_out - k_in)
    alpha = 1.0 if couple is True else (0.0 if couple is False else float(couple))
    if alpha != 1.0:
        # scale the TRANSVERSE (rock) part of the accepted-vector shift away from the reference setting by ``alpha``:
        # 1 = the geometry's full coupling, 0 = only the longitudinal part is kept (a diagnostic switch, not a physical statement)
        q0 = (2.0 * math.pi / slab.wavelength_A) * (geom.k_out(slab.two_theta0_deg, dtype=DT) - k_in)
        e0 = q0 / torch.linalg.vector_norm(q0)
        dq = q_nom - q0
        q_par = torch.dot(dq, e0) * e0
        q_nom = q0 + q_par + alpha * (dq - q_par)
    th = math.radians(0.5 * slab.two_theta0_deg)
    qm = slab.q_mag
    rad = math.radians
    _arr = lambda v: torch.as_tensor(np.asarray(v, float), dtype=DT) if np.ndim(v) else v
    s_rock = _arr(qm * np.radians(np.asarray(sigma_mu_deg, float)))
    s_roll = _arr(qm * math.cos(th) * np.radians(np.asarray(sigma_chi_deg, float)))
    s_par = (2.0 * math.pi / slab.wavelength_A) * math.cos(th) * rad(sigma_ob_deg)
    # the in-plane direction perpendicular to Q: rotate Q by 90 degrees about the plane normal
    n = geom.plane_normal(dtype=DT)
    e_par = q_nom / torch.linalg.vector_norm(q_nom)
    rock_dir = torch.linalg.cross(n, e_par)
    par_flat = None
    if flat_top_ob_deg is not None:
        rate = (2.0 * math.pi / slab.wavelength_A) * math.cos(th)         # 1/A per radian of obpitch
        par_flat = (rate * rad(flat_top_ob_deg[0]), rate * rad(flat_top_ob_deg[1]))
    return ResolutionFunction(q_nom=q_nom, sigma_par=s_par, sigma_perp=s_rock,
                              sigma_perp2=s_roll, rock_dir=rock_dir, par_flat=par_flat)


def render_weights(slab: ThinSlab, mu_deg, chi_deg, ob_deg, *, sigma_mu_deg: float,
                   sigma_chi_deg, sigma_ob_deg: float = 0.0, flat_top_ob_deg=None, couple=True) -> np.ndarray:
    """Unit-peak acceptance weights ``(n_mu, n_chi, n_ob, P)`` for the whole scan.

    ``mu_deg`` and ``chi_deg`` are the ABSOLUTE motor grids; the slab's references
    (``mu_ref_deg``, ``chi_ref_deg``) are subtracted here. ``ob_deg`` is the obpitch grid.
    """
    mu = np.asarray(mu_deg, float); chi = np.asarray(chi_deg, float); ob = np.asarray(ob_deg, float)
    Q = slab.field.local_Q(slab.hkl)                                  # (P, 3), sample frame
    P = Q.shape[0]
    out = np.empty((len(mu), len(chi), len(ob), P), np.float32)
    res = [resolution_for_plane(slab, o, sigma_mu_deg=sigma_mu_deg, sigma_chi_deg=sigma_chi_deg,
                                sigma_ob_deg=sigma_ob_deg, flat_top_ob_deg=flat_top_ob_deg, couple=couple) for o in ob]
    # ID03 mu turns about -y, the package's phi about +y: phi = -(mu - mu_ref); chi is the same axis
    for i, m in enumerate(mu):
        for j, c in enumerate(chi):
            G = GoniometerSetting(phi=-(float(m) - slab.mu_ref_deg),
                                  chi=float(c) - slab.chi_ref_deg).sample_rotation(dtype=DT)
            Ql = Q @ G.transpose(-1, -2)
            for k, r in enumerate(res):
                out[i, j, k, :] = r.weight(Ql).numpy()
    return out


def recovered_centres(W: np.ndarray, mu_deg, chi_deg, ob_deg):
    """Centre of mass of the rendered curve along each axis, per voxel ``(P,)`` x 3.

    The round trip that fixes the sign chain algebraically: feed the maps in, render, take the
    moments the inversion takes, and the input maps must come back.
    """
    mu = np.asarray(mu_deg, float); chi = np.asarray(chi_deg, float); ob = np.asarray(ob_deg, float)
    W = W.astype(np.float64)
    Mm = W.sum((1, 2)); Mc = W.sum((0, 2)); Mo = W.sum((0, 1))
    com = lambda M, g: (M * g[:, None]).sum(0) / np.maximum(M.sum(0), 1e-30)
    return com(Mm, mu), com(Mc, chi), com(Mo, ob)


def fit_amplitudes(W: np.ndarray, mob_train: np.ndarray, base_ob: np.ndarray, train) -> np.ndarray:
    """Per-pixel amplitude: least squares of the training planes' totals on the rendered totals.

    ``mob_train`` ``(n_train, P)`` are the training planes' summed counts, ``base_ob`` ``(P,)`` their
    angular baseline, ``train`` the plane indices into ``W``'s third axis. Clipped at zero.
    """
    Wt = W[:, :, list(train), :].sum((0, 1)).astype(np.float64)       # (n_train, P)
    S = mob_train - base_ob[None]
    return np.maximum((S * Wt).sum(0) / np.maximum((Wt ** 2).sum(0), 1e-30), 0.0)


def marginal_sigmas(W: np.ndarray, mu_deg, chi_deg, ob_deg) -> np.ndarray:
    """Second-moment sigma of the rendered curve summed over the other two axes, per voxel ``(3, P)``.

    These are the marginals the inversion measures (`angular_moments` of the summed frames), so they
    are what a measured grain-median width should be compared with. Moving ``obpitch`` shifts the
    accepted vector partly along the rock direction, so the mu marginal summed over obpitch is wider
    than the single-plane mu width; this function measures what the data measures.
    """
    W = W.astype(np.float64)
    out = []
    for M, g in ((W.sum((1, 2)), np.asarray(mu_deg, float)), (W.sum((0, 2)), np.asarray(chi_deg, float)),
                 (W.sum((0, 1)), np.asarray(ob_deg, float))):
        w = M / np.maximum(M.sum(0), 1e-30)
        c = (w * g[:, None]).sum(0)
        out.append(np.sqrt((w * (g[:, None] - c) ** 2).sum(0)))
    return np.stack(out)


def match_marginal_widths(slab: ThinSlab, mu_deg, chi_deg, ob_deg, target_sigma_deg, *,
                          n_iter: int = 4, tol: float = 0.02, flat_top_ob_deg=None, couple=True):
    """Resolution widths whose RENDERED marginal sigmas equal ``target_sigma_deg`` ``(mu, chi, ob)``.

    A fixed-point scaling on a single untilted, unstrained voxel at the slab's references. The
    targets are the measured grain-median marginal widths; the returned widths are what to pass to
    :func:`render_weights`. Returns ``(sigmas_deg, achieved_deg)``. With ``flat_top_ob_deg`` the obpitch
    acceptance is that box and only the mu and chi widths are matched (the obpitch target is then ignored and the
    returned obpitch sigma is the requested one unchanged).
    """
    ref = thin_slab_from_maps([0.0], [0.0], [0.0], two_theta0_deg=slab.two_theta0_deg,
                              wavelength_A=slab.wavelength_A, hkl=slab.hkl,
                              mu_ref_deg=slab.mu_ref_deg, chi_ref_deg=slab.chi_ref_deg,
                              geometry=slab.geometry)
    tgt = np.asarray(target_sigma_deg, float)
    cur = tgt.copy()
    for _ in range(n_iter):
        W = render_weights(ref, mu_deg, chi_deg, ob_deg, sigma_mu_deg=cur[0], sigma_chi_deg=cur[1],
                           sigma_ob_deg=cur[2], flat_top_ob_deg=flat_top_ob_deg, couple=couple)
        ach = marginal_sigmas(W, mu_deg, chi_deg, ob_deg)[:, 0]
        use = slice(0, 2) if flat_top_ob_deg is not None else slice(0, 3)
        if np.all(np.abs(ach[use] / tgt[use] - 1) < tol):
            break
        cur[use] = (cur * tgt / ach)[use]
    return cur, ach


def physical_references(mu0, chi0, c_steps, mask, ob_g, step, wavelength_A, mu_g, chi_g, target_sigma_deg, *,
                        flat_top_ob_deg=None, couple=True):
    """Fold-global references and matched widths for :func:`physical_score`.

    ``mu0``, ``chi0`` (deg) and ``c_steps`` (obpitch centre in motor steps from plane 0) are per-pixel; ``mask`` selects the pixels
    whose medians define the references; ``target_sigma_deg`` = ``(mu, chi, obpitch)`` measured grain-median marginal sigmas.
    Returns ``dict(mu_ref, chi_ref, ob_ref, sigma_mu, sigma_chi, sigma_ob)`` (``sigma_ob`` only meaningful without a flat top).
    """
    ob_c = ob_g[0] + np.asarray(c_steps) * step
    ob_ref = float(np.median(ob_c[mask])); mu_ref = float(np.median(np.asarray(mu0)[mask]))
    chi_ref = float(np.median(np.asarray(chi0)[mask]))
    ref_slab = thin_slab_from_maps([0.0], [0.0], [0.0], two_theta0_deg=ob_ref, wavelength_A=wavelength_A,
                                   mu_ref_deg=mu_ref, chi_ref_deg=chi_ref)
    sig, _ = match_marginal_widths(ref_slab, mu_g, chi_g, ob_g, list(target_sigma_deg), flat_top_ob_deg=flat_top_ob_deg,
                                  couple=couple)
    return dict(mu_ref=mu_ref, chi_ref=chi_ref, ob_ref=ob_ref, sigma_mu=float(sig[0]), sigma_chi=float(sig[1]),
                sigma_ob=float(sig[2]))


def physical_score(obs, mu0, chi0, c_steps, mob_train, base_ob, ob_g, k, mu_g, chi_g, refs, step, wavelength_A, *,
                   flat_top_ob_deg=None, chunk: int = 50_000, return_pred: bool = False, couple=True,
                   per_pixel_widths=None, amplitude=None):
    """Per-pixel SSE and SST of the thin-slab physical forward against ``obs`` ``(n_mu, n_chi, P)`` of held-out plane ``k``.

    Everything per pixel comes from the TRAINING planes: ``mu0``, ``chi0``, ``c_steps`` (centres), ``mob_train`` ``(n_train, P)`` (their
    totals) and ``base_ob``. The slab is rendered for ALL planes (it has no free shape parameters), the amplitude is the
    least squares of the training planes' totals on the rendered ones, and plane ``k`` is predicted from it.
    ``couple`` is the obpitch-to-rock coupling strength of the accepted vector (``True`` = 1, ``False`` = 0, or any float); ``per_pixel_widths=(smu, schi, tmu, tchi)`` gives each
    voxel its own rock and roll width, ``smu_i * refs['sigma_mu'] / tmu`` (the fold-global factor keeps the coupling correction).
    ``amplitude`` (``(P,)``) replaces the least-squares amplitude; only plane ``k`` is then rendered. Pixels are independent,
    so the work is chunked; ``refs`` are fold-global (:func:`physical_references`). With ``return_pred`` a third
    value, the prediction ``(n_mu, n_chi, P)`` float32, is returned (small ``P`` only).
    """
    ob_g = np.asarray(ob_g, float)
    train = [j for j in range(len(ob_g)) if j != k]
    n_mu, n_chi = len(mu_g), len(chi_g)
    P = obs.shape[-1]
    sse = np.empty(P); sst = np.empty(P)
    pred_all = np.empty(obs.shape, np.float32) if return_pred else None
    th = np.deg2rad(refs["ob_ref"]) / 2
    for a in range(0, P, chunk):
        b = min(a + chunk, P)
        ob_c = ob_g[0] + np.asarray(c_steps[a:b]) * step
        strain = -(np.deg2rad(ob_c - refs["ob_ref"]) / 2) / np.tan(th)
        slab = thin_slab_from_maps(np.asarray(mu0[a:b]) - refs["mu_ref"], np.asarray(chi0[a:b]) - refs["chi_ref"], strain,
                                   two_theta0_deg=refs["ob_ref"], wavelength_A=wavelength_A,
                                   mu_ref_deg=refs["mu_ref"], chi_ref_deg=refs["chi_ref"])
        if per_pixel_widths is None:
            sg_mu, sg_chi = refs["sigma_mu"], refs["sigma_chi"]
        else:
            smu, schi, tmu, tchi = per_pixel_widths
            sg_mu = np.asarray(smu[a:b]) * refs["sigma_mu"] / tmu
            sg_chi = np.asarray(schi[a:b]) * refs["sigma_chi"] / tchi
        if amplitude is None:
            W = render_weights(slab, mu_g, chi_g, ob_g, sigma_mu_deg=sg_mu, sigma_chi_deg=sg_chi,
                               sigma_ob_deg=refs["sigma_ob"], flat_top_ob_deg=flat_top_ob_deg, couple=couple)
            A = fit_amplitudes(W, mob_train[:, a:b], base_ob[a:b], train)
            Wk = W[:, :, k, :]
        else:
            W = render_weights(slab, mu_g, chi_g, ob_g[[k]], sigma_mu_deg=sg_mu, sigma_chi_deg=sg_chi,
                               sigma_ob_deg=refs["sigma_ob"], flat_top_ob_deg=flat_top_ob_deg, couple=couple)
            A = np.asarray(amplitude[a:b], float)
            Wk = W[:, :, 0, :]
        pred = (base_ob[a:b] / (n_mu * n_chi))[None, None] + A[None, None] * Wk
        y = obs[:, :, a:b].astype(np.float64)
        sse[a:b] = ((y - pred) ** 2).sum((0, 1))
        sst[a:b] = ((y - y.mean((0, 1), keepdims=True)) ** 2).sum((0, 1))
        if return_pred:
            pred_all[:, :, a:b] = pred
        del W
    return (sse, sst, pred_all) if return_pred else (sse, sst)


def physical_totals(mu0, chi0, c_steps, ob_g, mu_g, chi_g, refs, step, wavelength_A, *, flat_top_ob_deg=None, couple=True,
                    per_pixel_widths=None, chunk: int = 50_000):
    """Rendered totals ``(n_ob, P)``: the sum over all (mu, chi) settings of each plane's weights, per pixel (same slab as :func:`physical_score`)."""
    ob_g = np.asarray(ob_g, float)
    P = len(mu0)
    out = np.empty((len(ob_g), P))
    th = np.deg2rad(refs["ob_ref"]) / 2
    for a in range(0, P, chunk):
        b = min(a + chunk, P)
        ob_c = ob_g[0] + np.asarray(c_steps[a:b]) * step
        strain = -(np.deg2rad(ob_c - refs["ob_ref"]) / 2) / np.tan(th)
        slab = thin_slab_from_maps(np.asarray(mu0[a:b]) - refs["mu_ref"], np.asarray(chi0[a:b]) - refs["chi_ref"], strain,
                                   two_theta0_deg=refs["ob_ref"], wavelength_A=wavelength_A,
                                   mu_ref_deg=refs["mu_ref"], chi_ref_deg=refs["chi_ref"])
        if per_pixel_widths is None:
            sg_mu, sg_chi = refs["sigma_mu"], refs["sigma_chi"]
        else:
            smu, schi, tmu, tchi = per_pixel_widths
            sg_mu = np.asarray(smu[a:b]) * refs["sigma_mu"] / tmu
            sg_chi = np.asarray(schi[a:b]) * refs["sigma_chi"] / tchi
        W = render_weights(slab, mu_g, chi_g, ob_g, sigma_mu_deg=sg_mu, sigma_chi_deg=sg_chi, sigma_ob_deg=refs["sigma_ob"],
                           flat_top_ob_deg=flat_top_ob_deg, couple=couple)
        out[:, a:b] = W.sum((0, 1), dtype=np.float64)
        del W
    return out
