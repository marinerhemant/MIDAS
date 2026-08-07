"""Takagi-Taupin dynamical diffraction: differentiable 2-beam solver for DFXM.

The geometrical/kinematical forward (:mod:`midas_dfxm.forward`) reads the per-pixel
centroid shift ``dQ`` and is valid in the weakly-deformed regime; dynamical
diffraction (extinction, Pendellosung, the finite Darwin width, and the
characteristic black-white dislocation contrast) is the sharpest physical bound on
it. This module adds the *dynamical* diffracted exit wave by integrating the
two-beam Takagi-Taupin equations through the (possibly deformed) crystal.

Formulation (symmetric Laue, transmission -- the classic dynamical-topography
geometry, and where the analytic checks live). The two slowly-varying amplitudes
``D0`` (forward) and ``Dh`` (diffracted) obey, marching in depth ``z``::

    d/dz [D0; Dh] = A(x, z) [D0; Dh],

    A = i*pi/lambda * [[ chi0/g0,                 C*chihbar*e^{+i h.u}/g0 ],
                       [ C*chih*e^{-i h.u}/gh,     chi0/gh                ]]
        + i * [[0, 0], [0, 2*kappa*y]],

with ``g0 = gh = cos(theta_B)`` the direction cosines, ``C`` the polarization
factor (1 for sigma, cos 2theta for pi), ``y`` the dimensionless deviation
parameter (0 at exact Bragg), ``kappa = pi C |chih| / (lambda sqrt(g0 gh)) = pi/Lambda``
the coupling rate (``Lambda`` the extinction / Pendellosung length), and
``h.u(x, z)`` the local lattice displacement projected on the reflection -- the
*only* place the deformation enters (Takagi's phase form). With ``h.u = 0`` and a
uniform crystal the solver reproduces, to machine precision, the closed-form Laue
solution ``|Dh|^2 = sin^2(pi t/Lambda sqrt(1+y^2)) / (1 + y^2)`` -- the primary
validation gate (:func:`laue_intensity_analytic`).

The susceptibility Fourier coefficients come from ``midas_hkls`` structure factors
(never an external x-ray library). Everything is torch-differentiable in the
susceptibility, geometry, deviation, and the displacement field; the depth march is
a loop of 2x2 matrix exponentials, so gradients flow and (for large volumes)
gradient checkpointing bounds memory.

Units: wavelength in Angstrom; lengths (thickness, dx) in micrometers.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

# Classical electron radius, Angstrom.
R_E_A = 2.8179403262e-5
_A_PER_UM = 1.0e4


# --------------------------------------------------------------------------- physics
def susceptibility_fourier(crystal, hkl, *, wavelength_A: float, absorption: bool = True,
                           dtype=torch.float64):
    """Return ``(chi0, chih, chihbar)`` -- the susceptibility Fourier coefficients.

    ``chi_h = -(r_e lambda^2 / (pi V)) F_h`` with ``F_h`` the (complex) structure
    factor from ``midas_2d``/``midas_hkls`` and ``V`` the cell volume.

    ``absorption`` folds photoabsorption (anomalous ``f''``) into the imaginary
    parts, so the solver damps Pendellosung fringes and models transmission. It adds
    the **exact** ``f''``-weighted structure factor to every coefficient,
    ``chi_g += 1j (r_e lambda^2 / pi V) sum_j occ_j f''_j e^{2pi i g.r_j} DWF_j`` (the
    ``f0(Q)`` variation between reflections is captured, not approximated by a single
    dispersion ratio). ``Im(chi0)`` reduces to ``(r_e lambda^2/pi V) sum occ f''``.
    Set ``absorption=False`` for the non-absorbing (real-chi) limit used by the
    analytic validation. ``f''`` comes from ``midas_hkls.anomalous`` (never an
    external x-ray library).
    """
    from midas_2d.forward import structure_factor_q
    from midas_hkls.crystal_torch import crystal_to_tensor
    from midas_hkls.lattice_torch import cell_volume

    ct = crystal_to_tensor(crystal, dtype=dtype)
    V = float(cell_volume(ct.lattice_params))                       # Angstrom^3
    pref = -(R_E_A * wavelength_A ** 2) / (math.pi * V)

    def chi(idx):
        F = structure_factor_q(ct, torch.as_tensor(idx, dtype=dtype))
        return pref * complex(float(F.real), float(F.imag))

    h = tuple(int(v) for v in hkl)
    chi0, chih, chihbar = chi((0, 0, 0)), chi(h), chi((-h[0], -h[1], -h[2]))

    if absorption:
        from functools import lru_cache
        from midas_hkls.anomalous import anomalous_correction
        from midas_hkls.lattice_torch import d_spacing

        @lru_cache(maxsize=None)
        def fpp(el):
            _, fdp = anomalous_correction([el], wavelength_A)
            return float(fdp[0]) if hasattr(fdp, "__len__") else float(fdp)

        atoms = crystal.unit_cell_atoms()

        def Fpp(idx):
            """f''-weighted structure factor sum_j occ_j f''_j exp(2pi i g.r_j) DWF_j."""
            if idx == (0, 0, 0):
                s2 = 0.0                                          # forward: DWF=1, phase=0
            else:
                d = float(d_spacing(torch.tensor(idx, dtype=dtype), ct.lattice_params))
                s2 = (1.0 / (2.0 * d)) ** 2                       # (sin theta / lambda)^2
            tot = 0.0 + 0.0j
            for a in atoms:
                f = a.fract
                ph = 2.0 * math.pi * (idx[0] * f[0] + idx[1] * f[1] + idx[2] * f[2])
                dwf = math.exp(-float(getattr(a, "B_iso", 0.0)) * s2)
                tot += (float(getattr(a, "occupancy", 1.0)) * fpp(a.element) * dwf
                        * complex(math.cos(ph), math.sin(ph)))
            return tot

        # exact f'' imaginary part; my sign convention has absorption = +Im (wave decays),
        # and -pref > 0, so add 1j*(-pref)*Fpp to each coefficient.
        neg_pref = -pref
        chi0 = chi0 + 1j * neg_pref * Fpp((0, 0, 0))
        chih = chih + 1j * neg_pref * Fpp(h)
        chihbar = chihbar + 1j * neg_pref * Fpp((-h[0], -h[1], -h[2]))
    return chi0, chih, chihbar


def bragg_angle_deg(d_spacing_A: float, wavelength_A: float) -> float:
    """Bragg angle (degrees) from ``lambda = 2 d sin(theta)``."""
    return math.degrees(math.asin(wavelength_A / (2.0 * d_spacing_A)))


def polarization_factor(theta_B_deg: float, mode: str = "sigma") -> float:
    """``C`` = 1 (sigma) or ``cos 2theta`` (pi)."""
    if mode == "sigma":
        return 1.0
    if mode == "pi":
        return abs(math.cos(2.0 * math.radians(theta_B_deg)))
    raise ValueError("mode must be 'sigma' or 'pi'")


def extinction_length(chih: complex, chihbar: complex, *, wavelength_A: float,
                      theta_B_deg: float, C: float = 1.0) -> float:
    """Pendellosung / extinction length ``Lambda`` (micrometers), symmetric Laue.

    ``Lambda = lambda sqrt(g0 gh) / (C |Re sqrt(chih chihbar)|)`` with
    ``g0 = gh = cos(theta_B)``. Returned in micrometers.
    """
    g = math.cos(math.radians(theta_B_deg))
    root = (chih * chihbar) ** 0.5
    Lambda_A = wavelength_A * g / (C * abs(root.real))
    return Lambda_A / _A_PER_UM


def laue_intensity_analytic(thickness_um, y, Lambda_um):
    """Closed-form symmetric-Laue diffracted intensity ``|Dh|^2`` (non-absorbing).

    ``|Dh|^2 = sin^2( pi t / Lambda * sqrt(1 + y^2) ) / (1 + y^2)``. The reference
    the solver must reproduce with ``h.u = 0`` and real susceptibilities.
    """
    t = torch.as_tensor(thickness_um, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    arg = math.pi * t / Lambda_um * torch.sqrt(1.0 + y ** 2)
    return torch.sin(arg) ** 2 / (1.0 + y ** 2)


def deviation_to_angle(y, chih: complex, *, wavelength_A: float, theta_B_deg: float,
                       C: float = 1.0) -> float:
    """Map the dimensionless deviation ``y`` to a glancing-angle offset ``dtheta`` (rad).

    Symmetric case: ``dtheta = y * C |chih| / sin(2 theta_B)``. Provided for real-use
    rocking scans; the solver and its validation are driven by ``y`` directly.
    """
    return float(y) * C * abs(chih) / math.sin(2.0 * math.radians(theta_B_deg))


# --------------------------------------------------------------------------- solver
def _expm2x2(M: torch.Tensor) -> torch.Tensor:
    """Differentiable matrix exponential of a batch of complex ``2x2`` matrices.

    Closed form: ``expm(M) = e^s [ cosh(q) I + sinhc(q) (M - s I) ]`` with
    ``s = tr(M)/2``, ``q = sqrt(s^2 - det M)``, ``sinhc(q) = sinh(q)/q`` (stable as
    ``q -> 0``). ``M`` has shape ``(..., 2, 2)``.
    """
    a = M[..., 0, 0]; b = M[..., 0, 1]; c = M[..., 1, 0]; d = M[..., 1, 1]
    s = 0.5 * (a + d)
    det = a * d - b * c
    q2 = s * s - det
    q = torch.sqrt(q2 + 0j)
    # sinhc = sinh(q)/q, stable near 0 via series; cosh(q).
    small = q.abs() < 1e-6
    sinhc = torch.where(small, 1.0 + q2 / 6.0 + q2 * q2 / 120.0, torch.sinh(q) / q)
    coshq = torch.cosh(q)
    es = torch.exp(s)
    eye = torch.zeros_like(M); eye[..., 0, 0] = 1.0; eye[..., 1, 1] = 1.0
    Ms = M - s[..., None, None] * eye
    return es[..., None, None] * (coshq[..., None, None] * eye + sinhc[..., None, None] * Ms)


def _fourier_shift(field: torch.Tensor, shift_px: float) -> torch.Tensor:
    """Shift a 1-D complex field along its last axis by ``shift_px`` pixels (differentiable)."""
    n = field.shape[-1]
    f = torch.fft.fftfreq(n, device=field.device)
    ramp = torch.exp(-2j * math.pi * f * shift_px)
    return torch.fft.ifft(torch.fft.fft(field) * ramp)


def solve_tt_laue(
    chi0: complex, chih: complex, chihbar: complex, *,
    wavelength_A: float, theta_B_deg: float, thickness_um: float,
    y: float = 0.0, C: float = 1.0,
    hu: Optional[torch.Tensor] = None, dx_um: Optional[float] = None,
    n_depth: int = 400, incident: Optional[torch.Tensor] = None,
    checkpoint: bool = False, dtype=torch.complex128,
):
    """Integrate the 2-beam Takagi-Taupin equations, symmetric Laue transmission.

    Returns ``(D0_exit, Dh_exit)`` complex tensors over the transverse grid.

    ``hu`` -- the local ``h . u`` phase (radians), shape ``(n_depth, nx)`` (depth x
    transverse) for a deformed crystal, or ``None`` for a perfect crystal (plane
    wave, ``nx = 1``). ``dx_um`` is the transverse pixel; required when ``nx > 1``
    for the beam transport. ``incident`` sets ``D0`` at the entrance surface
    (default: uniform 1). ``checkpoint`` wraps each depth step in
    ``torch.utils.checkpoint`` to bound memory for large volumes.
    """
    thetaB = math.radians(theta_B_deg)
    g0 = gh = math.cos(thetaB)
    lam_um = wavelength_A / _A_PER_UM
    kappa = math.pi * C * abs(chih) / (lam_um * math.sqrt(g0 * gh))   # = pi/Lambda
    pre = 1j * math.pi / lam_um

    nx = 1 if hu is None else hu.shape[-1]
    D0 = torch.ones(nx, dtype=dtype) if incident is None else incident.to(dtype)
    Dh = torch.zeros(nx, dtype=dtype)
    dz = thickness_um / n_depth
    # transverse transport per depth step (symmetric Laue: beams diverge by +-tan)
    shift = math.tan(thetaB) * dz / (dx_um if dx_um else 1.0)

    c0 = torch.as_tensor(chi0, dtype=dtype)
    ch = torch.as_tensor(chih, dtype=dtype)
    chb = torch.as_tensor(chihbar, dtype=dtype)
    Cc = torch.as_tensor(C, dtype=dtype)
    dev = 1j * 2.0 * kappa * y

    def step(D0, Dh, iz):
        # beam transport (skip for plane wave)
        if nx > 1:
            D0 = _fourier_shift(D0, +shift / 2)
            Dh = _fourier_shift(Dh, -shift / 2)
        # perfect-crystal coupling phase is 1 (h.u = 0), NOT 0
        phase = torch.ones(nx, dtype=dtype) if hu is None else \
            torch.exp(1j * hu[iz].to(dtype))                     # e^{+i h.u}
        A = torch.zeros(nx, 2, 2, dtype=dtype)
        A[:, 0, 0] = pre * c0 / g0
        A[:, 0, 1] = pre * Cc * chb * phase / g0                 # e^{+i h.u}
        A[:, 1, 0] = pre * Cc * ch * torch.conj(phase) / gh      # e^{-i h.u}
        A[:, 1, 1] = pre * c0 / gh + dev
        U = _expm2x2(A * dz)                                     # (nx,2,2)
        v = torch.stack([D0, Dh], dim=-1).unsqueeze(-1)         # (nx,2,1)
        w = (U @ v).squeeze(-1)
        D0, Dh = w[..., 0], w[..., 1]
        if nx > 1:
            D0 = _fourier_shift(D0, +shift / 2)
            Dh = _fourier_shift(Dh, -shift / 2)
        return D0, Dh

    for iz in range(n_depth):
        if checkpoint:
            D0, Dh = torch.utils.checkpoint.checkpoint(step, D0, Dh, iz, use_reentrant=False)
        else:
            D0, Dh = step(D0, Dh, iz)
    return D0, Dh


def diffracted_intensity(chi0, chih, chihbar, *, thickness_um, y=0.0, **kw):
    """Convenience: ``|Dh_exit|^2`` for a perfect crystal at deviation ``y`` (plane wave)."""
    _, Dh = solve_tt_laue(chi0, chih, chihbar, thickness_um=thickness_um, y=y, **kw)
    return (Dh.abs() ** 2).squeeze()


def darwin_width(chih: complex, chihbar: complex, *, wavelength_A: float,
                 theta_B_deg: float, C: float = 1.0) -> float:
    """Darwin (total-reflection) angular width (radians), symmetric Bragg.

    ``omega_D = 2 C |Re sqrt(chih chihbar)| / sin(2 theta_B)``.
    """
    root = abs(((chih * chihbar) ** 0.5).real)
    return 2.0 * C * root / math.sin(2.0 * math.radians(theta_B_deg))


def solve_tt_bragg(
    chi0: complex, chih: complex, chihbar: complex, *,
    wavelength_A: float, theta_B_deg: float, thickness_um: float,
    y: float = 0.0, C: float = 1.0,
    hu: Optional[torch.Tensor] = None, n_depth: int = 600, center: bool = True,
    dtype=torch.complex128,
):
    """2-beam Takagi-Taupin in **Bragg** (reflection) geometry, symmetric.

    Reflection is a boundary-value problem (the diffracted wave exits the entrance
    surface), so the forward depth march of :func:`solve_tt_laue` does not apply.
    We integrate the **Riccati equation** for the amplitude ratio ``X = Dh/D0``,

        dX/dz = a e^{-i h.u} + b X + c e^{+i h.u} X^2,

    from the back surface (``X = 0``) up to the entrance; ``X`` there is the
    reflected amplitude ratio, and the reflectivity is ``|X|^2 |gamma_h/gamma_0|``.
    This is a stable forward integration and handles the deformation ``h.u(z, x)``
    per column. Returns ``X_surface`` (complex, shape ``(nx,)``); the reflected
    exit wave for imaging is ``X_surface`` times the incident amplitude.

    Perfect-crystal limit: a thick crystal gives the Darwin total-reflection
    plateau (``|X|^2 ~ 1`` for ``|y| < 1``), validated against :func:`darwin_width`.
    Symmetric geometry (``gamma_0 = sin theta_B``, ``gamma_h = -sin theta_B``);
    transverse beam walk-off is omitted (a refinement). Differentiable throughout.
    """
    thetaB = math.radians(theta_B_deg)
    s = math.sin(thetaB)
    g0, gh = s, -s
    lam_um = wavelength_A / _A_PER_UM
    kappa = math.pi * C * abs(chih) / (lam_um * s)                # = pi/Lambda_Bragg
    pre = 1j * math.pi / lam_um
    ch = torch.as_tensor(chih, dtype=dtype)
    chb = torch.as_tensor(chihbar, dtype=dtype)
    a0 = pre * C * ch / gh
    c0 = -pre * C * chb / g0
    # center=True shifts the deviation origin so y=0 is the (refraction-shifted)
    # dynamical Bragg peak (standard reduced-deviation convention). The mean
    # refraction chi0 otherwise puts the total-reflection plateau at y=chi0/|chih|.
    y_eff = y + (chi0.real / (C * abs(chih)) if center else 0.0)
    b = pre * torch.as_tensor(chi0, dtype=dtype) * (1.0 / gh - 1.0 / g0) + 1j * 2.0 * kappa * y_eff

    nx = 1 if hu is None else hu.shape[-1]
    X = torch.zeros(nx, dtype=dtype)
    dz = thickness_um / n_depth

    def rhs(X, iz):
        phase = torch.ones(nx, dtype=dtype) if hu is None else torch.exp(1j * hu[iz].to(dtype))
        return a0 * torch.conj(phase) + b * X + c0 * phase * X * X

    for step in range(n_depth):                                  # march bottom -> top (up)
        iz = n_depth - 1 - step
        h = -dz                                                  # moving in -z (toward surface)
        k1 = rhs(X, iz)
        k2 = rhs(X + 0.5 * h * k1, iz)
        k3 = rhs(X + 0.5 * h * k2, iz)
        k4 = rhs(X + h * k3, iz)
        X = X + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return X


def bragg_reflectivity(chi0, chih, chihbar, *, thickness_um, y=0.0, theta_B_deg, **kw):
    """Convenience: Bragg reflectivity ``|X|^2`` (symmetric, ``|gamma_h/gamma_0| = 1``)."""
    X = solve_tt_bragg(chi0, chih, chihbar, thickness_um=thickness_um, y=y,
                       theta_B_deg=theta_B_deg, **kw)
    return (X.abs() ** 2).squeeze()


# --------------------------------------------------------------------------- pink beam
def pink_deviation_offsets(bandwidth: float, theta_B_deg: float, chih: complex, *,
                           C: float = 1.0, n: int = 21, shape: str = "gaussian",
                           n_sigma: float = 3.0):
    """Map a fractional energy bandwidth to deviation offsets + weights (dimensionless).

    A fractional wavelength shift ``dl/l`` moves the Bragg condition, equivalent to a
    deviation offset ``dy = 2 sin^2(theta_B) (dl/l) / (C |chih|)`` (from
    ``dtheta = tan(theta_B) dl/l`` and ``dy = dtheta sin(2theta)/(C|chih|)``). Returns
    ``(y_offsets, weights)`` sampling the spectrum (``bandwidth`` is the FWHM of
    ``dl/l``; ``shape`` = 'gaussian' or 'flat'). Pink beam = incoherently summing the
    rocking curve over these offsets -- the source of its intensity gain in deformed
    (broadened-peak) microstructures.
    """
    thetaB = math.radians(theta_B_deg)
    dy_per_frac = 2.0 * math.sin(thetaB) ** 2 / (C * abs(chih))
    sigma_y = (bandwidth / 2.35482) * dy_per_frac
    if n <= 1 or sigma_y == 0.0:
        return torch.zeros(1), torch.ones(1)
    ys = torch.linspace(-n_sigma * sigma_y, n_sigma * sigma_y, n, dtype=torch.float64)
    w = torch.exp(-0.5 * (ys / sigma_y) ** 2) if shape == "gaussian" else torch.ones_like(ys)
    return ys, w / w.sum()


def pink_diffracted_intensity(chi0, chih, chihbar, *, thickness_um, y0: float = 0.0,
                              bandwidth: float, theta_B_deg: float, C: float = 1.0,
                              geometry: str = "laue", n_lambda: int = 21,
                              shape: str = "gaussian", **kw):
    """Pink-beam diffracted intensity: the rocking curve integrated over the spectrum.

    ``I = sum_i w_i R(y0 + dy_i)`` with ``(dy_i, w_i)`` from
    :func:`pink_deviation_offsets`. ``geometry`` = 'laue' (``|Dh|^2``) or 'bragg'
    (reflectivity). ``bandwidth = 0`` (or ``n_lambda = 1``) reduces to the
    monochromatic value. As ``bandwidth`` exceeds the Darwin width the integral
    saturates to the integrated reflectivity -- the pink-beam gain.
    """
    ys, w = pink_deviation_offsets(bandwidth, theta_B_deg, chih, C=C, n=n_lambda, shape=shape)
    total = None
    for yo, wi in zip(ys.tolist(), w.tolist()):
        if geometry == "bragg":
            r = bragg_reflectivity(chi0, chih, chihbar, thickness_um=thickness_um,
                                   y=y0 + yo, theta_B_deg=theta_B_deg, C=C, **kw)
        else:
            r = diffracted_intensity(chi0, chih, chihbar, thickness_um=thickness_um,
                                     y=y0 + yo, theta_B_deg=theta_B_deg, C=C, **kw)
        total = wi * r if total is None else total + wi * r
    return total


def solve_tt_2d(
    chi0: complex, chih: complex, chihbar: complex, *,
    wavelength_A: float, theta_B_deg: float, thickness_um: float,
    y: float = 0.0, C: float = 1.0, hu: Optional[torch.Tensor] = None,
    nx: int = 1, nz: int = 400, dx_um: Optional[float] = None,
    geometry: str = "bragg", center: bool = True,
    incident: Optional[torch.Tensor] = None, dtype=torch.complex128,
    backend: str = "dense",
):
    """Full 2D **Bragg** two-beam Takagi-Taupin by a semi-Lagrangian characteristic solve.

    Where the 1D column Riccati (:func:`solve_tt_bragg`) integrates each depth column
    independently, this integrates the coupled TT equations on the whole ``(x, z)`` grid
    with lateral wavefield transport (the Borrmann fan) included. Each beam is transported
    along its OWN characteristic by an exact lateral shift (``D0`` marches ``z: 0->t``,
    ``Dh`` marches ``z: t->0``; both drift ``+cot(theta_B)`` per depth step), with the
    coupling handled by a Crank-Nicolson reaction -- assembled as one linear system with the
    reflection boundary conditions (``D0=incident`` at the entrance, ``Dh=0`` at the back)
    and solved. Using characteristic transport instead of a Cartesian ``d/dx`` difference is
    what makes this **stable** (a naive upwind advection diverges off the Bragg plateau).

    For no interpolation diffusion (and exact energy conservation) choose the depth steps so
    the per-step shift is an integer number of cells: ``nz = round(thickness*cot(theta_B)/dx)``
    (``a=1``). :func:`bragg_2d_nz_for_unit_shift` returns that ``nz``.

    ``backend='dense'`` (default) assembles a torch dense system and solves it with
    :func:`torch.linalg.solve` -- **differentiable** via the linear-solve adjoint, for the
    inverse problem, but O(N^3) and intended for modest grids / the differentiable core.
    ``backend='sparse'`` assembles the same system as a scipy sparse matrix -- forward-only
    (no autograd) but scales to larger grids. Both return the identical system (~1e-13).

    ``hu`` is the ``h.u`` phase, shape ``(nz+1, nx)`` (or ``None`` for a perfect crystal).
    Returns ``(D0, Dh)`` complex grids of shape ``(nz+1, nx)``; the diffracted amplitude that
    exits is ``Dh`` at the entrance surface (row 0), the transmitted ``D0`` at the back (row ``nz``).

    Validated (``tests/test_takagi_taupin_2d.py``): ``nx=1`` reproduces the 1D Riccati Darwin
    curve across the whole rocking curve (max diff ~7e-3, tails converging); ``nx>1`` is stable
    off-plateau and under lateral-grid refinement (no divergence); energy is conserved in the
    plane-wave limit (reflectivity ->0.9999 on the plateau); differentiable via the adjoint.

    Note: only ``geometry='bragg'`` is implemented here. Laue (transmission) transverse
    transport is provided -- validated to ~1e-14 -- by :func:`solve_tt_laue`.
    """
    if geometry == "laue":
        raise NotImplementedError(
            "solve_tt_2d implements Bragg (reflection) geometry; for Laue transverse "
            "transport use solve_tt_laue (validated split-step Fourier-shift march).")
    if geometry != "bragg":
        raise ValueError("geometry must be 'bragg'")
    if backend not in ("dense", "sparse"):
        raise ValueError("backend must be 'dense' or 'sparse'")
    thetaB = math.radians(theta_B_deg)
    s, cth = math.sin(thetaB), math.cos(thetaB)
    lam = wavelength_A / _A_PER_UM
    g0, gh = s, -s                                             # symmetric Bragg (reflection)
    kappa = math.pi * C * abs(chih) / (lam * s)
    pre = 1j * math.pi / lam
    y_eff = y + (chi0.real / (C * abs(chih)) if center else 0.0)
    A00 = pre * chi0 / g0
    Ahh = pre * chi0 / gh + 1j * 2.0 * kappa * y_eff
    A0h = pre * C * chihbar / g0
    Ah0 = pre * C * chih / gh

    nzp = nz + 1
    dz = thickness_um / nz
    idz = 1.0 / dz
    if hu is None:
        hu = torch.zeros(nzp, nx, dtype=torch.float64)
    N0 = nzp * nx
    def i0(i, j): return i * nx + j
    def ih(i, j): return N0 + i * nx + j
    a_cells = (cth / s) * dz / dx_um if (nx > 1 and dx_um) else 0.0   # shift right per step
    m = int(math.floor(a_cells)); f = a_cells - m

    if backend == "sparse":                                    # forward-only, scales
        import numpy as _np
        import scipy.sparse as _sp
        import scipy.sparse.linalg as _spla
        ph = _np.exp(1j * _np.asarray(hu, dtype=float))
        inc = (_np.ones(nx) if incident is None
               else _np.asarray(incident, dtype=complex))
        rows, cols, vals = [], [], []
        add = lambda r, c, v: (rows.append(r), cols.append(c), vals.append(complex(v)))
        conj = _np.conj
        bvec = _np.zeros(2 * N0, dtype=complex)
        set_b = lambda r, v: bvec.__setitem__(r, complex(v))
    else:                                                      # dense torch, differentiable
        ph = torch.exp(1j * hu.to(dtype))
        inc = torch.ones(nx, dtype=dtype) if incident is None else incident
        A = torch.zeros(2 * N0, 2 * N0, dtype=dtype)
        b = torch.zeros(2 * N0, dtype=dtype)
        add = lambda r, c, v: A.__setitem__((r, c), A[r, c] + v)
        conj = torch.conj
        set_b = lambda r, v: b.__setitem__(r, v)

    for j in range(nx):
        ja = (j - m) % nx                                      # value at j comes from j-a (upstream)
        jb = (j - m - 1) % nx
        add(i0(0, j), i0(0, j), 1.0); set_b(i0(0, j), inc[j])  # D0 entrance BC
        add(ih(nz, j), ih(nz, j), 1.0)                         # Dh back BC (=0)
        for i in range(nz):
            # D0 cell (row i0(i+1,j)): (D0[i+1]-S(D0[i]))/dz = 0.5 A00 (D0[i+1]+S(D0[i]))
            #                          + 0.5 A0h (ph[i+1]Dh[i+1] + S(ph[i]Dh[i]))
            r = i0(i + 1, j)
            add(r, i0(i + 1, j), idz - 0.5 * A00)
            add(r, ih(i + 1, j), -0.5 * A0h * ph[i + 1, j])
            add(r, i0(i, ja), (1 - f) * (-idz - 0.5 * A00))
            add(r, i0(i, jb), f * (-idz - 0.5 * A00))
            add(r, ih(i, ja), (1 - f) * (-0.5 * A0h * ph[i, ja]))
            add(r, ih(i, jb), f * (-0.5 * A0h * ph[i, jb]))
            # Dh cell (row ih(i,j)): (S(Dh[i+1])-Dh[i])/dz = 0.5 Ahh (S(Dh[i+1])+Dh[i])
            #                        + 0.5 Ah0 (conj(ph[i])D0[i] + S(conj(ph[i+1])D0[i+1]))
            r = ih(i, j)
            add(r, ih(i, j), -idz - 0.5 * Ahh)
            add(r, i0(i, j), -0.5 * Ah0 * conj(ph[i, j]))
            add(r, ih(i + 1, ja), (1 - f) * (idz - 0.5 * Ahh))
            add(r, ih(i + 1, jb), f * (idz - 0.5 * Ahh))
            add(r, i0(i + 1, ja), (1 - f) * (-0.5 * Ah0 * conj(ph[i + 1, ja])))
            add(r, i0(i + 1, jb), f * (-0.5 * Ah0 * conj(ph[i + 1, jb])))

    if backend == "sparse":
        Amat = _sp.csc_matrix((vals, (rows, cols)), shape=(2 * N0, 2 * N0))
        u = _spla.spsolve(Amat, bvec)
        u = torch.as_tensor(u, dtype=dtype)
    else:
        u = torch.linalg.solve(A, b)
    return u[:N0].reshape(nzp, nx), u[N0:].reshape(nzp, nx)


def bragg_2d_nz_for_unit_shift(thickness_um: float, theta_B_deg: float, dx_um: float) -> int:
    """Depth-step count giving a 1-cell lateral shift per step (``a=1``) for :func:`solve_tt_2d`
    -- no interpolation diffusion, exact energy conservation."""
    return max(1, int(round(thickness_um / math.tan(math.radians(theta_B_deg)) / dx_um)))
