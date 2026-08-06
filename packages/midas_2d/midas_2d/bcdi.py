"""BCDI geometry: mapping the FFT of a nanocrystal onto detector pixels.

:mod:`midas_2d.coherent` gives the coherent forward ``I = |FFT(psi)|^2`` and a
differentiable phase retrieval. What it does *not* say is where in reciprocal
space each array element sits -- and in Bragg CDI that is the part which bites,
because the sampling grid is not orthogonal.

The measured array is indexed by ``(detector column, detector row, rocking
step)``. Those three directions span q as

    dq1 = k (p/D) e1        e1, e2 span the detector plane (perpendicular k_f)
    dq2 = k (p/D) e2
    dq3 = -dtheta (omega_hat x G)          the rocking arc

which is a *sheared* parallelepiped, not a box: dq1 and dq2 lie in the tangent
plane of the Ewald sphere while dq3 is tangent to a circle about the rotation
axis. For a typical geometry the shear is of order the Bragg angle (17 degrees
for Au(111) at 9 keV), so it is not a small correction.

A plain FFT therefore does not reconstruct onto a Cartesian real-space grid.
Requiring ``q.r = 2 pi sum_k n_k m_k / N_k`` pins the conjugate basis:

    B^T C = 2 pi diag(1/N_k)      =>      C = 2 pi B^-T diag(1/N_k)

so the object lives on the columns of ``C`` -- generally non-orthogonal, with
unequal edge lengths. The workflow is:

    1. :func:`q_basis`             -- B from the beamline geometry
    2. :func:`conjugate_real_basis` -- C, the grid to build/reconstruct the object on
    3. phase retrieval in index space with plain FFTs (:mod:`midas_2d.coherent`)
    4. :func:`sheared_to_lab`      -- interpolate the RECONSTRUCTED OBJECT onto a
                                      Cartesian lab grid, as the last step

Never interpolate the measured intensity onto an orthogonal q grid before
phasing: it smears the fringes and breaks the oversampling bookkeeping that
:func:`oversampling` accounts for. Correct the shear on the object, at the end.

Conventions
-----------
Incident beam along ``+z`` and the scattering plane is ``x-z``, matching
:func:`midas_2d.instrument.project_to_detector`. The default rocking axis is
``y`` (perpendicular to the scattering plane). Reciprocal lengths are 1/Angstrom,
real-space lengths Angstrom, detector lengths mm, angles degrees on input.

The transform pairs with ``A(q) = sum psi(r) exp(-i q.r)`` -- i.e. plain
``torch.fft.fftn`` -- so a displacement field enters as ``psi = s exp(-i G.u)``.
Flip one sign and you swap tension for compression: conjugating psi maps
``I(q) -> I(-q)``, so the conjugate-twin ambiguity of phase retrieval *is* the
strain-sign ambiguity.

How badly that bites depends on the object, and the condition is sharp: the two
signs give a *bit-identical* pattern only when ``psi`` is centrosymmetric, i.e.
both ``|psi|`` and the phase are even. Measured on a 60 A Au sphere at
``|G.u| ~ 3.6 rad``, correlation between the two sign choices:

    spherical support, even phase (x^2)   1.00000   sign unrecoverable, ever
    spherical support, odd phase  (x^3)   0.465     recoverable
    faceted support,   even phase         0.755     recoverable
    faceted support,   odd phase          0.730     recoverable

Real grains are faceted and inhomogeneously strained, so the sign *is* in the
data -- but nothing in the optimiser knows which twin it landed on. Fix it with
known facets or a second reflection.

Note when comparing a pattern with its own inversion: on an fftshift-ed array of
even length, ``q -> -q`` is ``flip`` followed by ``roll(+1)``, not ``flip``
alone. Plain ``flip`` is off by one voxel, which reads as a real discrepancy
(0.62 instead of 1.00 on the test above).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = [
    "bragg_geometry",
    "q_basis",
    "conjugate_real_basis",
    "oversampling",
    "shear_angles_deg",
    "detector_distance_for_oversampling",
    "rocking_step_for_oversampling",
    "sheared_to_lab",
    "q_grid",
    "rotation_to_bragg",
    "object_to_amplitude",
    "detector_signal",
    "sample_counts",
    "speckle_from_atoms",
    "atoms_to_object",
    "atom_sum_cost",
]

_TWOPI = 2.0 * math.pi


def _real_dtype(device, prefer=None):
    """A real float dtype the device actually supports.

    Geometry here wants float64 -- the basis inversion and the shear resampling
    both lose accuracy in single -- but Apple's MPS backend has no float64 at
    all, so fall back rather than raise.
    """
    import torch

    prefer = torch.float64 if prefer is None else prefer
    if torch.device(device).type == "mps" and prefer == torch.float64:
        return torch.float32
    return prefer


# --------------------------------------------------------------- lab vectors
def bragg_geometry(wavelength_A, d_spacing_A, *, dtype=None):
    """Lab-frame vectors at the Bragg condition for one reflection.

    Parameters
    ----------
    wavelength_A, d_spacing_A : float
        Wavelength and the d-spacing of the reflection (Angstrom). Get the
        latter from :mod:`midas_hkls` for a real crystal.
    dtype : torch dtype, optional
        Defaults to ``torch.float64`` -- the basis inversion in
        :func:`conjugate_real_basis` is worth doing in double.

    Returns
    -------
    dict
        ``ki``, ``kf0``, ``G`` (all 1/Angstrom), the unit vectors ``khat``,
        ``e1``, ``e2`` spanning the detector plane, and ``theta_rad``.

    Raises
    ------
    ValueError
        If the reflection is inaccessible at this wavelength (lambda > 2d).
    """
    import torch

    dtype = torch.float64 if dtype is None else dtype
    sin_th = float(wavelength_A) / (2.0 * float(d_spacing_A))
    if not -1.0 <= sin_th <= 1.0:
        raise ValueError(
            f"reflection inaccessible: lambda/(2d) = {sin_th:.4f} > 1 "
            f"(lambda = {wavelength_A} A, d = {d_spacing_A} A)")
    theta = math.asin(sin_th)
    tth = 2.0 * theta
    k = _TWOPI / float(wavelength_A)

    ki = torch.tensor([0.0, 0.0, k], dtype=dtype)
    kf0 = torch.tensor([k * math.sin(tth), 0.0, k * math.cos(tth)], dtype=dtype)
    G = kf0 - ki
    khat = kf0 / k
    # Detector plane perpendicular to k_f0; e1 lies in the scattering plane.
    up = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
    e1 = torch.linalg.cross(up, khat)
    e1 = e1 / torch.linalg.norm(e1)
    e2 = torch.linalg.cross(khat, e1)
    return {"ki": ki, "kf0": kf0, "G": G, "khat": khat, "e1": e1, "e2": e2,
            "theta_rad": theta, "k": k}


# ------------------------------------------------------------------ the basis
def q_basis(wavelength_A, d_spacing_A, *, distance_mm, pixel_mm,
            rocking_step_deg, rocking_axis: Sequence[float] = (0.0, 1.0, 0.0),
            dtype=None):
    """(3, 3) matrix ``B``; columns are dq per unit step of the measured array.

    Column order is ``(detector column, detector row, rocking step)``, units
    1/Angstrom.

    The detector columns are linearised in the pixel angle -- i.e. the Ewald
    sphere is treated as locally flat. The error grows quadratically towards the
    array corner; check it is a small fraction of a pixel for your array size
    before trusting it (``||k_i + G + q| - k|`` divided by ``|dq1|``).

    Notes
    -----
    The rocking column follows from ``q_sample(theta) = R(-theta) Q_lab - G``,
    so ``dq/dtheta|_0 = -omega_hat x G``: the sample rotates by ``+theta`` about
    ``omega_hat``. Reverse the sign of ``rocking_step_deg`` for the opposite
    rotation sense.
    """
    import torch

    dtype = torch.float64 if dtype is None else dtype
    v = bragg_geometry(wavelength_A, d_spacing_A, dtype=dtype)

    ang_per_pixel = float(pixel_mm) / float(distance_mm)
    dq1 = v["k"] * ang_per_pixel * v["e1"]
    dq2 = v["k"] * ang_per_pixel * v["e2"]

    omega = torch.as_tensor(rocking_axis, dtype=dtype)
    omega = omega / torch.linalg.norm(omega)
    dq3 = -math.radians(float(rocking_step_deg)) * torch.linalg.cross(omega, v["G"])

    return torch.stack([dq1, dq2, dq3], dim=1)


def conjugate_real_basis(q_basis_matrix, shape: Sequence[int]):
    """(3, 3) matrix ``C = 2 pi B^-T diag(1/N)``: the real-space grid the FFT uses.

    Columns are the real-space voxel vectors in Angstrom, in the same axis order
    as ``shape``. They are generally neither orthogonal nor of equal length --
    which is exactly why a raw BCDI reconstruction looks sheared, and what
    :func:`sheared_to_lab` undoes.
    """
    import torch

    B = torch.as_tensor(q_basis_matrix)
    if B.shape[-2:] != (3, 3):
        raise ValueError(f"q_basis must be (3, 3), got {tuple(B.shape)}")
    if len(shape) != 3:
        raise ValueError(f"shape must have 3 entries, got {tuple(shape)}")
    N = torch.as_tensor(list(shape), dtype=B.dtype, device=B.device)
    return _TWOPI * torch.linalg.inv(B).transpose(-1, -2) @ torch.diag(1.0 / N)


# ---------------------------------------------------------------- sampling
def oversampling(q_basis_matrix, object_size_A: float):
    """Fringe spacing over q-step along each array axis: ``(2 pi / a) / |dq_i|``.

    Must exceed ~2 in *every* dimension for the intensity to determine the
    object: ``|A|^2`` has an autocorrelation support twice the object support,
    so Shannon applies at half the object's own Nyquist rate. In practice 3-5 is
    the usual working range.
    """
    import torch

    B = torch.as_tensor(q_basis_matrix)
    return (_TWOPI / float(object_size_A)) / torch.linalg.norm(B, dim=0)


def shear_angles_deg(q_basis_matrix):
    """Angles between the q-basis vectors, ordered ``(1-2, 1-3, 2-3)``.

    All 90 would mean the sampling grid is orthogonal and a plain FFT would
    reconstruct onto a Cartesian grid. It never is.
    """
    import torch

    B = torch.as_tensor(q_basis_matrix)
    u = B / torch.linalg.norm(B, dim=0, keepdim=True)
    cos = torch.stack([u[:, 0] @ u[:, 1], u[:, 0] @ u[:, 2], u[:, 1] @ u[:, 2]])
    return torch.rad2deg(torch.acos(torch.clamp(cos, -1.0, 1.0)))


def detector_distance_for_oversampling(wavelength_A, object_size_A, pixel_mm,
                                       target: float = 4.0) -> float:
    """Detector distance (mm) giving ``target`` oversampling on the detector.

    ``sigma = lambda D / (a p)``, so ``D = sigma a p / lambda``. This is how you
    pick D at the beamline: it is set by the grain size you expect, not by the
    reflection.
    """
    p_A = float(pixel_mm) * 1e7                       # mm -> Angstrom
    return target * float(object_size_A) * p_A / float(wavelength_A) / 1e7


def rocking_step_for_oversampling(wavelength_A, d_spacing_A, object_size_A,
                                  target: float = 4.0,
                                  rocking_axis: Sequence[float] = (0.0, 1.0, 0.0)
                                  ) -> float:
    """Rocking step (deg) giving ``target`` oversampling along the arc.

    For a rocking axis perpendicular to G this reduces to ``d / (sigma a)``.
    """
    import torch

    v = bragg_geometry(wavelength_A, d_spacing_A)
    omega = torch.as_tensor(rocking_axis, dtype=v["G"].dtype)
    omega = omega / torch.linalg.norm(omega)
    G_mag = float(torch.linalg.norm(v["G"]))
    arc = float(torch.linalg.norm(torch.linalg.cross(omega, v["G"])))  # per radian
    # Relative test: |omega x G| = |G| sin(angle), so this is sin(angle) <= 1e-9.
    # An exactly-parallel axis lands at ~1e-17 rather than 0 in floating point.
    if arc <= 1e-9 * G_mag:
        raise ValueError(
            "rocking axis is parallel to G: rocking sweeps no q "
            f"(|omega x G| / |G| = {arc / G_mag:.2e})")
    return math.degrees((_TWOPI / float(object_size_A)) / target / arc)


# ------------------------------------------------------------ shear correction
def sheared_to_lab(obj, real_basis, *, voxel_A: float | None = None,
                   shape: Sequence[int] | None = None, pad: float = 1.0):
    """Interpolate an object off the sheared FFT grid onto a Cartesian lab grid.

    This is the LAST step of a BCDI reconstruction. The object comes out of
    phase retrieval sampled on the columns of ``C`` (see
    :func:`conjugate_real_basis`), which are non-orthogonal and unequal in
    length; only after this resampling do distances and shapes mean what they
    look like.

    Amplitude and phase are interpolated separately via the real and imaginary
    parts, which is correct as long as the phase is smooth on the voxel scale --
    the usual BCDI assumption, and the same one that makes the displacement
    field recoverable at all.

    Parameters
    ----------
    obj : tensor, shape (N1, N2, N3)
        Real or complex object on the sheared grid, index-centred (element
        ``[N1//2, N2//2, N3//2]`` sits at r = 0).
    real_basis : (3, 3) tensor
        ``C`` from :func:`conjugate_real_basis`; columns match ``obj``'s axes.
    voxel_A : float, optional
        Cubic output voxel edge. Defaults to the shortest input voxel vector,
        so nothing is lost.
    shape : sequence of 3 ints, optional
        Output shape. Defaults to a box enclosing the input parallelepiped.
    pad : float
        Scale on the default output box.

    Returns
    -------
    dict
        ``obj`` (resampled, same dtype as input), ``voxel_A``, and ``extent_A``
        (the half-width of the output box along x, y, z).
    """
    import torch
    import torch.nn.functional as F

    obj = torch.as_tensor(obj)
    if obj.dim() != 3:
        raise ValueError(f"obj must be 3-D, got {obj.dim()}-D")
    wdt = _real_dtype(obj.device)          # MPS has no float64
    C = torch.as_tensor(real_basis, dtype=wdt, device=obj.device)
    N = obj.shape

    if voxel_A is None:
        voxel_A = float(torch.linalg.norm(C, dim=0).min())
    if shape is None:
        # Corners of the input parallelepiped, in lab coordinates.
        idx = torch.tensor([[i, j, k] for i in (-N[0] / 2, N[0] / 2)
                            for j in (-N[1] / 2, N[1] / 2)
                            for k in (-N[2] / 2, N[2] / 2)], dtype=wdt, device=obj.device)
        half = (idx @ C.transpose(0, 1)).abs().max(dim=0).values * float(pad)
        shape = [max(2, int(2 * math.ceil(float(h) / voxel_A))) for h in half]

    # Lab-frame coordinates of every output voxel.
    axes = [(torch.arange(n, dtype=wdt, device=obj.device) - n // 2) * voxel_A
            for n in shape]
    r = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)      # (...,3)

    # Lab -> fractional index on the sheared basis, then -> array index.
    m = r @ torch.linalg.inv(C).transpose(0, 1)                        # centred
    idx = m + torch.tensor([n // 2 for n in N], dtype=wdt, device=obj.device)

    # grid_sample wants coords in [-1, 1] with align_corners=True, and its last
    # grid axis is x <-> the LAST array axis, so the triple is reversed.
    denom = torch.tensor([max(n - 1, 1) for n in N], dtype=wdt,
                         device=obj.device)
    g = (2.0 * idx / denom - 1.0).flip(-1).unsqueeze(0)                # (1,D,H,W,3)

    def _sample(vol):
        v = vol.to(wdt)[None, None]                          # (1,1,N1,N2,N3)
        return F.grid_sample(v, g, mode="bilinear", padding_mode="zeros",
                             align_corners=True)[0, 0]

    if obj.is_complex():
        out = torch.complex(_sample(obj.real), _sample(obj.imag)).to(obj.dtype)
    else:
        out = _sample(obj).to(obj.dtype)

    return {"obj": out, "voxel_A": voxel_A,
            "extent_A": [float(n // 2 * voxel_A) for n in shape]}


# ================================================================ forward chain
#
# Everything below is differentiable end to end (the one exception is
# :func:`sample_counts`, which draws random numbers and says so). Gradients flow
# back to the object, to atomic coordinates, and through midas_hkls to the
# lattice and structure factor.

def q_grid(q_basis_matrix, shape: Sequence[int], *, offset=None,
           centered: bool = True):
    """(N1, N2, N3, 3) q-vectors for every element of a measured array.

    Parameters
    ----------
    q_basis_matrix : (3, 3) tensor
        ``B`` from :func:`q_basis`; columns are the per-step q increments.
    shape : sequence of 3 ints
    offset : (3,) tensor, optional
        Added to every vector. Pass ``G`` to get absolute ``Q`` rather than the
        deviation ``q`` -- which is what the polarisation/solid-angle correction
        and :func:`speckle_from_atoms` both need.
    centered : bool
        If True (default) index ``N//2`` sits at q = 0, matching an fftshift-ed
        array. Set False for raw FFT order.
    """
    import torch

    B = torch.as_tensor(q_basis_matrix)
    axes = []
    for n in shape:
        a = torch.arange(n, dtype=B.dtype, device=B.device)
        axes.append(a - n // 2 if centered else a)
    m = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    q = m @ B.transpose(-1, -2)
    if offset is not None:
        q = q + torch.as_tensor(offset, dtype=B.dtype, device=B.device)
    return q


def rotation_to_bragg(g_crystal, wavelength_A, d_spacing_A, *, dtype=None):
    """(3, 3) rotation ``R`` (lab <- crystal) putting a reflection at the Bragg
    condition, so atomic coordinates share the frame of the q-grid.

    Atom coordinates from MD or a builder live in the crystal frame, where the
    reflection points along ``g_crystal`` -- e.g. ``(2 pi / a) (1, 1, 1)`` for a
    cubic (111). :func:`bragg_geometry` instead puts ``G`` in the lab x-z
    scattering plane. Feed unrotated coordinates to
    :func:`speckle_from_atoms` and the Bragg peak lands somewhere off the array
    entirely; the intensity you get is real, but it is not the reflection you
    asked for.

    Apply as ``coords_lab = coords_crystal @ R.T``.

    This is the minimal (Rodrigues) rotation carrying ``g_crystal`` onto ``G``.
    The residual freedom -- a rotation about ``G`` itself -- is the azimuth,
    which is a real experimental choice: any value satisfies Bragg but they
    give different patterns. Compose your own rotation about ``G`` to set it.
    """
    import torch

    dtype = torch.float64 if dtype is None else dtype
    g = torch.as_tensor(g_crystal, dtype=dtype)
    if g.shape != (3,):
        raise ValueError(f"g_crystal must be a 3-vector, got {tuple(g.shape)}")
    gn = torch.linalg.norm(g)
    if gn <= 0:
        raise ValueError("g_crystal has zero length")
    a = g / gn
    G = bragg_geometry(wavelength_A, d_spacing_A, dtype=dtype)["G"]
    b = G / torch.linalg.norm(G)

    v = torch.linalg.cross(a, b)
    c = torch.dot(a, b)
    s = torch.linalg.norm(v)
    eye = torch.eye(3, dtype=dtype)
    if float(s) < 1e-12:                       # parallel or antiparallel
        if float(c) > 0:
            return eye
        # 180 degrees about any axis perpendicular to a
        perp = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
        if abs(float(torch.dot(perp, a))) > 0.9:
            perp = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
        axis = torch.linalg.cross(a, perp)
        axis = axis / torch.linalg.norm(axis)
        K = _skew(axis)
        return eye + 2.0 * (K @ K)
    K = _skew(v)
    return eye + K + K @ K * ((1.0 - c) / (s * s))


def _skew(v):
    import torch

    z = torch.zeros((), dtype=v.dtype, device=v.device)
    return torch.stack([
        torch.stack([z, -v[2], v[1]]),
        torch.stack([v[2], z, -v[0]]),
        torch.stack([-v[1], v[0], z]),
    ])


def object_to_amplitude(psi, *, centered: bool = True):
    """Far-field complex amplitude of a real-space object: ``A(q) = FFT(psi)``.

    Pairs with ``A(q) = sum psi(r) exp(-i q.r)`` -- plain :func:`torch.fft.fftn`
    -- so a displacement field enters as ``psi = s exp(-i G.u)``. Use the same
    pairing or the reconstructed strain comes out sign-flipped.

    ``centered=True`` means ``psi`` is index-centred (r = 0 at ``N//2``) and the
    returned amplitude is fftshift-ed, so q = 0 also sits at ``N//2``.
    """
    import torch

    psi = torch.as_tensor(psi)
    if not psi.is_complex():
        psi = torch.complex(psi, torch.zeros_like(psi))
    A = torch.fft.fftn(torch.fft.ifftshift(psi) if centered else psi)
    return torch.fft.fftshift(A) if centered else A


def detector_signal(intensity, *, Q=None, wavelength_A=None,
                    structure_factor_sq=None, polarization: float = 0.5,
                    coherence_length_A=None, real_basis=None,
                    photons_per_peak=None, background=0.0):
    """Turn ``|A|^2`` into the expected photon rate on the detector.

    Fully differentiable -- this is the *expectation*, not a noisy draw. For a
    likelihood, pair it with :func:`midas_2d.instrument.poisson_nll` against
    fixed counts; do not backpropagate through :func:`sample_counts`.

    Applied in order, each step optional:

    ``structure_factor_sq``
        ``|F_hkl|^2`` -- a scalar for one reflection. Get it from
        :func:`midas_hkls.structure_factors`; a tensor keeps the gradient to the
        lattice and the atomic positions in the unit cell.
    ``Q`` + ``wavelength_A``
        Polarisation and solid-angle obliquity via
        :func:`midas_2d.instrument.solid_angle_polarization`. Nearly constant
        across a BCDI array (``|q| << |G|``), but it is the honest place for it.
    ``coherence_length_A`` + ``real_basis``
        Partial coherence. In the Schell model a finite coherence length
        multiplies the object AUTOCORRELATION by the coherence factor, which is
        a convolution of the intensity. Doing it in the autocorrelation domain
        is the physically correct formulation rather than an ad-hoc blur.
        Omitting this is the single most common way a simulated BCDI pattern
        looks sharper than any real one.
    ``photons_per_peak``
        Rescale so the brightest voxel has this expected count.
    ``background``
        Flat additive rate, added last.

    Returns
    -------
    real tensor, same shape as ``intensity``.
    """
    import torch

    I = torch.as_tensor(intensity)
    if I.is_complex():
        raise TypeError("detector_signal expects intensity, not amplitude; "
                        "pass |A|^2 (or use object_to_amplitude then abs()**2)")

    if structure_factor_sq is not None:
        I = I * torch.as_tensor(structure_factor_sq, dtype=I.dtype, device=I.device)

    if Q is not None:
        if wavelength_A is None:
            raise ValueError("Q given without wavelength_A")
        from .instrument import solid_angle_polarization
        lp = solid_angle_polarization(Q, wavelength_A=wavelength_A,
                                      polarization=polarization)
        I = I * lp.to(I.dtype)

    if coherence_length_A:
        if real_basis is None:
            raise ValueError("coherence_length_A requires real_basis "
                             "(the grid the autocorrelation lives on)")
        C = torch.as_tensor(real_basis, dtype=I.dtype, device=I.device)
        r = _real_grid(C, I.shape)
        gamma = torch.exp(-(r.pow(2).sum(dim=-1))
                          / (2.0 * float(coherence_length_A) ** 2)).to(I.dtype)
        ac = torch.fft.ifftn(torch.fft.ifftshift(I))
        ac = ac * torch.fft.ifftshift(gamma).to(ac.dtype)
        I = torch.fft.fftshift(torch.fft.fftn(ac)).real
        I = torch.clamp(I, min=0.0)          # kill O(1e-16) round-off negatives

    if photons_per_peak is not None:
        I = I * (float(photons_per_peak) / torch.clamp(I.max(), min=1e-30))

    if background:
        I = I + float(background)
    return I


def _real_grid(C, shape):
    """(N1, N2, N3, 3) real-space positions on the columns of C, index-centred."""
    import torch

    axes = [torch.arange(n, dtype=C.dtype, device=C.device) - n // 2 for n in shape]
    m = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    return m @ C.transpose(-1, -2)


def sample_counts(rate, *, generator=None):
    """Poisson draw from an expected rate. **Not differentiable** -- it samples.

    Use it to make synthetic data, then fit the *rate* from
    :func:`detector_signal` to those fixed counts with a Poisson likelihood.
    """
    import torch

    rate = torch.as_tensor(rate)
    if not torch.isfinite(rate).all():
        raise ValueError("non-finite rate passed to sample_counts")
    return torch.poisson(torch.clamp(rate, min=0.0).detach(), generator=generator)


def atoms_to_object(coords, real_basis, shape, *, reference=None, G=None,
                    fill: float | None = None):
    """Bin atomic coordinates onto the BCDI voxel grid -> the complex object psi.

    **The scalable MD route.** :func:`speckle_from_atoms` sums over every
    (atom, q-point) pair, which is ``O(N_atoms * N_q)`` and therefore capped at
    nanometre-scale crystals -- a 300 nm grain holds ~1e9 atoms and the direct
    sum is hopeless. This instead bins the atoms once, ``O(N_atoms)``, producing
    the envelope object

        psi(r) = occupancy(r) * exp(-i G . u(r))

    which then goes through :func:`object_to_amplitude` at real grain sizes.
    That is how MD couples to BCDI in practice: MD supplies the displacement
    field, the envelope model supplies the diffraction.

    Parameters
    ----------
    coords : (M, 3) tensor
        Deformed atomic positions, Angstrom, in the lab frame (see
        :func:`rotation_to_bragg`).
    real_basis : (3, 3) tensor
        ``C`` from :func:`conjugate_real_basis`; the voxel grid to bin onto.
    shape : sequence of 3 ints
    reference : (M, 3) tensor, optional
        Undeformed positions. ``u = coords - reference``, averaged per voxel.
        Without it the object is real (shape only, no phase).
    G : (3,) tensor, optional
        Bragg vector; required with ``reference``. Sets the phase ``-G.u``.
    fill : float, optional
        Atoms per voxel corresponding to occupancy 1. Defaults to the 95th
        percentile of the occupied-voxel counts, which is robust to the partly
        filled voxels at the surface (the mean would bias the whole object
        low, the max would be set by a single lucky voxel).

    Returns
    -------
    dict
        ``psi``, ``occupancy``, ``u`` (mean displacement per voxel), and
        ``n_outside`` -- atoms that fell off the grid, which is a real warning
        that the array is too small or the object is off-centre.

    Notes
    -----
    Differentiable with respect to ``coords`` **through the displacement**, which
    is what a refinement needs. The bin assignment itself is a discrete
    ``round`` and carries no gradient, so the occupancy is piecewise constant in
    the coordinates.
    """
    import torch

    coords = torch.as_tensor(coords)
    C = torch.as_tensor(real_basis, dtype=coords.dtype, device=coords.device)
    if len(shape) != 3:
        raise ValueError(f"shape must have 3 entries, got {tuple(shape)}")
    N1, N2, N3 = (int(n) for n in shape)

    # lab -> fractional index -> nearest voxel
    frac = coords @ torch.linalg.inv(C).transpose(0, 1)
    idx = torch.round(frac).long() + torch.tensor([N1 // 2, N2 // 2, N3 // 2],
                                                  device=coords.device)
    inside = ((idx >= 0) & (idx < torch.tensor([N1, N2, N3], device=coords.device))).all(-1)
    n_outside = int((~inside).sum())
    idx = idx[inside]
    flat = (idx[:, 0] * N2 + idx[:, 1]) * N3 + idx[:, 2]

    n_vox = N1 * N2 * N3
    counts = torch.zeros(n_vox, dtype=coords.dtype, device=coords.device)
    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=coords.dtype))

    u_mean = torch.zeros(n_vox, 3, dtype=coords.dtype, device=coords.device)
    if reference is not None:
        if G is None:
            raise ValueError("reference given without G: no phase can be formed")
        ref = torch.as_tensor(reference, dtype=coords.dtype, device=coords.device)
        if ref.shape != coords.shape:
            raise ValueError(f"reference {tuple(ref.shape)} != coords {tuple(coords.shape)}")
        u = (coords - ref)[inside]
        usum = torch.zeros(n_vox, 3, dtype=coords.dtype, device=coords.device)
        usum.index_add_(0, flat, u)                     # differentiable in u
        u_mean = usum / torch.clamp(counts, min=1.0).unsqueeze(-1)

    occupied = counts[counts > 0]
    if fill is None:
        fill = float(torch.quantile(occupied, 0.95)) if occupied.numel() else 1.0
    occupancy = torch.clamp(counts / max(fill, 1e-30), max=1.0).reshape(N1, N2, N3)

    if reference is not None:
        Gt = torch.as_tensor(G, dtype=coords.dtype, device=coords.device)
        phase = -(u_mean @ Gt).reshape(N1, N2, N3)
        psi = torch.polar(occupancy, phase * (occupancy > 0))
    else:
        psi = torch.complex(occupancy, torch.zeros_like(occupancy))

    return {"psi": psi, "occupancy": occupancy,
            "u": u_mean.reshape(N1, N2, N3, 3), "n_outside": n_outside}


def atom_sum_cost(n_q: int, n_atoms: int) -> dict:
    """Rough cost of a direct :func:`speckle_from_atoms` evaluation.

    Returns ``terms`` (the atom x q-point product) and an order-of-magnitude
    ``seconds`` on a couple of CPU threads, measured at ~1e8 terms/s. Use it to
    decide between the direct sum and :func:`atoms_to_object`: past ~1e9 terms
    the direct sum stops being interactive, and past ~1e11 it is hopeless.
    """
    terms = int(n_q) * int(n_atoms)
    return {"terms": terms, "seconds": terms / 1.0e8,
            "advice": ("direct sum is fine" if terms < 1e9 else
                       "slow -- prefer atoms_to_object" if terms < 1e11 else
                       "infeasible -- use atoms_to_object")}


def speckle_from_atoms(coords, elements, Q, *, max_elements: int = 1 << 24,
                       amplitude: bool = False):
    """Coherent signal straight from atomic coordinates -- the MD path.

    ``A(Q) = sum_i f_i(|Q|) exp(i Q . r_i)``, evaluated on the BCDI ``Q`` grid
    (build it with ``q_grid(B, shape, offset=G)``). This makes *no* small-strain
    or envelope approximation: it is the reference the
    ``psi = s exp(-i G.u)`` model is an approximation to, so the two agreeing is
    a real check on the envelope linearisation.

    .. warning::
       Cost is ``O(N_atoms * N_q)`` with no way around it. A 10 nm crystal on a
       32^3 array is ~1e9 terms (seconds); a real 300 nm BCDI grain is ~1e9
       atoms and 1e14 terms, which is hopeless. Check with :func:`atom_sum_cost`
       and switch to :func:`atoms_to_object` -- which bins the same coordinates
       in ``O(N_atoms)`` -- for anything above a few tens of nanometres. Use
       this function to *validate* the envelope model on a small crystal, then
       use the envelope model at the real size.

    Differentiable with respect to ``coords`` -- so a detector-level loss
    backpropagates to every atom.

    Convention
    ----------
    :func:`midas_2d.debye.coherent_amplitude` uses ``exp(+i Q.r)`` while
    :func:`torch.fft.fftn` uses ``exp(-i q.r)``. These are *consistently paired*,
    not in conflict: expanding the ``+`` convention about a Bragg peak gives the
    envelope ``s exp(+i G.u)``, whose conjugate is the ``s exp(-i G.u)`` used
    with ``fftn``. Conjugation does not change the modulus, so

        |speckle_from_atoms(..., G + q)|^2  ==  |object_to_amplitude(psi)(q)|^2

    at the *same* q -- no inversion. Returned ``amplitude=True`` values are the
    conjugate of the FFT-convention amplitude; take ``.conj()`` to mix them.

    Parameters
    ----------
    coords : (M, 3) tensor, Angstrom
    elements : sequence[str], length M
    Q : (..., 3) tensor, 1/Angstrom -- absolute Q, not the deviation q
    max_elements : int
        Chunking budget. The kernel materialises a ``(chunk, M)`` phase array,
        so the chunk is set to ``max_elements // M``. Lower it if memory is
        tight; it does not change the result.
    amplitude : bool
        Return the complex amplitude instead of ``|A|^2``.

    Notes
    -----
    Atoms are grouped by element before summing. The form factor depends only on
    ``|Q|`` and the species, so calling
    :func:`midas_2d.debye.coherent_amplitude` directly would evaluate the same
    Cromer-Mann curve once per *atom* and hold an ``(n_q, M)`` array that is one
    column repeated M times -- measured at ~69% of the runtime and 128 MB for a
    4000-atom single-element case. Grouping computes it once per species. The
    result is identical; ``coherent_amplitude`` remains the reference and the
    two are checked against each other in the tests.
    """
    import torch

    from .debye import atomic_form_factors

    coords = torch.as_tensor(coords)
    Q = torch.as_tensor(Q, dtype=coords.dtype, device=coords.device)
    M = coords.shape[0]
    if M == 0:
        raise ValueError("coords is empty")
    if len(elements) != M:
        raise ValueError(f"elements has {len(elements)} entries for {M} atoms")

    groups: dict[str, list[int]] = {}
    for i, e in enumerate(elements):
        groups.setdefault(e, []).append(i)
    grouped = [(e, torch.as_tensor(idx, dtype=torch.long, device=coords.device))
               for e, idx in groups.items()]

    shape = Q.shape[:-1]
    flat = Q.reshape(-1, 3)
    chunk = max(1, int(max_elements) // max(M, 1))

    out = []
    for start in range(0, flat.shape[0], chunk):
        qc = flat[start:start + chunk]                       # (c, 3)
        qmag = torch.linalg.vector_norm(qc, dim=-1)          # (c,)
        re = torch.zeros_like(qmag)
        im = torch.zeros_like(qmag)
        for element, idx in grouped:
            f = atomic_form_factors([element], qmag)[..., 0]  # (c,) once per species
            ph = qc @ coords.index_select(0, idx).T           # (c, n_e)
            re = re + f * torch.cos(ph).sum(dim=-1)
            im = im + f * torch.sin(ph).sum(dim=-1)
        A = torch.complex(re, im)
        out.append(A if amplitude else (re * re + im * im))
    return torch.cat(out, dim=0).reshape(shape)
