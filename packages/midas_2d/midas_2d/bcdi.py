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
Flip one sign and you swap tension for compression, invisibly: conjugating psi
maps ``I(q) -> I(-q)``, so the conjugate-twin ambiguity of phase retrieval *is*
the strain-sign ambiguity.
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
]

_TWOPI = 2.0 * math.pi


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
    C = torch.as_tensor(real_basis, dtype=torch.float64, device=obj.device)
    N = obj.shape

    if voxel_A is None:
        voxel_A = float(torch.linalg.norm(C, dim=0).min())
    if shape is None:
        # Corners of the input parallelepiped, in lab coordinates.
        idx = torch.tensor([[i, j, k] for i in (-N[0] / 2, N[0] / 2)
                            for j in (-N[1] / 2, N[1] / 2)
                            for k in (-N[2] / 2, N[2] / 2)], dtype=torch.float64)
        half = (idx @ C.transpose(0, 1)).abs().max(dim=0).values * float(pad)
        shape = [max(2, int(2 * math.ceil(float(h) / voxel_A))) for h in half]

    # Lab-frame coordinates of every output voxel.
    axes = [(torch.arange(n, dtype=torch.float64, device=obj.device) - n // 2) * voxel_A
            for n in shape]
    r = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)      # (...,3)

    # Lab -> fractional index on the sheared basis, then -> array index.
    m = r @ torch.linalg.inv(C).transpose(0, 1)                        # centred
    idx = m + torch.tensor([n // 2 for n in N], dtype=torch.float64, device=obj.device)

    # grid_sample wants coords in [-1, 1] with align_corners=True, and its last
    # grid axis is x <-> the LAST array axis, so the triple is reversed.
    denom = torch.tensor([max(n - 1, 1) for n in N], dtype=torch.float64,
                         device=obj.device)
    g = (2.0 * idx / denom - 1.0).flip(-1).unsqueeze(0)                # (1,D,H,W,3)

    def _sample(vol):
        v = vol.to(torch.float64)[None, None]                          # (1,1,N1,N2,N3)
        return F.grid_sample(v, g, mode="bilinear", padding_mode="zeros",
                             align_corners=True)[0, 0]

    if obj.is_complex():
        out = torch.complex(_sample(obj.real), _sample(obj.imag)).to(obj.dtype)
    else:
        out = _sample(obj).to(obj.dtype)

    return {"obj": out, "voxel_A": voxel_A,
            "extent_A": [float(n // 2 * voxel_A) for n in shape]}
