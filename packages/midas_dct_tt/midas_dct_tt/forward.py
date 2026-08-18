"""The forward model: TT topographs and DCT frames.

Phase 1 of ``implementation_plan.md`` -- the deliverable.

A topograph is the line integral, along the diffracted beam, of

    chi(r) * |F|^2 * Lp * A(Q(r))

where ``chi`` is grain occupancy, ``A`` the reciprocal-space acceptance
(:mod:`midas_dct_tt.acceptance`, i.e. DFXM's with ``na = 0``), and
``Q(r) = F(r)^-T G0`` the local scattering vector. Two distinct things happen
when the grain is deformed, and both are kept exactly:

1. **Intensity**: a voxel whose local ``Q`` has moved out of the acceptance
   contributes less, or nothing.
2. **Position**: that voxel also diffracts in a *different direction*
   (``k_h = k_in + Q``), so it lands elsewhere on the detector.

(2) is what makes naive back-projection place material at the wrong 3-D
position -- the "pseudo distorted grain volume". Carrying it here, rather than
linearising it, is the point of the package.

Scope of the kinematical approximation
--------------------------------------
This is a **kinematical** forward model by default: no dynamical diffraction and
no multiple scattering. For a near-perfect grain in a topography geometry that is
a load-bearing assumption, not a detail (``implementation_plan.md`` Section 9.1).

**Primary extinction is available and opt-in**, via the ``extinction=`` argument
of :func:`voxel_scattering` / :func:`topograph_image` / :func:`topograph_stack`,
fed by :func:`midas_dct_tt.extinction.extinction_weights`. Switching it on changes
the physics qualitatively rather than by a scale factor: kinematically a distorted
voxel falls out of the acceptance and gets *darker*, whereas with extinction it
escapes the coherent-domain attenuation and gets *brighter* -- the classical
"direct image" of a defect in a topograph. If you care about defect contrast you
must turn it on; if you care only about grain shape you need not.

It remains primary extinction only. **Takagi-Taupin is out of scope and is not
promised.** Use :func:`~midas_hkls.extinction.kinematical_path_limit_um` to check
where you are relative to the boundary; do not assume you are inside it.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .geometry import incident_wavevector
from .project import offdetector_fraction, project_rays_to_plane

__all__ = [
    "PlaneDetector",
    "dct_frames",
    "expand_subvoxels",
    "topograph_image",
    "topograph_stack",
    "voxel_scattering",
]


@dataclass
class PlaneDetector:
    """A plane detector: pixel size, shape, centre, and distance from the sample.

    ``distance_um`` is measured along the detector normal from the sample origin.
    It has **no effect on a parallel projection** (an undeformed grain, all rays
    along the normal) -- it only sets how far per-voxel ray deflections have to
    propagate before they land. That is exactly the lever arm that makes a
    topograph sensitive to intragranular orientation: doubling the distance
    doubles the displacement a given lattice tilt produces.
    """

    pixel_um: float = 1.0
    shape: tuple = (256, 256)
    center_px: tuple = None
    distance_um: float = 5000.0


def expand_subvoxels(positions: torch.Tensor, values: torch.Tensor, spacing_um, n: int):
    """Split each voxel into ``n^3`` sub-points carrying ``1/n^3`` of its value.

    A bilinear splat treats a voxel as a point, which is fine while the pixel is
    comparable to the voxel but leaves gaps once pixels are much finer. This
    spreads each voxel deterministically over its own cube, converging to the box
    footprint as ``n`` grows -- a simpler route to the same place as generalising
    ``midas_diffract.forward_nf_triangles`` to 3-D box voxels, and differentiable
    throughout. Cost is ``n^3`` in memory, so ``n = 2`` or ``3`` in practice.

    Offsets are applied in whatever frame ``positions`` is in; call it **before**
    rotating to the lab so the sub-voxels rotate rigidly with the grain.

    Note the bookkeeping: the *value* is divided by ``n^3`` here, so the caller
    must keep passing the **original** voxel volume to the projector. Dividing
    the volume as well scales the whole image by ``1/n^3`` -- silently, since the
    image still looks entirely plausible.
    """
    if n <= 1:
        return positions, values
    s = spacing_um
    if not isinstance(s, (tuple, list)):
        s = (s, s, s)
    dev, dt = positions.device, positions.dtype
    t = (torch.arange(n, device=dev, dtype=dt) + 0.5) / n - 0.5
    ax = [t * float(si) for si in s]
    gx, gy, gz = torch.meshgrid(ax[0], ax[1], ax[2], indexing="ij")
    off = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)  # (n^3, 3)
    m = off.shape[0]
    pos = (positions.unsqueeze(1) + off.unsqueeze(0)).reshape(-1, 3)
    val = (values.unsqueeze(1).expand(-1, m) / m).reshape(-1)
    return pos, val


def voxel_scattering(
    grain,
    sample_to_lab: torch.Tensor,
    *,
    hkl=None,
    resolution=None,
    k_in: torch.Tensor = None,
    sf2=1.0,
    Q_sample=None,
    extinction=None,
):
    """Per-voxel ``(values, directions)`` for one sample orientation.

    ``Q_sample`` overrides the local scattering vector, so the caller chooses the
    deformation model (exact ``F^-T G0`` or a linearised approximation) rather
    than having it hardwired here. See :mod:`midas_dct_tt.inverse`.

    ``extinction`` is an optional per-voxel primary-extinction factor
    (:func:`~midas_dct_tt.extinction.extinction_weights`); pass it to model the
    dominant contrast mechanism of X-ray topography, in which a *perfect* region
    is dim and a *distorted* one bright. Without it this channel is purely
    kinematical.

    ``values`` is ``chi * sf2 * A(Q_lab) * y`` ``(N,)`` and ``directions`` is the
    per-voxel diffracted beam ``k_in + Q_lab`` ``(N, 3)``.

    With no deformation field, no ``hkl``, or no ``resolution``, the acceptance
    is unity and every ray runs along the nominal beam -- the ideal-grain limit,
    which is the right default for pure shape work and makes the projector an
    exact Radon transform.
    """
    chi = grain.occupancy
    dev, dt = chi.device, chi.dtype
    R = sample_to_lab.to(device=dev, dtype=dt)

    if Q_sample is None and (grain.field is None or hkl is None):
        v = chi * torch.as_tensor(sf2, device=dev, dtype=dt)
        if extinction is not None:
            v = v * torch.as_tensor(extinction, device=dev, dtype=dt)
        return v, None

    if k_in is None:
        raise ValueError(
            "k_in is required once a deformation field is in play: the per-voxel "
            "diffracted direction is k_in + Q(r)"
        )
    if Q_sample is None:
        Q_sample = grain.field.local_Q(hkl)
    Q_lab = Q_sample.to(device=dev, dtype=dt) @ R.transpose(-1, -2)
    w = torch.ones_like(chi) if resolution is None else resolution.weight(Q_lab)
    values = chi * w * torch.as_tensor(sf2, device=dev, dtype=dt)
    if extinction is not None:
        values = values * torch.as_tensor(extinction, device=dev, dtype=dt)
    directions = k_in.to(device=dev, dtype=dt) + Q_lab
    return values, directions


def topograph_image(
    grain,
    alignment,
    psi_deg,
    *,
    detector: PlaneDetector = None,
    hkl=None,
    resolution=None,
    sf2=1.0,
    supersample: int = 1,
    Q_sample=None,
    extinction=None,
    return_offdetector: bool = False,
):
    """One TT topograph ``(n_u, n_v)`` at rotation angle ``psi``.

    The detector is perpendicular to the nominal diffracted beam
    ``alignment.beam_direction()``. Units are ``[sf2] * um``: with the defaults
    (``sf2 = 1``, no acceptance) a pixel reads the **path length through the
    grain in micrometers**, so the centre of a sphere of radius R reads ``2R``.

    Differentiable in the occupancy logits, the deformation field, the alignment,
    and the wavelength.

    Set ``return_offdetector=True`` to get ``(image, fraction)`` where
    ``fraction`` is the mass that missed the detector -- worth checking once per
    geometry, since the splat drops it silently.
    """
    detector = detector or PlaneDetector()
    R = alignment.sample_rotation(psi_deg)
    values, directions = voxel_scattering(
        grain, R, hkl=hkl, resolution=resolution, k_in=alignment.k_in, sf2=sf2,
        extinction=extinction,
        Q_sample=Q_sample,
    )
    normal = alignment.beam_direction()
    if directions is None:
        directions = normal

    pos = grain.positions
    if supersample > 1:
        pos, values = expand_subvoxels(pos, values, grain.spacing_um, supersample)
        if directions.dim() == 2:
            directions = directions.repeat_interleave(supersample ** 3, dim=0)
    pos_lab = pos @ R.to(device=pos.device, dtype=pos.dtype).transpose(-1, -2)

    kw = dict(
        normal=normal,
        distance_um=detector.distance_um,
        voxel_volume_um3=grain.voxel_volume_um3,
        pixel_um=detector.pixel_um,
        detector_shape=detector.shape,
        center_px=detector.center_px,
    )
    img = project_rays_to_plane(pos_lab, values, directions, **kw)
    if not return_offdetector:
        return img
    return img, offdetector_fraction(pos_lab, values, directions, **{
        k: v for k, v in kw.items() if k != "voxel_volume_um3"
    })


def topograph_stack(
    grain,
    alignment,
    psi_deg,
    *,
    detector: PlaneDetector = None,
    hkl=None,
    resolution=None,
    sf2=1.0,
    supersample: int = 1,
    Q_sample=None,
    extinction=None,
) -> torch.Tensor:
    """A TT sinogram: topographs over a psi scan, ``(S, n_u, n_v)``.

    This is the laminographic data a shape reconstruction inverts. Remember the
    geometry it was taken in: the projection direction sits at ``90 - theta``
    from the rotation axis, so the sampling has a missing cone of half-angle
    ``theta`` (:func:`~midas_dct_tt.geometry.missing_cone_half_angle_deg`).
    """
    angles = psi_deg if isinstance(psi_deg, torch.Tensor) else torch.as_tensor(psi_deg)
    return torch.stack(
        [
            topograph_image(
                grain, alignment, a, detector=detector, hkl=hkl,
                resolution=resolution, sf2=sf2, supersample=supersample,
                Q_sample=Q_sample, extinction=extinction,
            )
            for a in angles
        ],
        dim=0,
    )


def dct_frames(
    grains,
    flashes_per_grain,
    *,
    wavelength_A,
    detector: PlaneDetector = None,
    omega_centres: torch.Tensor,
    hkl=None,
    resolution=None,
    sf2=1.0,
    supersample: int = 1,
    omega_sign: float = None,
) -> torch.Tensor:
    """DCT diffraction frames ``(n_frames, n_u, n_v)`` for a grain population.

    Each grain's spot is added to the frame whose omega bin contains its Bragg
    flash, projected along that flash's ``k_out`` onto a detector perpendicular
    to the **incident beam** (unlike TT, whose detector faces the diffracted
    beam). One frame therefore carries spots from many grains at many
    ``2*theta`` -- which is what makes DCT a grain-mapping technique and what
    makes spot-to-grain assignment (Phase 2, ``pairing.py``) necessary.

    Parameters
    ----------
    grains : list
        :class:`~midas_dct_tt.grain.GrainShape` objects, positioned in the sample
        frame (offset their ``positions`` to place them).
    flashes_per_grain : list[list[BraggFlash]]
        Output of :func:`~midas_dct_tt.scan.bragg_flashes` per grain. Passed in
        rather than computed here so the caller controls reflection choice and
        can see which reflections are blind.
    omega_centres : (n_frames,) tensor
        Frame centres from :func:`~midas_dct_tt.scan.dct_omega_scan`.

    Notes
    -----
    This is the **diffraction channel only**. The direct beam and its extinction
    depletion are :mod:`midas_dct_tt.extinction`; a real DCT frame is the sum of
    the two.
    """
    from .conventions import DCT_OMEGA_SIGN_AERO, dct_sample_rotation

    detector = detector or PlaneDetector()
    if omega_sign is None:
        omega_sign = DCT_OMEGA_SIGN_AERO
    if len(grains) != len(flashes_per_grain):
        raise ValueError(
            f"{len(grains)} grains but {len(flashes_per_grain)} flash lists -- "
            "pass one list of flashes per grain (possibly empty)"
        )

    ref = grains[0].positions
    k_in = incident_wavevector(wavelength_A, device=ref.device, dtype=ref.dtype)
    beam = k_in / torch.linalg.vector_norm(k_in)

    centres = omega_centres.to(device=ref.device, dtype=ref.dtype)
    n_frames = centres.shape[0]
    step = float(centres[1] - centres[0]) if n_frames > 1 else 360.0
    lo = float(centres[0]) - 0.5 * step

    nu, nv = detector.shape
    frames = torch.zeros(n_frames, nu, nv, device=ref.device, dtype=ref.dtype)
    for grain, flashes in zip(grains, flashes_per_grain):
        for flash in flashes:
            idx = int(((flash.omega_deg - lo) % 360.0) // step)
            if idx < 0 or idx >= n_frames:
                continue
            R = dct_sample_rotation(flash.omega_deg, omega_sign=omega_sign,
                                    device=ref.device, dtype=ref.dtype)
            values, directions = voxel_scattering(
                grain, R, hkl=hkl, resolution=resolution, k_in=k_in, sf2=sf2
            )
            if directions is None:
                directions = flash.k_out
            pos = grain.positions
            if supersample > 1:
                pos, values = expand_subvoxels(pos, values, grain.spacing_um, supersample)
                if directions.dim() == 2:
                    directions = directions.repeat_interleave(supersample ** 3, dim=0)
            pos_lab = pos @ R.to(device=pos.device, dtype=pos.dtype).transpose(-1, -2)
            frames[idx] = frames[idx] + project_rays_to_plane(
                pos_lab, values, directions,
                normal=beam,
                distance_um=detector.distance_um,
                voxel_volume_um3=grain.voxel_volume_um3,
                pixel_um=detector.pixel_um,
                detector_shape=detector.shape,
                center_px=detector.center_px,
            )
    return frames
