"""Magnifying objective optics: sample voxel -> detector pixel projection.

Phase 1 of ``implementation_plan.md`` — the NEW real-space imaging geometry.

The objective images the illuminated sample volume along the diffracted-beam
optical axis ``k_out`` with magnification ``M`` onto a detector perpendicular to
``k_out``. A sample voxel at lab position ``r`` maps to detector coordinates by
projecting ``r`` onto the plane transverse to ``k_out`` (an orthonormal detector
basis ``(u, v)``), scaling by ``M``, and converting to pixels. The imaging is an
*inclined* projection because the sample plane is tilted relative to the optical
axis — captured here by the choice of ``k_out`` (from ``2*theta``) rather than by
an ad-hoc tilt.

Differentiable in positions, magnification, and ``k_out`` direction; device/dtype
portable. Detector distortion, when needed, composes via ``midas_distortion``
(single source of truth) — omitted from the ideal Phase-1 projection.

Units: positions in micrometers; pixel size in micrometers; pixels dimensionless.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .conventions import as_geometry


def diffracted_beam_direction(two_theta_deg: float, *, geometry=None,
                              device=None, dtype=torch.float64) -> torch.Tensor:
    """Unit ``k_out`` for a reflection at ``2*theta`` in the scattering plane.

    ``k_out = cos(2theta) * beam + sin(2theta) * deflection``, with both
    directions taken from ``geometry`` (a :class:`~midas_dfxm.conventions.
    ScatteringGeometry`, the string ``"vertical"``/``"horizontal"``, or ``None``).

    The default is the **vertical** scattering plane in the MIDAS frame -- beam
    along ``x``, plane ``x``-``z``, ``k_out = (cos 2theta) xhat + (sin 2theta)
    zhat`` -- which is the ESRF ID06-HXM geometry. Pass
    ``geometry="horizontal"`` for a transmission geometry that scatters in the
    ``x``-``y`` plane (APS 6-ID-C).

    Techniques that image *one chosen grain's* reflection -- topotomography, DCT
    -- get whatever direction crystallography gives them, and should pass an
    explicit ``k_out`` to :class:`ObjectiveOptics` instead. See
    :meth:`ObjectiveOptics.from_k_out`.
    """
    return as_geometry(geometry).k_out(two_theta_deg, device=device, dtype=dtype)


def two_theta_from_k_out(k_out, *, geometry=None) -> float:
    """Scattering angle ``2*theta`` (degrees) of an arbitrary ``k_out``.

    ``2*theta`` is by definition the angle between the incident beam and
    ``k_out``, so this inverts :func:`diffracted_beam_direction` for any
    direction, not only in-plane ones.

    ``geometry`` supplies the beam direction. The default assumes the beam is
    along ``x`` (MIDAS frame) -- pass a geometry built on
    :data:`~midas_dfxm.conventions.APS_FRAME` if ``k_out`` has APS components,
    or the answer is silently wrong rather than an error. How wrong depends on
    the scattering plane, and neither case raises: APS components read as MIDAS
    give ``90 - 2*theta`` for a horizontal plane (12 deg reads as 78) and a flat
    ``90`` for a vertical one, where the beam component is identically zero.
    """
    k = torch.as_tensor(k_out)
    if not torch.is_floating_point(k):
        k = k.to(torch.float64)
    n = torch.linalg.vector_norm(k.detach())
    if float(n) == 0.0:
        raise ValueError("k_out must be a non-zero vector")
    beam = as_geometry(geometry).beam_direction(device=k.device, dtype=k.dtype)
    c = ((k.detach() @ beam) / n).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(c)))


def detector_basis(k_out: torch.Tensor, *, geometry=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Orthonormal detector axes ``(u, v)`` spanning the plane transverse to ``k_out``.

    ``u`` is transverse to the frame's up direction, ``v`` completes a
    right-handed set (so for an in-plane ``k_out``, ``v`` is the frame's up).
    Returns two ``(3,)`` unit vectors.
    """
    geom = as_geometry(geometry)
    k = k_out / torch.linalg.vector_norm(k_out)
    up = geom.frame.axis("up", device=k.device, dtype=k.dtype)
    if torch.abs(torch.dot(k, up)) > 0.9:
        up = geom.frame.axis("outboard", device=k.device, dtype=k.dtype)
    u = torch.linalg.cross(up, k)
    u = u / torch.linalg.vector_norm(u)
    v = torch.linalg.cross(k, u)
    return u, v


@dataclass
class ObjectiveOptics:
    """Magnifying objective imaging model.

    Attributes
    ----------
    two_theta_deg : float
        Bragg ``2*theta`` of the imaged reflection (sets the optical axis ``k_out``).
    magnification : float
        Transverse magnification ``M`` of the objective.
    pixel_um : float
        Detector pixel size in micrometers.
    detector_shape : tuple[int, int]
        ``(n_u, n_v)`` detector dimensions in pixels.
    center_px : tuple[float, float]
        Pixel coordinate of the optical axis (defaults to detector centre).
    detector_tilt_deg : float
        Physical tilt of the detector face away from perpendicular to ``k_out``,
        about ``tilt_axis`` (default ``0`` = perpendicular, unchanged). A grazing
        detector is used to *increase resolution*: the in-plane axis orthogonal to
        ``tilt_axis`` is stretched by ``1 / cos(tilt)`` (a sample feature spreads
        over more pixels), so its effective object-space pixel shrinks by
        ``cos(tilt)``. For the parallel (telecentric) projection this class models
        the scaling is **exact** -- it equals the ray/tilted-plane intersection with
        no keystone (validated in ``tests/test_optics_kout.py`` against an
        independent geometric construction). A real finite-conjugate objective adds
        perspective keystone and Scheimpflug defocus, which are NOT modelled here;
        panel distortion is deferred to ``midas_distortion``. Note this is a
        *separate* effect from the inclined-projection foreshortening already set
        by ``2*theta`` -- that one is present at zero tilt.
    tilt_axis : {"u", "v"}
        Detector in-plane axis the panel is rotated about (default ``"u"``). The
        stretched axis is the orthogonal one.
    k_out : tensor (3,), optional
        Explicit optical axis in the lab frame. When ``None`` (the default) the
        axis is derived from ``two_theta_deg`` and lies in ``geometry``'s
        scattering plane. Supply it when the imaged reflection is *not* in that
        plane -- topotomography and DCT image one chosen grain's ``G``, which
        points wherever crystallography puts it. Need not be normalised.
        Gradients flow through it.

        It must agree with ``two_theta_deg`` (checked at construction, since
        ``two_theta_deg`` remains the scattering angle other code reads);
        :meth:`from_k_out` derives the pair consistently for you.
    geometry : ScatteringGeometry | str, optional
        Lab frame + scattering plane. ``None`` (the default) is the vertical
        plane in the MIDAS frame (ESRF ID06-HXM); pass ``"horizontal"`` for an
        APS 6-ID-C style transmission geometry. It also sets the frame used to
        read an explicit ``k_out`` and to build the detector basis.

    Notes
    -----
    Setting ``M = 1`` with an explicit ``k_out`` gives the unmagnified parallel
    projection along an arbitrary diffracted beam -- i.e. the TT/DCT topograph
    geometry -- with no other change to this class.
    """

    two_theta_deg: float
    magnification: float = 10.0
    pixel_um: float = 1.0
    detector_shape: tuple[int, int] = (256, 256)
    center_px: tuple[float, float] | None = None
    k_out: torch.Tensor | None = None
    NA: float | None = None
    wavelength_A: float | None = None
    detector_tilt_deg: float = 0.0
    tilt_axis: str = "u"
    geometry: object = None

    def __post_init__(self):
        self.geometry = as_geometry(self.geometry)
        if self.tilt_axis not in ("u", "v"):
            raise ValueError(f"tilt_axis must be 'u' or 'v', got {self.tilt_axis!r}")
        if self.k_out is None:
            return
        # An inconsistent (k_out, two_theta_deg) pair is a silent physics bug:
        # the projection would use one direction while anything reading
        # two_theta_deg (resolution widths, Lorentz factors) uses another.
        # Check once, here, rather than on every project() call.
        tt = two_theta_from_k_out(self.k_out, geometry=self.geometry)
        if abs(tt - float(self.two_theta_deg)) > 1e-6:
            raise ValueError(
                f"k_out is at 2*theta = {tt:.9g} deg but two_theta_deg = "
                f"{float(self.two_theta_deg):.9g} deg. They must agree -- "
                "use ObjectiveOptics.from_k_out(k_out, ...) to derive both."
            )

    @classmethod
    def from_k_out(cls, k_out, **kwargs) -> "ObjectiveOptics":
        """Build from an explicit optical axis, deriving ``two_theta_deg`` from it.

        The TT/DCT entry point: pass ``k_out = k_in + G`` for the grain and
        reflection being imaged, plus ``magnification=1.0`` for an unmagnified
        parallel projection. Remaining keyword arguments are as for the
        constructor.

        ``two_theta_deg`` is stored as a plain float, so gradients w.r.t. the
        scattering angle do not flow through *it* -- but the projection uses
        ``k_out`` directly, so gradients w.r.t. the beam direction do.
        """
        if "two_theta_deg" in kwargs:
            raise TypeError(
                "from_k_out derives two_theta_deg from k_out; do not pass it as well"
            )
        geometry = as_geometry(kwargs.pop("geometry", None))
        return cls(two_theta_deg=two_theta_from_k_out(k_out, geometry=geometry),
                   k_out=k_out, geometry=geometry, **kwargs)

    def optical_axis(self, *, device=None, dtype=torch.float64) -> torch.Tensor:
        """Unit optical axis ``(3,)``: explicit ``k_out`` if set, else from ``2*theta``."""
        if self.k_out is None:
            return diffracted_beam_direction(self.two_theta_deg, geometry=self.geometry,
                                             device=device, dtype=dtype)
        k = self.k_out
        if not isinstance(k, torch.Tensor):
            k = torch.as_tensor(k, device=device, dtype=dtype)
        else:
            k = k.to(device=device, dtype=dtype)
        return k / torch.linalg.vector_norm(k)

    def project(self, positions_lab: torch.Tensor) -> torch.Tensor:
        """Project lab-frame voxel positions ``(N, 3)`` to pixel coords ``(N, 2)``.

        Returns fractional ``(u_px, v_px)``. Differentiable in positions, ``M``,
        and (when supplied) ``k_out``.
        """
        k_out = self.optical_axis(device=positions_lab.device, dtype=positions_lab.dtype)
        u, v = detector_basis(k_out, geometry=self.geometry)
        cu, cv = self._center(positions_lab)
        su, sv = self._anamorphic_scales()
        pu = su * self.magnification * (positions_lab @ u) / self.pixel_um + cu
        pv = sv * self.magnification * (positions_lab @ v) / self.pixel_um + cv
        return torch.stack([pu, pv], dim=-1)

    def _anamorphic_scales(self) -> tuple[float, float]:
        """``(s_u, s_v)`` magnification factors from a physical detector tilt.

        A planar detector rotated by ``detector_tilt_deg`` about ``tilt_axis``
        records the grazing intersection of the (along-``k_out``) imaging rays with
        the tilted face: extents along the in-plane axis *orthogonal* to
        ``tilt_axis`` are stretched by ``1 / cos(tilt)`` -- more pixels per object
        micrometer, the reason grazing detectors are used to gain resolution. Zero
        tilt returns ``(1, 1)`` and leaves the projection untouched.
        """
        import math
        g = math.radians(float(self.detector_tilt_deg))
        if g == 0.0:
            return 1.0, 1.0
        c = math.cos(g)
        if abs(c) < 1e-9:
            raise ValueError(
                "detector_tilt_deg too close to 90 deg -- grazing projection is singular"
            )
        s = 1.0 / c
        return (1.0, s) if self.tilt_axis == "u" else (s, 1.0)

    def _center(self, ref: torch.Tensor) -> tuple[float, float]:
        if self.center_px is not None:
            return self.center_px
        nu, nv = self.detector_shape
        return (nu - 1) / 2.0, (nv - 1) / 2.0

    def psf(self, coeffs=None, *, defocus=0.0, grid_size=128, extent=1.4,
            apodization=1.5, device=None, dtype=torch.float64) -> torch.Tensor:
        """Objective point-spread function sampled at the DETECTOR pixel grid.

        Builds the wave-optics pupil PSF (``aberration.aberrated_psf`` --
        ``|FFT(pupil)|^2`` for a Zernike-aberrated, Gaussian-apodized aperture),
        then resamples it from its native diffraction scale onto this detector's
        object-space pixel so it can be convolved with a rendered image.

        Sampling: the pupil grid spans the unit-disk aperture, so the native PSF
        pixel is ``lambda / (2 * extent * NA)`` in object space; one detector pixel
        images ``pixel_um / M`` of object space. The PSF is scaled by their ratio.
        A diffraction spot finer than a detector pixel collapses to a delta (no
        blur), which is physically correct.

        ``coeffs`` is a dict/tensor of Zernike amplitudes (rad); ``None`` gives the
        unaberrated diffraction-limited PSF. Requires ``NA`` and ``wavelength_A``
        to be set. Differentiable in ``coeffs`` and ``defocus``.
        """
        if self.NA is None or self.wavelength_A is None:
            raise ValueError("ObjectiveOptics.psf requires NA and wavelength_A to be set")
        from .aberration import aberrated_psf
        psf = aberrated_psf(coeffs if coeffs is not None else {}, defocus=defocus,
                            grid_size=grid_size, extent=extent, apodization=apodization,
                            device=device, dtype=dtype)
        lam_um = float(self.wavelength_A) * 1e-4
        dx_native = lam_um / (2.0 * extent * float(self.NA))   # object-space um/PSF-pixel
        dx_image = float(self.pixel_um) / float(self.magnification)  # object-space um/img-pixel
        scale = dx_native / dx_image
        n = max(1, int(round(grid_size * scale)))
        if n != grid_size:
            psf = torch.nn.functional.interpolate(
                psf[None, None], size=(n, n), mode="bilinear", align_corners=False
            )[0, 0].clamp_min(0.0)
        # keep the PSF no larger than the detector (a concentrated PSF's cropped
        # tails carry negligible mass); convolve_psf embeds a smaller PSF centred.
        cap = min(self.detector_shape)
        if psf.shape[-1] > cap:
            off = (psf.shape[-1] - cap) // 2
            psf = psf[off:off + cap, off:off + cap]
        return psf / psf.sum()

    def _psf_scale(self, extent, grid_size):
        """Interp size to map the native pupil-PSF pixel onto the detector pixel."""
        lam_um = float(self.wavelength_A) * 1e-4
        dx_native = lam_um / (2.0 * extent * float(self.NA))
        dx_image = float(self.pixel_um) / float(self.magnification)
        return max(1, int(round(grid_size * (dx_native / dx_image))))

    def amplitude_psf(self, coeffs=None, *, defocus=0.0, grid_size=128, extent=1.4,
                      apodization=1.5, device=None, dtype=torch.float64) -> torch.Tensor:
        """Complex amplitude PSF ``h`` (for coherent imaging), detector-sampled.

        Like :meth:`psf` but returns the field-amplitude PSF from
        ``coherence.coherent_psf`` (normalized ``sum |h|^2 = 1``) instead of the
        intensity PSF, for use with :func:`midas_dfxm.dfxm_image_wave`.
        """
        if self.NA is None or self.wavelength_A is None:
            raise ValueError("ObjectiveOptics.amplitude_psf requires NA and wavelength_A")
        from .coherence import coherent_psf
        h = coherent_psf(coeffs if coeffs is not None else {}, defocus=defocus,
                         grid_size=grid_size, extent=extent, apodization=apodization,
                         device=device, dtype=dtype)
        n = self._psf_scale(extent, grid_size)
        if n != grid_size:
            hr = torch.nn.functional.interpolate(h.real[None, None], size=(n, n),
                                                 mode="bilinear", align_corners=False)[0, 0]
            hi = torch.nn.functional.interpolate(h.imag[None, None], size=(n, n),
                                                 mode="bilinear", align_corners=False)[0, 0]
            h = torch.complex(hr, hi)
        cap = min(self.detector_shape)
        if h.shape[-1] > cap:
            off = (h.shape[-1] - cap) // 2
            h = h[off:off + cap, off:off + cap]
        return h / torch.sqrt((h.abs() ** 2).sum())

    def render(self, positions_lab: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
        """Accumulate per-voxel ``intensity`` onto the detector via bilinear splat.

        Returns an image ``(n_u, n_v)``. Differentiable in ``intensity`` and
        positions (bilinear weights carry gradients). Voxels off the detector are
        dropped.
        """
        px = self.project(positions_lab)
        nu, nv = self.detector_shape
        img = torch.zeros(nu, nv, device=positions_lab.device, dtype=intensity.dtype)
        u = px[:, 0]
        v = px[:, 1]
        u0 = torch.floor(u).long()
        v0 = torch.floor(v).long()
        du = u - u0
        dv = v - v0
        for iu, ju, wu in ((u0, 0, 1 - du), (u0 + 1, 1, du)):
            for iv, jv, wv in ((v0, 0, 1 - dv), (v0 + 1, 1, dv)):
                inb = (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv)
                if not bool(inb.any()):
                    continue
                w = (wu * wv * intensity)[inb]
                flat = (iu[inb] * nv + iv[inb])
                img.view(-1).index_add_(0, flat, w)
        return img
