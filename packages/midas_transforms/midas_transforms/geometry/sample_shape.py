"""Sample shape — the illuminated volume a diffraction experiment actually had.

Why this exists
---------------
Grain size in MIDAS is an intensity ratio against a *gauge volume*:

    V_grain = 0.5 * m_hkl * dTheta * cos(Theta) * V_gauge * I_spot
              / (N_imgs * I_powder)                 (radius/core.py:185)

and ``V_gauge = Hbeam * pi * Rsample^2``, overridden by ``Vsample`` when set.
``Hbeam`` and ``Rsample`` are, by a hard project rule, deliberately GENEROUS
SEARCH BOUNDS and never the specimen — ``midas_calibrate_v2`` writes
``Rsample 1000, Hbeam 1000, Vsample 50000000`` as template defaults. So the
**absolute** scale of every reported grain size is a canned constant that
nothing in the pipeline measures.

Measured on the FF reference run (`ff_refiner_prepost/result/LayerNr_1`,
6112 grains): no ``Vsample`` line, so ``V_gauge = 2000 * pi * 2000^2 =
2.513e10 um^3``, and the sum of all grain volumes is **6.5 %** of it. Relative
sizes within a ring survive that; absolute ones carry the constant.

This module supplies the missing quantity. A :class:`SampleShape` is a binary
occupancy volume plus an explicit affine into the MIDAS lab frame, from which
``illuminated_volume`` computes the beam-sample overlap that ``V_gauge`` was
always meant to be.

**A tomogram is not required.** Many FF specimens are cylinders or boxes of
known dimensions, and :meth:`SampleShape.cylinder` / :meth:`box` give the
correction with no reconstruction at all. Tomography buys a real shape, and
the absorption path on top; it is not a prerequisite for the volume term.

Frames
------
The affine is delegated entirely to
:func:`midas_stress.frames.tomo_grid_to_midas`, which is the single source of
truth for the MIDAS/APS/reconstruction-grid relationship and refuses to default
the pixel size, the rotation-axis position, or the in-plane handedness. See
``manuals/tomo/COORDINATES.md``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = ["SampleShape"]


@dataclass
class SampleShape:
    """An occupancy volume registered into the MIDAS lab frame.

    ``occupancy`` is indexed ``[slice, iy, ix]`` — ``slice`` runs along the
    rotation axis (MIDAS z), matching a tomographic reconstruction. Values are
    a fraction in [0, 1]: 1 inside the sample, 0 outside. Fractional values are
    allowed so an analytic shape can anti-alias its boundary, which matters
    because the boundary is where the volume estimate is most sensitive.

    Every geometric parameter is required, never defaulted — see
    :func:`midas_stress.frames.tomo_grid_to_midas` for why each one cannot be
    guessed.
    """

    occupancy: np.ndarray
    pixel_size_um: float
    slice_pitch_um: float
    rot_axis_ix: float
    rot_axis_iy: float
    slice0_z_um: float = 0.0
    in_plane: str = "xy"
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.occupancy = np.asarray(self.occupancy, dtype=np.float64)
        if self.occupancy.ndim != 3:
            raise ValueError(
                f"occupancy must be 3-D [slice, iy, ix]; got shape "
                f"{self.occupancy.shape}. A single slice is shape (1, ny, nx)."
            )
        if self.occupancy.min() < -1e-9 or self.occupancy.max() > 1 + 1e-9:
            raise ValueError(
                "occupancy must lie in [0, 1]; it is a volume fraction, not a "
                "density or an attenuation coefficient."
            )
        if not (self.pixel_size_um > 0 and self.slice_pitch_um > 0):
            raise ValueError("pixel_size_um and slice_pitch_um must be > 0")
        from midas_stress.frames import TOMO_IN_PLANE

        if self.in_plane not in TOMO_IN_PLANE:
            raise ValueError(
                f"in_plane must be one of {sorted(TOMO_IN_PLANE)}; got "
                f"{self.in_plane!r}. There is no safe default — the wrong "
                "choice mirrors the sample, which is silent."
            )

    # ---------------------------------------------------------------- shape

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(int(s) for s in self.occupancy.shape)

    @property
    def voxel_volume_um3(self) -> float:
        return float(self.pixel_size_um ** 2 * self.slice_pitch_um)

    def total_volume_um3(self) -> float:
        """Volume of the whole sample as represented, ignoring the beam."""
        return float(self.occupancy.sum()) * self.voxel_volume_um3

    # ------------------------------------------------------- analytic shapes

    @classmethod
    def cylinder(
        cls,
        *,
        diameter_um: float,
        height_um: float,
        pixel_size_um: float,
        slice_pitch_um: Optional[float] = None,
        centre_z_um: float = 0.0,
        supersample: int = 4,
    ) -> "SampleShape":
        """A right circular cylinder on the rotation axis — the common FF rod.

        Needs no tomography. The axis is MIDAS z, so the cylinder is the shape
        whose illuminated volume is *independent of omega* — which is exactly
        why it is also the shape on which most registration checks have no
        power (see ``manuals/tomo/DIAGNOSIS.md``).

        ``supersample`` anti-aliases the boundary by evaluating a sub-grid per
        voxel. The boundary is a surface term, so at 1 (no supersampling) a
        50-voxel-diameter cylinder carries a few percent volume error — which
        propagates directly into every grain volume.
        """
        if diameter_um <= 0 or height_um <= 0:
            raise ValueError("diameter_um and height_um must be > 0")
        pitch = float(slice_pitch_um if slice_pitch_um is not None else pixel_size_um)
        r_um = 0.5 * float(diameter_um)

        n_xy = int(math.ceil(diameter_um / pixel_size_um)) + 2
        n_z = max(1, int(round(height_um / pitch)))
        c = (n_xy - 1) / 2.0

        occ = _disc_occupancy(n_xy, c, r_um / pixel_size_um, supersample)
        vol = np.repeat(occ[None, :, :], n_z, axis=0)

        return cls(
            occupancy=vol,
            pixel_size_um=float(pixel_size_um),
            slice_pitch_um=pitch,
            rot_axis_ix=c, rot_axis_iy=c,
            slice0_z_um=float(centre_z_um) - 0.5 * (n_z - 1) * pitch,
            in_plane="xy",
            provenance={
                "source": "analytic:cylinder",
                "diameter_um": float(diameter_um),
                "height_um": float(height_um),
                "supersample": int(supersample),
                "registration": "exact by construction (axis-centred)",
            },
        )

    @classmethod
    def box(
        cls,
        *,
        size_x_um: float,
        size_y_um: float,
        height_um: float,
        pixel_size_um: float,
        slice_pitch_um: Optional[float] = None,
        centre_z_um: float = 0.0,
    ) -> "SampleShape":
        """A rectangular prism centred on the rotation axis.

        Unlike a cylinder its illuminated volume *does* vary with omega, which
        makes it the useful case for the V1 sinogram registration check — a
        cylinder gives that check zero power.
        """
        pitch = float(slice_pitch_um if slice_pitch_um is not None else pixel_size_um)
        nx = max(1, int(round(size_x_um / pixel_size_um)))
        ny = max(1, int(round(size_y_um / pixel_size_um)))
        n_z = max(1, int(round(height_um / pitch)))
        vol = np.ones((n_z, ny, nx), dtype=np.float64)
        return cls(
            occupancy=vol,
            pixel_size_um=float(pixel_size_um),
            slice_pitch_um=pitch,
            rot_axis_ix=(nx - 1) / 2.0, rot_axis_iy=(ny - 1) / 2.0,
            slice0_z_um=float(centre_z_um) - 0.5 * (n_z - 1) * pitch,
            in_plane="xy",
            provenance={
                "source": "analytic:box",
                "size_x_um": float(size_x_um), "size_y_um": float(size_y_um),
                "height_um": float(height_um),
                "registration": "exact by construction (axis-centred)",
            },
        )

    # ------------------------------------------------------ lab-frame coords

    def voxel_positions_um(self):
        """``(n_slice, n_iy, n_ix, 3)`` MIDAS lab coordinates of every voxel.

        Delegates the affine to ``midas_stress.frames`` — one source of truth
        for the frame relationship.
        """
        from midas_stress.frames import tomo_grid_to_midas

        ns, ny, nx = self.shape
        s, iy, ix = np.meshgrid(
            np.arange(ns, dtype=np.float64),
            np.arange(ny, dtype=np.float64),
            np.arange(nx, dtype=np.float64),
            indexing="ij",
        )
        x, y, z = tomo_grid_to_midas(
            s, iy, ix,
            pixel_size_um=self.pixel_size_um,
            slice_pitch_um=self.slice_pitch_um,
            rot_axis_ix=self.rot_axis_ix, rot_axis_iy=self.rot_axis_iy,
            slice0_z_um=self.slice0_z_um, in_plane=self.in_plane,
        )
        return np.stack([x, y, z], axis=-1)

    def contains(self, points_um, *, threshold: float = 0.5,
                 translation_um=(0.0, 0.0, 0.0)) -> np.ndarray:
        """Is each MIDAS-lab point inside the sample? ``(N, 3) -> (N,) bool``.

        Nearest-voxel lookup through
        :func:`midas_stress.frames.midas_to_tomo_grid`, so the forward and
        inverse affines cannot drift apart. Points outside the reconstructed
        volume are **False**, never clamped to the edge voxel — clamping would
        report a point a millimetre above the tomogram as inside the sample.

        ``translation_um`` shifts the *points* before the lookup, which is the
        registration offset the V2 check fits.
        """
        from midas_stress.frames import midas_to_tomo_grid

        p = np.asarray(points_um, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError(f"points_um must be (N, 3); got {p.shape}")
        t = np.asarray(translation_um, dtype=np.float64).reshape(3)
        p = p + t

        s, iy, ix = midas_to_tomo_grid(
            p[:, 0], p[:, 1], p[:, 2],
            pixel_size_um=self.pixel_size_um, slice_pitch_um=self.slice_pitch_um,
            rot_axis_ix=self.rot_axis_ix, rot_axis_iy=self.rot_axis_iy,
            slice0_z_um=self.slice0_z_um, in_plane=self.in_plane,
        )
        ns, ny, nx = self.shape
        si = np.rint(s).astype(np.int64)
        yi = np.rint(iy).astype(np.int64)
        xi = np.rint(ix).astype(np.int64)
        ok = ((si >= 0) & (si < ns) & (yi >= 0) & (yi < ny)
              & (xi >= 0) & (xi < nx))
        out = np.zeros(p.shape[0], dtype=bool)
        out[ok] = self.occupancy[si[ok], yi[ok], xi[ok]] >= float(threshold)
        return out

    def mirrored(self) -> "SampleShape":
        """The same shape with its in-plane handedness flipped.

        The meta-null for every registration check. A mirrored mask is smooth,
        plausible and wrong, and it is the failure mode centroid containment
        cannot see on a near-symmetric sample. If a check scores the same on
        this as on the original, that check has no power over the failure it
        exists to catch — see ``manuals/tomo/DIAGNOSIS.md``.
        """
        flip = {"xy": "x-y", "x-y": "xy", "-xy": "-x-y", "-x-y": "-xy",
                "yx": "-yx", "-yx": "yx", "y-x": "-y-x", "-y-x": "y-x"}
        prov = dict(self.provenance)
        prov["mirrored_from"] = prov.get("source", "unknown")
        prov["source"] = "META-NULL: mirrored copy, not a real sample shape"
        return SampleShape(
            occupancy=self.occupancy.copy(),
            pixel_size_um=self.pixel_size_um, slice_pitch_um=self.slice_pitch_um,
            rot_axis_ix=self.rot_axis_ix, rot_axis_iy=self.rot_axis_iy,
            slice0_z_um=self.slice0_z_um, in_plane=flip[self.in_plane],
            provenance=prov,
        )

    # -------------------------------------------------- the illuminated volume

    def illuminated_volume_um3(
        self,
        *,
        beam_height_um: Optional[float] = None,
        beam_width_um: Optional[float] = None,
        beam_centre_z_um: float = 0.0,
        beam_centre_y_um: float = 0.0,
        omega_deg: float = 0.0,
    ) -> float:
        """Volume of sample actually lit, at one omega. THE replacement for
        ``V_gauge``.

        ``beam_height_um`` is the slab height along MIDAS z; ``beam_width_um``
        the horizontal extent transverse to the beam. Either may be None,
        meaning "unbounded in that direction" — a beam wider than the sample.

        The beam is fixed in the lab while the sample rotates, so the
        horizontal cut is applied to the sample's projection at ``omega_deg``.
        A cylinder gives the same answer at every omega; anything else does
        not, and that omega-dependence is the V1 registration signal.

        Both edges are treated as hard (top-hat). A measured profile belongs in
        ``geometry/beam.py``; this is the zeroth-order number and is what
        ``V_gauge`` was always trying to be.
        """
        pos = self.voxel_positions_um()
        occ = self.occupancy
        keep = np.ones_like(occ, dtype=bool)

        if beam_height_um is not None:
            dz = np.abs(pos[..., 2] - beam_centre_z_um)
            keep &= dz <= 0.5 * float(beam_height_um)

        if beam_width_um is not None:
            # Sample-frame (x, y) seen by a lab-fixed beam at this omega. The
            # transverse lab coordinate of a sample point is
            #   y_lab = x sin(omega) + y cos(omega)
            # matching midas_index.compute.matching's projection convention.
            w = math.radians(omega_deg)
            y_lab = pos[..., 0] * math.sin(w) + pos[..., 1] * math.cos(w)
            keep &= np.abs(y_lab - beam_centre_y_um) <= 0.5 * float(beam_width_um)

        return float((occ * keep).sum()) * self.voxel_volume_um3

    def illuminated_volume_sinogram(
        self, omegas_deg, **kw
    ) -> np.ndarray:
        """``illuminated_volume_um3`` across omegas — the V1 check's prediction.

        Its modulation is what a per-frame ring intensity should track. On a
        cylinder it is flat, so the check has **no power** there; compute it
        before claiming a registration was verified.
        """
        return np.array(
            [self.illuminated_volume_um3(omega_deg=w, **kw) for w in np.asarray(omegas_deg)]
        )

    def gauge_volume_ratio(self, v_gauge_um3: float, **kw) -> float:
        """``V_illum / V_gauge`` — the factor every grain VOLUME is wrong by.

        Grain *radius* is wrong by its cube root. This is the whole of the
        first-order correction: it is global, needs no absorption model, and
        cancels nothing (unlike the per-spot terms, where only the spread
        survives the powder-normalisation ratio).
        """
        if v_gauge_um3 <= 0:
            raise ValueError("v_gauge_um3 must be > 0")
        return self.illuminated_volume_um3(**kw) / float(v_gauge_um3)

    # --------------------------------------------------------- to SampleGrid

    def to_sample_grid(self, *, threshold: float = 0.5, device=None, dtype=None):
        """Bridge to :class:`midas_transforms.geometry.SampleGrid`.

        Uses ``from_regular_grid`` so ``grid_origin_um`` and ``grid_shape`` are
        populated — ``absorption_factor`` raises without them, which is why the
        existing V-map path (which uses ``from_arrays``) cannot reach the
        absorption code at all.
        """
        import torch

        from .sample import SampleGrid

        if not (self.pixel_size_um == self.slice_pitch_um):
            raise ValueError(
                f"SampleGrid assumes cubic voxels but pixel_size_um="
                f"{self.pixel_size_um} != slice_pitch_um={self.slice_pitch_um}. "
                "Resample before converting rather than letting the ray tracer "
                "silently use the wrong length along one axis."
            )
        pos = self.voxel_positions_um().reshape(-1, 3)
        mask = (self.occupancy.reshape(-1) >= float(threshold))
        dt = dtype or torch.float64
        ns, ny, nx = self.shape
        return SampleGrid.from_regular_grid(
            grid_origin_um=torch.tensor(pos[0], dtype=dt, device=device),
            grid_shape=(nx, ny, ns),
            voxel_size_um=float(self.pixel_size_um),
            sample_mask=torch.tensor(mask, dtype=torch.bool, device=device),
            device=device, dtype=dt,
        )


def _disc_occupancy(n: int, centre: float, r_px: float, supersample: int) -> np.ndarray:
    """Area fraction of a disc inside each pixel of an ``n x n`` grid."""
    ss = max(1, int(supersample))
    off = (np.arange(ss) + 0.5) / ss - 0.5
    iy, ix = np.mgrid[0:n, 0:n].astype(np.float64)
    acc = np.zeros((n, n), dtype=np.float64)
    for dy in off:
        for dx in off:
            rr = np.hypot(iy + dy - centre, ix + dx - centre)
            acc += (rr <= r_px)
    return acc / (ss * ss)
