"""Exact polygon-area pixel-bin intersection (the MIDAS differentiator).

pyFAI and most other azimuthal integrators approximate pixel-bin overlap
by either subpixel oversampling (sample many points inside the pixel)
or smooth kernels (Gaussian / linear interpolation). Both blur the
profile near peak edges.

MIDAS uses the **exact polygon area** of the intersection between the
pixel quad (in detector Y/Z space) and the polar bin (bounded by two
circular arcs and two radial rays). This is computed via Green's
theorem on the signed boundary, with arc segments contributing the
analytic ``R²/2 · dθ`` term — same algorithm as
:func:`midas_integrate.geometry.pixel_bin_intersect`, which this module
delegates to.

Key properties:

- **Exact** polygon-vs-arc intersection to fp64 precision per (pixel, bin)
  pair — no oversampling artefact at peak edges.

  Read that precisely. Pixel corners are mapped into the corrected frame
  through the package's shared geometry (tilt, distortion, parallax, panel
  shifts, residual map) and the quad's edges are then taken as straight
  chords. Where tilt or distortion is nonzero the true edge is slightly
  curved, so the intersection is exact for the chord quad rather than for the
  curved one — an error second order in pixel size. Before issue #69 this
  kernel applied NO corrections at all, an error first order in the
  correction itself (1.46 px at 5° tilt, 3.61 px for a ±2/±3 px panel
  shift), so this is a large net gain, but "exact" is not unqualified.
- **No numba** — v1's polygon math is pure numpy. The build is slower
  than the numba path (sub-second on a 24×24 demo, ~5-30 s on a 2.5 Mpx
  Pilatus), but is the only pure-Python route to v1-level accuracy.
- **Pure-torch integrate**: once the geometry is built, the
  :func:`integrate_polygon` call is a single ``index_add_`` per
  contribution — the same hot-path shape as the other v2 binning
  geometries.

When to use which:

- :class:`HardBinGeometry` — single sample per pixel; max throughput,
  visible bin-edge quantisation.
- :class:`SubpixelBinGeometry` (K=2-4) — oversampled hard-bin; closes
  most of the quantisation at modest extra cost; pyFAI-level fidelity.
- :class:`PolygonBinGeometry` — **exact**; differentiates MIDAS from
  every other integrator; ~5-30 s build on a 2.5 Mpx detector (one-time)
  and pure-torch O(n_entries) integrate after that.
- :class:`SoftBinGeometry` — differentiable in geometry; for refinement,
  not for batch integration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from midas_integrate.geometry import (
    pixel_bin_intersect, REta_to_YZ, PIXEL_CORNER_OFFSETS, QUAD_ORDER,
)

from ..forward import pixel_to_REta_from_spec
from ..forward.pixels import _panel_inputs_from_spec
from midas_calibrate_v2.forward.panels import panel_idx_for_points
from ..spec import IntegrationSpec
from .mask import normalise_mask
from .trans_opt import apply_trans_opt_forward, needs_trans_opt


def _build_one_row(
    j: int,
    NY: int,
    corners_row: np.ndarray,             # (NY, 4, 2)
    quad_row: np.ndarray,                # (NY,) mapped quad area per pixel
    in_range_row: np.ndarray,            # (NY,)
    r_lo_row: np.ndarray, r_hi_row: np.ndarray,
    e_lo_row: np.ndarray, e_hi_row: np.ndarray,
    RMin: float, RBinSize: float,
    EtaMin: float, EtaBinSize: float,
    n_r: int, n_eta: int,
    mask_row: Optional[np.ndarray] = None,    # (NY,) bool
) -> tuple:
    """Build polygon-area entries for one detector row.

    Returns three numpy arrays: ``(pix_idx, bin_idx, area)``. Top-level
    so ``joblib.Parallel`` can pickle it across worker processes.
    Pixels with ``mask_row[i] == True`` are skipped.

    **Performance**. The exact-polygon math (v1's
    :func:`pixel_bin_intersect`) is the math the MIDAS calibration
    paper uses; we never approximate it. Two optimizations are layered
    on top, both exact:

    1. **Trivial fast path**: pixels whose 4 corners all land in the
       same ``(R bin, η bin)`` get ``area = 1.0`` directly, no scalar
       Python kernel call. Helps roughly when ``RBinSize ≥ 1 px`` AND
       ``EtaBinSize ≥ 5°`` (typical production-batch integration).
       For sub-pixel ``RBinSize`` setups (production calibrant
       integration), no pixel meets the condition and every pixel
       falls through to the polygon kernel — that's the cost of
       sub-pixel R resolution; we never trade exactness for speed.
    2. **Per-row joblib parallelism** (``n_jobs`` in
       :meth:`PolygonBinGeometry.from_spec`). Embarrassingly parallel
       across detector rows; near-linear speedup up to physical core
       count.

    The build runs once per spec change; integration is fast (single
    ``index_add`` per (pixel, bin) entry). Build cost is amortised
    across all subsequent integrations.
    """
    bin_corners_cache: dict = {}

    def _bin_corners(r_idx: int, e_idx: int) -> np.ndarray:
        key = (r_idx, e_idx)
        if key not in bin_corners_cache:
            R0 = RMin + r_idx * RBinSize
            R1 = R0 + RBinSize
            E0 = EtaMin + e_idx * EtaBinSize
            E1 = E0 + EtaBinSize
            bc = np.empty((4, 2), dtype=np.float64)
            bc[0] = REta_to_YZ(R0, E0)
            bc[1] = REta_to_YZ(R0, E1)
            bc[2] = REta_to_YZ(R1, E0)
            bc[3] = REta_to_YZ(R1, E1)
            bin_corners_cache[key] = bc
        return bin_corners_cache[key]

    # Vectorised "all 4 corners in same bin" detection per pixel
    trivial = (r_lo_row == r_hi_row) & (e_lo_row == e_hi_row) & in_range_row
    if mask_row is not None:
        trivial = trivial & ~mask_row

    pix_idx_list: list = []
    bin_idx_list: list = []
    area_list:    list = []
    base = j * NY

    # Fast path: a pixel entirely inside one bin contributes its whole area to
    # that bin. That area is the pixel's MAPPED quad area, not 1.0: once the
    # corners are corrected for tilt/distortion the quad is no longer a unit
    # square, and hard-coding 1.0 here while the slow path below returns true
    # geometric areas would mix two different area conventions in one output
    # (measured: a fully-contained pixel reporting 1.000000 next to a quad of
    # 0.988316). Callers wanting unit-weight pixels get that from
    # ``area_weight="pixel"``, which divides by exactly this quantity.
    trivial_idx = np.where(trivial)[0]
    if trivial_idx.size > 0:
        pix_idx_list.extend((base + trivial_idx).tolist())
        bin_idx_list.extend((e_lo_row[trivial_idx] * n_r
                              + r_lo_row[trivial_idx]).tolist())
        area_list.extend(quad_row[trivial_idx].tolist())

    # Slow path: only the pixels that straddle a bin boundary
    boundary_mask = in_range_row & ~trivial
    if mask_row is not None:
        boundary_mask = boundary_mask & ~mask_row
    boundary_idx = np.where(boundary_mask)[0]
    for i in boundary_idx:
        pix_corners = corners_row[i]
        r_a = int(r_lo_row[i]); r_b = int(r_hi_row[i])
        e_a = int(e_lo_row[i]); e_b = int(e_hi_row[i])
        flat_idx = base + i
        for rr in range(r_a, r_b + 1):
            R0 = RMin + rr * RBinSize
            R1 = R0 + RBinSize
            for ee in range(e_a, e_b + 1):
                E0 = EtaMin + ee * EtaBinSize
                E1 = E0 + EtaBinSize
                bc = _bin_corners(rr, ee)
                a = pixel_bin_intersect(
                    pix_corners, R0, R1, E0, E1,
                    bin_corners=bc,
                )
                if a > 1e-12:
                    pix_idx_list.append(flat_idx)
                    bin_idx_list.append(ee * n_r + rr)
                    area_list.append(a)
    return (
        np.asarray(pix_idx_list, dtype=np.int64),
        np.asarray(bin_idx_list, dtype=np.int64),
        np.asarray(area_list,    dtype=np.float64),
    )


# Pixel corner offsets (Y, Z) in pixel units — the four corners of a
# unit-square pixel centred at integer (Y, Z). Same convention as v1.
#: Pixel corner offsets, taken from v1 rather than redefined here.
#
# The order MUST be the one ``pixel_bin_intersect`` and ``point_in_quad`` walk
# with ``QUAD_ORDER``; see the note on ``PIXEL_CORNER_OFFSETS`` in
# ``midas_integrate.geometry`` for why that is Z-order and what a bowtie
# traversal costs. v2 used to declare its own boundary-CCW order here, which
# was one of the two wrong ones.
_PIXEL_CORNERS = PIXEL_CORNER_OFFSETS


def _pixel_corner_YZ(spec: IntegrationSpec) -> np.ndarray:
    """Compute (Y, Z) of all 4 corners of every pixel.

    These are physical coordinates after centering on BC and converting
    to detector µm — precisely what v1's ``pixel_bin_intersect``
    expects. We use the spec's geometry forward to project into the
    same untilted coordinate frame v1 uses internally.

    Returns ``(NrPixelsZ, NrPixelsY, 4, 2)`` of floats.
    """
    NY, NZ = spec.NrPixelsY, spec.NrPixelsZ
    pxY, pxZ = spec.pxY, spec.pxZ
    BC_y = float(spec.BC_y.detach())
    BC_z = float(spec.BC_z.detach())
    Yidx, Zidx = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
    out = np.empty((NZ, NY, 4, 2), dtype=np.float64)
    for c, (dy, dz) in enumerate(_PIXEL_CORNERS):
        Y = (Yidx + dy)
        Z = (Zidx + dz)
        # Convert to v1's geometry coords: Yc, Zc in detector µm
        # (same convention as v1.geometry.pixel_to_REta).
        Yc = -(Y - BC_y) * pxY                     # µm
        Zc = (Z - BC_z) * pxZ                      # µm
        out[:, :, c, 0] = Yc
        out[:, :, c, 1] = Zc
    return out


def _pixel_corner_YZ_px(spec: IntegrationSpec) -> np.ndarray:
    """Pixel corners in the CORRECTED (Y, Z) frame, in pixel units.

    ``R = sqrt(Y² + Z²)`` on the result is directly comparable to
    ``spec.RMin``/``RMax``, and — crucially — lives in the same frame as the
    bin corners, which ``PolygonBinGeometry.from_spec`` builds with
    :func:`REta_to_YZ`. Bins are exact circular arcs in that frame, so the
    polygon-vs-arc intersection is well posed.

    Each corner is mapped through the package's shared geometry,
    :func:`pixel_to_REta_from_spec` — the same function ``HardBinGeometry``,
    ``SubpixelBinGeometry`` and ``corrections.integrated`` use — and then back
    into Cartesian with :func:`REta_to_YZ`::

        (Y_pix, Z_pix) --pixel_to_REta_from_spec--> (R_px, η) --REta_to_YZ--> (Y, Z)

    That picks up detector tilt, the distortion polynomial, parallax, per-panel
    rigid shifts and the residual-correction map in one step, because the
    shared function already applies all of them.

    This used to read only ``spec.BC_y``/``spec.BC_z`` — a bare beam-centre
    projection — so the polygon kernel silently discarded every one of those
    corrections while being documented as the exact/reference kernel
    (issue #69). Measured on a 200x200 synthetic detector, against the shared
    function at the same sub-pixel positions: ``ty = 5°`` gave max |ΔR| =
    1.4577 px, ``iso_R2 = 0.05`` gave 0.2871 px, and a ±2/±3 px two-panel
    shift gave 3.6056 px = sqrt(2² + 3²), i.e. exactly the discarded shift.
    With sub-pixel ``RBinSize`` — polygon's stated use case — those are many
    bins wide.

    Panel handling: ``panel_idx`` is looked up ONCE at each pixel's centre and
    reused for its four corners. A panel shift is a rigid transform of the
    whole pixel, and ``panel_idx_for_points`` rounds each point separately, so
    letting it run per-corner would give the corners of a boundary-straddling
    pixel different shifts and tear the quad.

    Returns ``(NrPixelsZ, NrPixelsY, 4, 2)`` of float64.
    """
    NY, NZ = spec.NrPixelsY, spec.NrPixelsZ
    dt, dev = spec.dtype(), spec.device()

    Yidx, Zidx = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
    Yc_t = torch.as_tensor(Yidx, dtype=dt, device=dev)
    Zc_t = torch.as_tensor(Zidx, dtype=dt, device=dev)

    # One panel index per PIXEL (from its centre), broadcast to its corners.
    panel_idx = None
    panels = _panel_inputs_from_spec(spec)
    if panels is not None:
        layout = panels[0]
        centre_idx = panel_idx_for_points(layout, Yc_t, Zc_t)      # (NZ, NY)
        panel_idx = centre_idx.unsqueeze(0).expand(len(_PIXEL_CORNERS), NZ, NY)

    # All four corners in a single batched call rather than four passes.
    dY = torch.as_tensor([c[0] for c in _PIXEL_CORNERS], dtype=dt, device=dev)
    dZ = torch.as_tensor([c[1] for c in _PIXEL_CORNERS], dtype=dt, device=dev)
    Y_all = Yc_t.unsqueeze(0) + dY.view(-1, 1, 1)                  # (4, NZ, NY)
    Z_all = Zc_t.unsqueeze(0) + dZ.view(-1, 1, 1)

    with torch.no_grad():
        out = pixel_to_REta_from_spec(Y_all, Z_all, spec, panel_idx=panel_idx)
        R_px = out.R_px.detach().cpu().numpy()                     # (4, NZ, NY)
        eta_deg = out.eta_deg.detach().cpu().numpy()

    Yq, Zq = REta_to_YZ(R_px, eta_deg)          # back into the bins' frame
    corners = np.empty((NZ, NY, len(_PIXEL_CORNERS), 2), dtype=np.float64)
    corners[..., 0] = np.moveaxis(Yq, 0, -1)
    corners[..., 1] = np.moveaxis(Zq, 0, -1)
    return corners


def _quad_area(corners: np.ndarray) -> np.ndarray:
    """Shoelace area of each pixel quad, ``(NZ, NY, 4, 2)`` -> ``(NZ, NY)``.

    The corners are stored in Z-order (see ``PIXEL_CORNER_OFFSETS``), which is
    NOT boundary order, so the shoelace must walk them via ``QUAD_ORDER`` --
    the same traversal ``pixel_bin_intersect`` uses. Rolling the stored order
    directly would trace a bowtie and give the wrong area. The absolute value
    then covers the orientation of the cycle, which is not meaningful here.
    """
    idx = list(QUAD_ORDER)
    Y = corners[..., idx, 0]
    Z = corners[..., idx, 1]
    Yn = np.roll(Y, -1, axis=-1)
    Zn = np.roll(Z, -1, axis=-1)
    return 0.5 * np.abs((Y * Zn - Yn * Z).sum(axis=-1))


@dataclass
class PolygonBinGeometry:
    """Exact polygon-area pixel-bin geometry.

    Each (pixel, bin) overlap becomes one entry with
    ``area = polygon-intersection area in pixel² units``
    (so a fully-contained pixel contributes ``area = 1.0``).

    Fields:
        pix_idx     : (n_entries,) long — flat pixel index (z*NY + y)
        bin_idx     : (n_entries,) long — flat (eta * n_r + r) bin index
        area        : (n_entries,) double — polygon intersection area
        n_r, n_eta  : ints
        n_pixels_y, n_pixels_z : detector dimensions
        trans_opt   : list of ImTransOpt op codes
    """
    pix_idx: torch.Tensor
    bin_idx: torch.Tensor
    area: torch.Tensor
    n_r: int
    n_eta: int
    n_pixels_y: int
    n_pixels_z: int
    trans_opt: list = None        # type: ignore[assignment]

    @classmethod
    def from_spec(cls, spec: IntegrationSpec, *,
                   verbose: bool = False,
                   n_jobs: int = 1,
                   mask: Optional[np.ndarray] = None,
                   area_weight: str = "pixel") -> "PolygonBinGeometry":
        """Build the exact polygon-area geometry for ``spec``.

        Parameters
        ----------
        spec :
            v2 :class:`IntegrationSpec`.
        verbose :
            Print progress (only honoured for ``n_jobs == 1``).
        n_jobs :
            Number of joblib workers for the per-row polygon build.
            ``1`` (default) runs serially; ``-1`` uses all available
            cores. Embarrassingly-parallel across detector rows; near-
            linear speedup up to ~8 cores on a 2.5 Mpx Pilatus build.
        mask :
            Optional 2D ``(NrPixelsZ, NrPixelsY)`` mask. Non-zero
            entries mark pixels to skip — beam stops, dead pixels,
            module gaps. v1 convention: 1.0 = masked. Masked pixels
            are dropped at build time (no per-bin contribution and no
            polygon math wasted on them).
        area_weight :
            How much weight each pixel carries once its corners are mapped
            into the corrected frame, where its quad is no longer exactly
            1 px².

            ``"pixel"`` (default) renormalises by the pixel's own mapped quad
            area, so every pixel still carries total weight 1 and the kernel
            agrees with ``HardBinGeometry`` / ``SubpixelBinGeometry`` in the
            fine-bin limit. This keeps the fix purely geometric.

            ``"projected"`` keeps the mapped area, so a pixel is weighted by
            the detector-plane area it actually subtends in the corrected
            frame (solid-angle-like). Arguably the more physical weighting,
            but it makes polygon differ from the other kernels by a smooth
            factor, so it is opt-in rather than the default.

        The build delegates to v1's :func:`pixel_bin_intersect`
        (pure-numpy Green's-theorem kernel — no numba dependency); we
        wrap it in a ``joblib.Parallel`` over rows when ``n_jobs != 1``.
        """
        if area_weight not in ("pixel", "projected"):
            raise ValueError(
                f"area_weight must be 'pixel' or 'projected', got "
                f"{area_weight!r}"
            )
        spec.validate()
        if getattr(spec, "lattice", "cartesian") != "cartesian":
            raise NotImplementedError(
                f"PolygonBinGeometry supports lattice='cartesian' only; "
                f"got {spec.lattice!r}. For hex lattices use "
                f"SubpixelBinGeometry with pixel_shape='hexagon' (opt-in "
                f"shape-aware sampling)."
            )
        NY, NZ = spec.NrPixelsY, spec.NrPixelsZ
        n_r, n_eta = spec.n_r_bins, spec.n_eta_bins
        mask_np = normalise_mask(mask, NrPixelsY=NY, NrPixelsZ=NZ)

        # Pixel corners in PX units, in the corrected frame — the same frame
        # the bin corners below are built in via REta_to_YZ.
        corners = _pixel_corner_YZ_px(spec)              # (NZ, NY, 4, 2)
        quad = _quad_area(corners)                       # (NZ, NY)

        # Per-pixel R/η bounding box in px / deg.  η is computed via
        # v1's :func:`calc_eta_angle` (== ``atan2(-Yc, Zc) · 180/π``)
        # on all corners at once — fully vectorised.
        Yc = corners[..., 0]; Zc = corners[..., 1]
        R = np.sqrt(Yc * Yc + Zc * Zc)
        Eta = np.degrees(np.arctan2(-Yc, Zc))           # (NZ, NY, 4)

        R_lo_px = R.min(axis=-1)                  # (NZ, NY)
        R_hi_px = R.max(axis=-1)
        Eta_lo = Eta.min(axis=-1)
        Eta_hi = Eta.max(axis=-1)
        # Detect wraparound (when one corner is near -180 and another near +180).
        wraparound = (Eta_hi - Eta_lo) > 180.0
        Eta_lo[wraparound] = spec.EtaMin
        Eta_hi[wraparound] = spec.EtaMax

        # Candidate bin index ranges per pixel
        r_lo = np.floor((R_lo_px - spec.RMin) / spec.RBinSize).astype(np.int64)
        r_hi = np.floor((R_hi_px - spec.RMin) / spec.RBinSize).astype(np.int64)
        e_lo = np.floor((Eta_lo - spec.EtaMin) / spec.EtaBinSize).astype(np.int64)
        e_hi = np.floor((Eta_hi - spec.EtaMin) / spec.EtaBinSize).astype(np.int64)
        r_lo = np.clip(r_lo, 0, n_r - 1)
        r_hi = np.clip(r_hi, 0, n_r - 1)
        e_lo = np.clip(e_lo, 0, n_eta - 1)
        e_hi = np.clip(e_hi, 0, n_eta - 1)

        # In-range mask: pixel touches the binned (R, η) area at all
        in_range = (R_hi_px > spec.RMin) & (R_lo_px < spec.RMax)

        # Build per-row.  Each row is independent → easy parallelism.
        per_row_args = (
            spec.RMin, spec.RBinSize,
            spec.EtaMin, spec.EtaBinSize,
            n_r, n_eta,
        )
        def _row_mask(j):
            return mask_np[j] if mask_np is not None else None

        if n_jobs == 1:
            row_results = []
            for j in range(NZ):
                row_results.append(_build_one_row(
                    j, NY, corners[j], quad[j], in_range[j],
                    r_lo[j], r_hi[j], e_lo[j], e_hi[j],
                    *per_row_args,
                    _row_mask(j),
                ))
                if verbose and (j + 1) % max(1, NZ // 10) == 0:
                    print(f"  polygon build: row {j + 1}/{NZ}")
        else:
            try:
                from joblib import Parallel, delayed
            except ImportError as e:
                raise ImportError(
                    "n_jobs != 1 requires joblib; pip install joblib"
                ) from e
            row_results = Parallel(n_jobs=n_jobs, batch_size="auto")(
                delayed(_build_one_row)(
                    j, NY, corners[j], quad[j], in_range[j],
                    r_lo[j], r_hi[j], e_lo[j], e_hi[j],
                    *per_row_args,
                    _row_mask(j),
                )
                for j in range(NZ)
            )

        pix_arrs, bin_arrs, area_arrs = zip(*row_results)
        pix_np = np.concatenate(pix_arrs) if pix_arrs else np.empty(0, np.int64)
        bin_np = np.concatenate(bin_arrs) if bin_arrs else np.empty(0, np.int64)
        area_np = np.concatenate(area_arrs) if area_arrs else np.empty(0, np.float64)

        if area_weight == "pixel" and pix_np.size:
            # Restore "each pixel carries weight 1". In the corrected frame a
            # pixel's quad is no longer exactly 1 px^2 -- tilt and distortion
            # stretch it -- and integrate_polygon normalises per BIN by the
            # summed area, so without this each pixel would silently be
            # reweighted by its own Jacobian.
            #
            # Divide by the pixel's FULL mapped quad area, not by the sum of
            # its in-range bin areas: a pixel lying partly outside
            # [RMin, RMax] must keep only the fraction that is in range, not
            # have that fraction renormalised back up to 1.
            denom = quad.reshape(-1)[pix_np]
            area_np = area_np / np.where(denom > 0.0, denom, 1.0)

        pix_idx = torch.from_numpy(pix_np).to(torch.long)
        bin_idx = torch.from_numpy(bin_np).to(torch.long)
        area    = torch.from_numpy(area_np).to(torch.float64)

        return cls(
            pix_idx=pix_idx, bin_idx=bin_idx, area=area,
            n_r=n_r, n_eta=n_eta,
            n_pixels_y=NY, n_pixels_z=NZ,
            trans_opt=list(spec.TransOpt),
        )

    @property
    def n_entries(self) -> int:
        return int(self.pix_idx.shape[0])

    @property
    def device(self) -> torch.device:
        return self.pix_idx.device


def integrate_polygon(
    image: torch.Tensor,
    geom: PolygonBinGeometry,
    *,
    apply_trans_opt: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """Integrate ``image`` against an exact polygon-area geometry.

    Returns ``(n_eta, n_r)``. Each pixel contributes
    ``image[pix] · area_pixel_in_bin`` to each bin it overlaps;
    ``normalize=True`` divides by the per-bin total area
    (``Σ area_pixel_in_bin``), giving area-weighted mean intensity per
    bin — matching v1's ``mode='floor', normalize=True`` semantics.
    """
    if image.shape != (geom.n_pixels_z, geom.n_pixels_y):
        raise ValueError(
            f"image shape {tuple(image.shape)} does not match "
            f"geometry ({geom.n_pixels_z}, {geom.n_pixels_y})"
        )
    if apply_trans_opt and geom.trans_opt and needs_trans_opt(geom.trans_opt):
        image = apply_trans_opt_forward(
            image, geom.trans_opt,
            NrPixelsY=geom.n_pixels_y, NrPixelsZ=geom.n_pixels_z,
        )
    img_flat = image.reshape(-1).to(torch.float64)
    n_bins = geom.n_eta * geom.n_r

    contrib = img_flat[geom.pix_idx] * geom.area
    sums = torch.zeros(n_bins, dtype=img_flat.dtype, device=img_flat.device)
    sums = sums.index_add(0, geom.bin_idx, contrib)

    if not normalize:
        return sums.reshape(geom.n_eta, geom.n_r)

    areas = torch.zeros(n_bins, dtype=img_flat.dtype, device=img_flat.device)
    areas = areas.index_add(0, geom.bin_idx, geom.area)
    out = sums / areas.clamp(min=1e-12)
    return out.reshape(geom.n_eta, geom.n_r)


__all__ = ["PolygonBinGeometry", "integrate_polygon"]
