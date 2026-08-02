"""Phase 1: hard, non-differentiable orientation screening.

Replicates the C ``CalcFracOverlap`` loop that survives between
``FitOrientationOMP``, ``FitOrientationParameters``, and
``FitOrientationParametersMultiPoint``. For each voxel, evaluate every
candidate orientation in ``OrientMat.bin`` and keep those whose fraction
of detector pixels matched against ``SpotsInfo.bin`` exceeds
``MinFracAccept``.

This version uses the precomputed lab-frame ``(yl, zl, omega_deg)``
spots written by ``midas_nf_preprocess`` (the
:program:`MakeDiffrSpots` replacement) into ``DiffractionSpots.bin``.
For each voxel we project those precomputed spots through each of the
three voxel-triangle vertices via the ``DisplacementSpots`` formula,
optionally apply the NF tilt correction at the primary distance, then
either rasterise the projected triangle on the detector grid (when
``2*gs > px``, matching ``CalcPixels2``) or use the rounded centroid
(``2*gs <= px`` branch, matching the inline path at
SharedFuncsFit.c:595-600). Each rasterised pixel is then looked up at
each detector distance and AND-ed across distances to produce the
``OverlapPixels / TotalPixels`` fraction the C code computes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import os

import numpy as np
import torch

from .io import GridTable, OrientationData
from .obs_volume import ObsVolume
from .params import FitParams
from .reparam import normalize_orient_mat


# ---------------------------------------------------------------------------
#  OrientMat → ZXZ Euler conversion
# ---------------------------------------------------------------------------

def orientmat_to_euler_zxz(matrices: np.ndarray) -> np.ndarray:
    """Convert ``(N, 3, 3)`` proper-rotation matrices to Bunge ZXZ Euler
    triplets (radians).

    Matches ``OrientMat2Euler`` in MIDAS C: when ``|R[2,2]| < 1`` use the
    standard formula, otherwise fall back to the gimbal-locked branch.

    Each input matrix is first passed through
    :func:`midas_nf_fitorientation.reparam.normalize_orient_mat`
    (the ``NormalizeMat`` port — scales by ``det^(-1/3)`` to remove the
    drift accumulated in ``OrientMat.bin``); the C readers do this
    every time they pull a row out of the binary, and skipping it
    leaves ``arccos`` slightly off when the on-disk determinant
    differs from 1.
    """
    R = np.asarray(matrices)
    if R.ndim == 2:
        R = R[None, ...]
    R = normalize_orient_mat(R)
    N = R.shape[0]
    out = np.zeros((N, 3), dtype=np.float64)
    cosPhi = R[:, 2, 2].clip(-1.0, 1.0)
    Phi = np.arccos(cosPhi)
    sinPhi = np.sqrt(1.0 - cosPhi * cosPhi)
    locked = sinPhi < 1e-9
    nl = ~locked
    out[nl, 1] = Phi[nl]
    out[nl, 0] = np.arctan2(R[nl, 0, 2], -R[nl, 1, 2])
    out[nl, 2] = np.arctan2(R[nl, 2, 0],  R[nl, 2, 1])
    out[locked, 1] = Phi[locked]
    out[locked, 0] = np.arctan2(R[locked, 1, 0], R[locked, 0, 0])
    out[locked, 2] = 0.0
    return out


# ---------------------------------------------------------------------------
#  Tilt rotation matrix (RotationTilts in SharedFuncsFit.c)
# ---------------------------------------------------------------------------

def build_rot_tilts(
    tx_deg: float, ty_deg: float, tz_deg: float,
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Construct the 3×3 NF tilt rotation matrix ``Rz(tz) @ Ry(ty) @ Rx(tx)``.

    Direct port of ``RotationTilts`` from
    ``NF_HEDM/src/SharedFuncsFit.c:230-266``.
    """
    d2r = math.pi / 180.0
    tx, ty, tz = tx_deg * d2r, ty_deg * d2r, tz_deg * d2r
    cx, sx = math.cos(tx), math.sin(tx)
    cy, sy = math.cos(ty), math.sin(ty)
    cz, sz = math.cos(tz), math.sin(tz)
    Rx = torch.tensor(
        [[1, 0, 0], [0, cx, -sx], [0, sx, cx]],
        dtype=dtype, device=device,
    )
    Ry = torch.tensor(
        [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]],
        dtype=dtype, device=device,
    )
    Rz = torch.tensor(
        [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]],
        dtype=dtype, device=device,
    )
    return Rz @ Ry @ Rx


def apply_nf_tilt(
    ydet: torch.Tensor, zdet: torch.Tensor,
    Lsd_val: float | torch.Tensor,
    R: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ray-plane intersection that applies a tilted-detector correction.

    Direct port of the inline block in
    ``NF_HEDM/src/SharedFuncsFit.c:947-958`` (also present as
    ``midas_diffract.HEDMForwardModel._apply_nf_tilt``). Reduces to the
    identity when ``R == I``.

    Both ``ydet`` and ``zdet`` are tensors of arbitrary shape;
    everything broadcasts.
    """
    if not torch.is_tensor(Lsd_val):
        Lsd_val = torch.tensor(float(Lsd_val), dtype=ydet.dtype, device=ydet.device)
    p0x = -Lsd_val * R[0, 0]
    p0y = -Lsd_val * R[1, 0]
    p0z = -Lsd_val * R[2, 0]
    P1x = ydet * R[0, 1] + zdet * R[0, 2]
    P1y = ydet * R[1, 1] + zdet * R[1, 2]
    P1z = ydet * R[2, 1] + zdet * R[2, 2]
    ABCx = P1x - p0x
    ABCy = P1y - p0y
    ABCz = P1z - p0z
    safe = torch.where(
        torch.abs(ABCx) < eps,
        torch.full_like(ABCx, eps),
        ABCx,
    )
    out_y = p0y - ABCy * p0x / safe
    out_z = p0z - ABCz * p0x / safe
    return out_y, out_z


# ---------------------------------------------------------------------------
#  Result containers
# ---------------------------------------------------------------------------

@dataclass
class Winner:
    voxel_idx: int
    orient_idx: int
    frac_overlap: float


@dataclass
class ScreenResult:
    """Per-voxel screening outcome.

    Attributes
    ----------
    winners : list[Winner]
        All ``(voxel, orient, frac)`` tuples whose ``frac`` ≥
        ``MinFracAccept``. Sorted by ``(voxel_idx, orient_idx)`` so it
        matches the C ``screen_cpu.csv`` diagnostic dump column-for-column.
    n_winners_per_voxel : np.ndarray (V,) int64
        Convenience counter for downstream allocation.
    """
    winners: List[Winner]
    n_winners_per_voxel: np.ndarray


# ---------------------------------------------------------------------------
#  Vectorised triangle rasteriser (port of CalcPixels2)
# ---------------------------------------------------------------------------

def rasterize_triangles(
    rel_y: torch.Tensor, rel_z: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorised rasteriser for ``T`` triangles in integer pixel space.

    Direct port of the half-plane + 0.99-pixel-distance test in
    ``CalcPixels2`` (``NF_HEDM/src/SharedFuncsFit.c:308-370``):

    - vertices are first ``round()``-snapped to integers (line 312-313);
    - a pixel is "inside" if all three signed-area orient2d products
      are non-negative;
    - if it isn't strictly inside, the pixel is still kept when its
      perpendicular squared distance to *any* of the three edges is
      below 0.9801 (= 0.99²).

    Parameters
    ----------
    rel_y, rel_z : Tensor (T, 3) int or float
        The three vertex pixel coordinates per triangle. Both inputs
        live in the same coordinate frame; the rasteriser returns its
        output offsets in that frame too.

    Returns
    -------
    abs_y, abs_z : Tensor (T, K) int64
        Per-triangle pixel coordinates over the bounding-box union.
        The same ``(P, Q)`` grid is used for every triangle so the
        leading-T tensors line up.
    valid : Tensor (T, K) bool
        Per-triangle in-triangle mask. Pixels failing both the
        half-plane test and the 0.99-edge-distance test are False.
    """
    if rel_y.shape != rel_z.shape:
        raise ValueError(
            f"rel_y / rel_z shape mismatch: {rel_y.shape} vs {rel_z.shape}"
        )
    if rel_y.ndim != 2 or rel_y.shape[1] != 3:
        raise ValueError(f"expected (T, 3), got {rel_y.shape}")

    device = rel_y.device
    rel_y = rel_y.round().long()
    rel_z = rel_z.round().long()

    min_y = rel_y.min(dim=1).values         # (T,)
    max_y = rel_y.max(dim=1).values
    min_z = rel_z.min(dim=1).values
    max_z = rel_z.max(dim=1).values

    P = int((max_y - min_y).max().item()) + 1
    Q = int((max_z - min_z).max().item()) + 1

    if P <= 0 or Q <= 0:
        # Degenerate (single-point triangles) — fall through with one
        # pixel per triangle.
        P = max(P, 1)
        Q = max(Q, 1)

    # Build the (T, P, Q) absolute pixel grid, anchored at each
    # triangle's bbox min corner.
    ys_grid = torch.arange(P, device=device, dtype=torch.long).reshape(1, P, 1)
    zs_grid = torch.arange(Q, device=device, dtype=torch.long).reshape(1, 1, Q)
    abs_y = min_y.reshape(-1, 1, 1) + ys_grid          # (T, P, Q)
    abs_z = min_z.reshape(-1, 1, 1) + zs_grid          # (T, P, Q)

    # Vertex tensors (T, 2) — broadcast against (T, P, Q) via reshape.
    v0y = rel_y[:, 0:1].unsqueeze(2)
    v0z = rel_z[:, 0:1].unsqueeze(2)
    v1y = rel_y[:, 1:2].unsqueeze(2)
    v1z = rel_z[:, 1:2].unsqueeze(2)
    v2y = rel_y[:, 2:3].unsqueeze(2)
    v2z = rel_z[:, 2:3].unsqueeze(2)

    # orient2d(a, b, p) = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)
    # using (y, z) as (x, y) per the C convention in CalcPixels2.
    def orient2d(ay, az, by, bz, py, pz):
        return (by - ay) * (pz - az) - (bz - az) * (py - ay)

    w0 = orient2d(v1y, v1z, v2y, v2z, abs_y, abs_z)
    w1 = orient2d(v2y, v2z, v0y, v0z, abs_y, abs_z)
    w2 = orient2d(v0y, v0z, v1y, v1z, abs_y, abs_z)
    inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)

    # 0.99-pixel edge tolerance: dist² < 0.9801.
    # distSq2d(a, b, c) = orient2d² / |ab|²  (perpendicular distance²
    # from c to line ab); see SharedFuncsFit.c:302-306.
    def dist_sq(ay, az, by, bz, w):
        # w is the precomputed orient2d(a, b, .). |ab|² = (a.y-b.y)²+(b.x-a.x)²
        # in the C (y, z) convention.
        denom = (az - bz) ** 2 + (by - ay) ** 2
        denom_f = denom.to(torch.float64).clamp(min=1.0)  # avoid div-by-0
        return (w.to(torch.float64) ** 2) / denom_f

    near_edge = (
        (dist_sq(v1y, v1z, v2y, v2z, w0) < 0.9801)
        | (dist_sq(v2y, v2z, v0y, v0z, w1) < 0.9801)
        | (dist_sq(v0y, v0z, v1y, v1z, w2) < 0.9801)
    )
    # Clip each triangle's mask to its own bbox: the C rasteriser
    # iterates only over a triangle's own [min_y..max_y] × [min_z..max_z],
    # so an edge-tolerance pixel belonging to a *different* triangle in
    # the batch must not bleed in here. Without this, batches of
    # triangles with different bbox sizes produce false-positive pixels
    # near the union-bbox boundary.
    own_bbox = (
        (abs_y >= min_y.reshape(-1, 1, 1))
        & (abs_y <= max_y.reshape(-1, 1, 1))
        & (abs_z >= min_z.reshape(-1, 1, 1))
        & (abs_z <= max_z.reshape(-1, 1, 1))
    )
    valid = (inside | near_edge) & own_bbox             # (T, P, Q)

    K = P * Q
    return (
        abs_y.expand(rel_y.shape[0], P, Q).reshape(-1, K),
        abs_z.expand(rel_y.shape[0], P, Q).reshape(-1, K),
        valid.reshape(-1, K),
    )


# ---------------------------------------------------------------------------
#  Screen kernel (DiffractionSpots.bin path)
# ---------------------------------------------------------------------------

def spot_weights_from_f2(
    orientations: "OrientationData",
    hkl,
    lsd0: float,
    *,
    metric: str = "filtered",
    eps: float = 1e-6,
    tol_um: float = 1.0,
) -> Optional[torch.Tensor]:
    """Per-spot weights for :func:`screen`, joined on RING RADIUS.

    ``DiffractionSpots.bin`` rows are ``(yl, zl, omega)`` -- lab-frame microns
    then degrees -- and carry no reflection index, so the join is on
    ``r = hypot(yl, zl)``, which is the ring, and |F|^2 is a ring-level property
    (symmetry equivalents share it).

    ``lsd0`` is ``Lsd[0]``: the spot coordinates are the nominal ring position
    at the FIRST distance, whereas the ``Radius`` column of ``hkls.csv`` is
    computed at the LAST one. Joining against that column directly is wrong by
    the ratio of the two (1.2769 on the Ce setup: spot radii 159.9-1092.1 um
    against a 204.1-1394.5 um column). Ring radii are therefore recomputed here
    from ``2*thetas_deg`` at ``lsd0`` rather than read from the file.

    An earlier version joined on 2-theta, because ``OrientationData``'s
    docstring claimed the columns were ``[2theta, eta, omega]``. They are not.
    That join matched ZERO rows and returned an all-ones weight -- a silent
    no-op that reproduced the unweighted result exactly, which is
    indistinguishable from "the weighting made no difference" unless you
    predicted the weighted number in advance.

    metric
        ``"raw"``      -> None (caller skips weighting; unchanged code path)
        ``"filtered"`` -> 1.0 for allowed reflections, 0.0 for |F|^2 <= eps
        ``"weighted"`` -> |F|^2 itself

    ``"filtered"`` is usually better done with ``DropForbiddenReflections``,
    which removes the rows from ``hkls.csv`` so they are never rasterised at
    all. This path is for a spot file generated with the full list.

    No Lorentz factor, deliberately -- it cancels for per-frame threshold
    detection (Domega = w_rlp*L, so peak height is L-free; measured flat in eta
    on nf_Ce_ht525_s2 where 1/|sin eta| predicts 4.3x).
    """
    if metric == "raw":
        return None
    if metric not in ("filtered", "weighted"):
        raise ValueError(f"metric must be raw|filtered|weighted, got {metric!r}")
    f2 = getattr(hkl, "f2", None)
    if f2 is None:
        return None
    f2 = np.asarray(f2)
    if f2.size == 0 or np.allclose(f2, 1.0):
        return None                      # no basis was supplied; nothing to do

    # ring radius at Lsd[0], the frame the spot coordinates live in
    r_ring = float(lsd0) * np.tan(np.radians(np.asarray(hkl.thetas_deg) * 2.0))
    order = np.argsort(r_ring)
    r_s, f2_s = r_ring[order], f2[order]

    r_spot = np.hypot(np.asarray(orientations.spots[:, 0], dtype=np.float64),
                      np.asarray(orientations.spots[:, 1], dtype=np.float64))
    j = np.clip(np.searchsorted(r_s, r_spot), 0, r_s.size - 1)
    j2 = np.clip(j - 1, 0, r_s.size - 1)
    pick = np.where(np.abs(r_s[j] - r_spot) <= np.abs(r_s[j2] - r_spot), j, j2)
    matched = np.abs(r_s[pick] - r_spot) <= tol_um

    if not matched.any():
        raise ValueError(
            "spot_weights_from_f2 matched no spots to a ring "
            f"(spot radii {r_spot.min():.1f}-{r_spot.max():.1f} um, rings "
            f"{r_ring.min():.1f}-{r_ring.max():.1f} um). Refusing to return a "
            "neutral weight, which would silently look like 'no effect'."
        )

    w = np.where(matched, f2_s[pick], 1.0)
    if metric == "filtered":
        w = (w > eps).astype(np.float64)
    return torch.from_numpy(w)


@torch.no_grad()
def screen(
    grid: GridTable,
    orientations: OrientationData,
    obs: ObsVolume,
    p: FitParams,
    *,
    spot_weight: Optional[torch.Tensor] = None,
    voxel_indices: Optional[np.ndarray] = None,
    progress: Optional[callable] = None,
    dtype: torch.dtype = torch.float64,
) -> ScreenResult:
    """Screen every (voxel, orientation) pair against the obs bitmap.

    Uses the precomputed ``DiffractionSpots.bin`` rows
    ``(yl, zl, omega_deg)`` produced by :program:`midas-nf-preprocess`
    (the :program:`MakeDiffrSpots` replacement). Per voxel we project
    those rows through the voxel position via the ``DisplacementSpots``
    formula, scale to each detector distance, apply the NF tilt
    correction if any tilts are non-zero, look up the ``ObsVolume``
    bitmap, AND-product across distances, and aggregate hits per
    orientation via ``scatter_add``.

    Parameters
    ----------
    grid : GridTable
        Voxel positions (centroids) from ``grid.txt``.
    orientations : OrientationData
        Bundle of ``OrientMat.bin`` + ``Key.bin`` + ``DiffractionSpots.bin``.
    obs : ObsVolume
        Decoded ``SpotsInfo.bin`` bitmap, on ``device``.
    p : FitParams
        Reads ``min_frac_accept``, per-distance ``Lsd``/``ybc``/``zbc``,
        ``tx/ty/tz``, ``px``, ``omega_start``, ``omega_step``,
        ``n_pixels_y``/``n_pixels_z``, and ``n_frames_per_distance``.
    voxel_indices : np.ndarray, optional
        Subset of voxel indices to screen (used by the block-decomposition
        in ``FitOrientationOMP``). ``None`` ⇒ all voxels.
    progress : callable, optional
        ``progress(done, total)`` callback after each voxel.
    dtype : torch.dtype
        Working precision for the projection arithmetic. Default
        ``float64`` matches the C path; ``float32`` is fine for
        screening on GPU.

    Returns
    -------
    :class:`ScreenResult`
    """
    device = obs.device

    # spot_weight arrives from numpy on the CPU; everything it multiplies lives
    # on the compute device, so move it once here rather than at each use.
    if spot_weight is not None:
        spot_weight = spot_weight.to(device=device)

    if voxel_indices is None:
        voxel_indices = np.arange(grid.n_voxels)

    # ---- ship orientation data to device (one shot) ----
    n_orient = orientations.n_orientations
    n_spots_t = torch.from_numpy(
        orientations.n_spots.astype(np.int64)
    ).to(device)
    starts_t = torch.from_numpy(
        orientations.starts.astype(np.int64)
    ).to(device)
    spots_t = torch.from_numpy(
        np.asarray(orientations.spots, dtype=np.float64)
    ).to(device=device, dtype=dtype)
    yl_0 = spots_t[:, 0]
    zl_0 = spots_t[:, 1]
    omega_deg = spots_t[:, 2]
    omega_rad = omega_deg * (math.pi / 180.0)
    cos_w = torch.cos(omega_rad)
    sin_w = torch.sin(omega_rad)
    T = spots_t.shape[0]

    # ---- per-spot orientation index for scatter_add ----
    # spot_to_orient[t] = orientation owning spot t. Build by repeating
    # arange(N) by n_spots[i], using torch.repeat_interleave.
    spot_to_orient = torch.repeat_interleave(
        torch.arange(n_orient, device=device, dtype=torch.int64),
        n_spots_t,
    )
    if spot_to_orient.numel() != T:
        raise ValueError(
            f"Inconsistent: spot_to_orient has {spot_to_orient.numel()} entries "
            f"but DiffractionSpots.bin has {T} rows; check Key.bin counts."
        )

    # ---- per-distance geometry tensors ----
    Lsd = torch.tensor(p.Lsd, device=device, dtype=dtype)        # (D,)
    ybc = torch.tensor(p.ybc, device=device, dtype=dtype)        # (D,)
    zbc = torch.tensor(p.zbc, device=device, dtype=dtype)        # (D,)
    Lsd_0 = Lsd[0]
    px = float(p.px)
    n_y = int(p.n_pixels_y)
    n_z = int(p.n_pixels_z)
    D = int(p.n_distances)
    n_frames = int(p.n_frames_per_distance)

    has_tilts = bool(p.tx != 0 or p.ty != 0 or p.tz != 0)
    R_tilt = build_rot_tilts(p.tx, p.ty, p.tz, device, dtype) if has_tilts else None

    # ---- per-spot frame index (orientation-independent of voxel) ----
    frame_idx = (
        (omega_deg - p.omega_start) / p.omega_step
    ).long()
    frame_in_range = (frame_idx >= 0) & (frame_idx < n_frames)
    frame_clamped = frame_idx.clamp(0, n_frames - 1)

    # ---- vectorised across voxels ----
    # Replaces the per-voxel Python loop. All V voxels go through the
    # projection / rasterisation / multi-distance lookup / scatter-add
    # together, leaving a single (V, n_orient) frac tensor at the end.
    # On H100 with V≈3.5k, T≈1k this drops the screen step from ~10s to
    # well under a second versus the old per-voxel loop.
    winners: List[Winner] = []
    counts = np.zeros(grid.n_voxels, dtype=np.int64)

    voxel_indices_arr = np.asarray(voxel_indices, dtype=np.int64)
    V = int(voxel_indices_arr.shape[0])
    if V == 0:
        return ScreenResult(winners=winners, n_winners_per_voxel=counts)

    # Per-voxel super-pixel decision.  All voxels in the batch must
    # pick the same path (the centroid vs. rasterise branch is decided
    # by the gs/px ratio).  In every NF dataset we've seen gs is
    # uniform across voxels, so this just collapses to one bool.  If
    # we ever encounter a mixed grid we fall back to the legacy
    # per-voxel path below.
    gs_arr = grid.gs[voxel_indices_arr]
    super_pixel_mask = (2.0 * gs_arr) > px
    all_super = bool(super_pixel_mask.all())
    none_super = bool((~super_pixel_mask).all())

    if not (all_super or none_super):
        # Mixed gs across voxels — fall back to the per-voxel loop.
        return _screen_per_voxel(
            grid=grid, voxel_indices=voxel_indices_arr,
            cos_w=cos_w, sin_w=sin_w, yl_0=yl_0, zl_0=zl_0,
            Lsd=Lsd, Lsd_0=Lsd_0, ybc=ybc, zbc=zbc,
            R_tilt=R_tilt, has_tilts=has_tilts, px=px,
            n_y=n_y, n_z=n_z, D=D, T=T,
            frame_clamped=frame_clamped, frame_in_range=frame_in_range,
            spot_to_orient=spot_to_orient, n_orient=n_orient,
            obs=obs, p=p, device=device, dtype=dtype,
            counts=counts, winners=winners,
        )

    # ---- voxel chunking ------------------------------------------------
    # The block below builds (V, T, 3) tensors. T is the TOTAL simulated-spot
    # count over every candidate orientation (~3e7 for a cubic seed list), so
    # one voxel already costs T*3*itemsize and the full grid is terabytes.
    # Process the voxels in chunks sized to the free device memory. Chunks are
    # contiguous slices of ``voxel_indices_arr``, so appending winners per
    # chunk preserves the global (voxel, orient) ordering the C diagnostic
    # expects.
    _k_est = _estimate_k_pixels(grid, voxel_indices_arr, px) if all_super else 1
    chunk, t_tile = _auto_chunks(
        T, V, device=device, dtype=dtype, k_pixels=_k_est,
    )
    for _c0 in range(0, V, chunk):
        _c1 = min(_c0 + chunk, V)
        vidx_chunk = voxel_indices_arr[_c0:_c1]
        Vc = _c1 - _c0
        if progress is not None:
            progress(_c0, V)

        # ---- one-shot HtoD of voxel triangle vertices ----
        # ``triangle_vertices`` returns a (3, 2) array; stack to (Vc, 3, 2).
        verts_np = np.empty((Vc, 3, 2), dtype=np.float64)
        for k, vi in enumerate(vidx_chunk):
            verts_np[k] = grid.triangle_vertices(int(vi))
        XG_all = torch.from_numpy(verts_np[:, :, 0].copy()).to(
            device=device, dtype=dtype,
        )                                                         # (Vc, 3)
        YG_all = torch.from_numpy(verts_np[:, :, 1].copy()).to(
            device=device, dtype=dtype,
        )

        # ---- accumulators over the whole spot axis ----
        # Allocated per voxel-chunk and filled by the spot-tile loop below.
        # int32 when unweighted (counts, and the fastest path); float when a
        # per-spot |F|^2 weight is supplied, since the sums stop being counts.
        _acc_dt = torch.int32 if spot_weight is None else dtype
        hits_per_orient = torch.zeros((Vc, n_orient), device=device,
                                      dtype=_acc_dt)
        total_per_orient = torch.zeros((Vc, n_orient), device=device,
                                       dtype=_acc_dt)

        # ---- spot (T) tiling --------------------------------------------
        # The block below materialises (Vc, Tt, K) tensors. Tiling ONLY over
        # voxels is not enough: K scales with triangle area, so a correctly
        # tiled 10 um grid (K~80) needs ~94 GiB for a SINGLE voxel at
        # T = 4.7e7 -- more than any card here. The C streams over candidate
        # spots instead of materialising them, and this loop does the same.
        #
        # The reduction is a scatter_add over spot_to_orient, which is additive,
        # so accumulating tile by tile is exactly equivalent to one shot.
        for _t0 in range(0, T, t_tile):
            _t1 = min(_t0 + t_tile, T)
            Tt = _t1 - _t0

            cos_w_t = cos_w[_t0:_t1]
            sin_w_t = sin_w[_t0:_t1]
            yl_0_t = yl_0[_t0:_t1]
            zl_0_t = zl_0[_t0:_t1]
            frame_in_range_t = frame_in_range[_t0:_t1]
            frame_clamped_t = frame_clamped[_t0:_t1]
            spot_to_orient_t = spot_to_orient[_t0:_t1]

            # ---- (Vc, Tt, 3) projection ----
            # DisplacementSpots (SharedFuncsFit.c:269-292), broadcast over the
            # voxel axis.  cos/sin are spot-only (1, Tt, 1); XG/YG are
            # voxel-only (Vc, 1, 3); the result is (Vc, Tt, 3).
            cos_w_ = cos_w_t.reshape(1, Tt, 1)
            sin_w_ = sin_w_t.reshape(1, Tt, 1)
            XG_ = XG_all.reshape(Vc, 1, 3)
            YG_ = YG_all.reshape(Vc, 1, 3)
            xa = XG_ * cos_w_ - YG_ * sin_w_                       # (Vc, Tt, 3)
            ya = XG_ * sin_w_ + YG_ * cos_w_
            t_0 = 1.0 - xa / Lsd_0
            yl_0_ = yl_0_t.reshape(1, Tt, 1)
            zl_0_ = zl_0_t.reshape(1, Tt, 1)
            dy_v = ya + yl_0_ * t_0
            dz_v = zl_0_ * t_0
            if has_tilts:
                dy_v, dz_v = apply_nf_tilt(dy_v, dz_v, Lsd_0, R_tilt)
            v_y_px = dy_v / px + ybc[0]                            # (Vc, Tt, 3)
            v_z_px = dz_v / px + zbc[0]
            del xa, ya, t_0, dy_v, dz_v

            # ---- spot centres at primary distance (voxel-independent) ----
            cy_lab = yl_0_t
            cz_lab = zl_0_t
            if has_tilts:
                cy_lab, cz_lab = apply_nf_tilt(cy_lab, cz_lab, Lsd_0, R_tilt)
            c_y_px = cy_lab / px + ybc[0]                          # (Tt,)
            c_z_px = cz_lab / px + zbc[0]

            # Vertex-on-detector + frame-in-range gate. (Vc, Tt)
            vert_in_bounds = (
                (v_y_px >= 0) & (v_y_px < n_y)
                & (v_z_px >= 0) & (v_z_px < n_z)
            ).all(dim=2) & frame_in_range_t.unsqueeze(0)

            rel_y = v_y_px - c_y_px.reshape(1, Tt, 1)              # (Vc, Tt, 3)
            rel_z = v_z_px - c_z_px.reshape(1, Tt, 1)
            del v_y_px, v_z_px

            if all_super:
                # CalcPixels2 path: rasterise Vc*Tt triangles in one shot.
                offsets_y_flat, offsets_z_flat, valid_flat = rasterize_triangles(
                    rel_y.reshape(Vc * Tt, 3), rel_z.reshape(Vc * Tt, 3),
                )
                K = offsets_y_flat.shape[1]
                offsets_y = offsets_y_flat.reshape(Vc, Tt, K)      # (Vc, Tt, K)
                offsets_z = offsets_z_flat.reshape(Vc, Tt, K)
                valid_mask = (valid_flat.reshape(Vc, Tt, K)
                              & vert_in_bounds.unsqueeze(-1))
            else:
                # Single rounded centroid (SharedFuncsFit.c:595-600).
                cent_y = ((rel_y[..., 0] + rel_y[..., 1] + rel_y[..., 2]) / 3.0
                          ).round().long()                         # (Vc, Tt)
                cent_z = ((rel_z[..., 0] + rel_z[..., 1] + rel_z[..., 2]) / 3.0
                          ).round().long()
                offsets_y = cent_y.unsqueeze(-1)                   # (Vc, Tt, 1)
                offsets_z = cent_z.unsqueeze(-1)
                valid_mask = vert_in_bounds.unsqueeze(-1)
                K = 1
            del rel_y, rel_z

            # ---- per-distance, per-pixel multi-distance lookup ----
            # These are 0/1 occupancy flags, not physical quantities: keep them
            # as bool and let the per-spot reduction produce integer counts.
            hits_per_pixel = torch.ones((Vc, Tt, K), device=device,
                                        dtype=torch.bool)
            bounds_per_pixel = valid_mask.clone()
            for d in range(D):
                scale = (Lsd[d] / Lsd_0)
                cy_d = (c_y_px - ybc[0]) * scale + ybc[d]          # (Tt,)
                cz_d = (c_z_px - zbc[0]) * scale + zbc[d]
                abs_y = cy_d.floor().long().reshape(1, Tt, 1) + offsets_y
                abs_z = cz_d.floor().long().reshape(1, Tt, 1) + offsets_z

                in_bd = (
                    (abs_y >= 0) & (abs_y < n_y)
                    & (abs_z >= 0) & (abs_z < n_z)
                )
                bounds_per_pixel = bounds_per_pixel & in_bd

                ay = abs_y.clamp(0, n_y - 1)
                az = abs_z.clamp(0, n_z - 1)
                f_idx = frame_clamped_t.reshape(1, Tt, 1).expand_as(ay)
                d_idx = torch.full_like(f_idx, d)
                hit = obs.lookup(d_idx, f_idx, ay, az)             # uint8 0/1
                hits_per_pixel = hits_per_pixel & hit.bool()
                del abs_y, abs_z, in_bd, ay, az, f_idx, d_idx, hit

            hits_per_pixel = hits_per_pixel & bounds_per_pixel
            # Counts, not physical quantities -> int32. K is bounded by the
            # rasteriser's bounding box, so int32 cannot overflow.
            total_per_spot = bounds_per_pixel.sum(dim=2, dtype=torch.int32)
            hits_per_spot = hits_per_pixel.sum(dim=2, dtype=torch.int32)
            del hits_per_pixel, bounds_per_pixel, offsets_y, offsets_z, valid_mask

            # ---- optional |F|^2 weighting ----------------------------------
            # The fraction is over PIXELS, so weighting a reflection means
            # scaling the pixel counts of every spot belonging to it. Applied
            # to numerator AND denominator, so an all-equal weight is a no-op.
            if spot_weight is not None:
                w_t = spot_weight[_t0:_t1].to(dtype).reshape(1, Tt)
                hits_per_spot = hits_per_spot.to(dtype) * w_t
                total_per_spot = total_per_spot.to(dtype) * w_t

            # ---- scatter this tile's spot totals into (Vc, n_orient) ----
            spot_to_orient_b = spot_to_orient_t.unsqueeze(0).expand(Vc, Tt)
            hits_per_orient.scatter_add_(1, spot_to_orient_b, hits_per_spot)
            total_per_orient.scatter_add_(1, spot_to_orient_b, total_per_spot)
            del hits_per_spot, total_per_spot, spot_to_orient_b
        # Only now go to float, for the ratio itself.
        frac = hits_per_orient.to(dtype) / total_per_orient.to(dtype).clamp(min=1.0)

        # ---- collect winners across all (voxel, orient) pairs ----
        keep = frac >= p.min_frac_accept
        keep_pairs = torch.nonzero(keep, as_tuple=False)          # (W, 2)
        if keep_pairs.numel() > 0:
            keep_frac_t = frac[keep_pairs[:, 0], keep_pairs[:, 1]]
            keep_v_local = keep_pairs[:, 0].cpu().numpy()
            keep_o = keep_pairs[:, 1].cpu().numpy().astype(np.int64)
            keep_v = vidx_chunk[keep_v_local]
            keep_f = keep_frac_t.cpu().numpy()
            # Sort by (voxel_idx, orient_idx) so winners list ordering
            # matches the legacy per-voxel iteration (and the C
            # screen_cpu.csv diagnostic) byte-for-byte.
            order = np.lexsort((keep_o, keep_v))
            for i in order:
                v_int = int(keep_v[i])
                winners.append(Winner(v_int, int(keep_o[i]), float(keep_f[i])))
                counts[v_int] += 1

    if progress is not None:
        progress(V, V)

    return ScreenResult(winners=winners, n_winners_per_voxel=counts)


#: Peak number of simultaneously-live (V, T, 3)-shaped temporaries in the
#: vectorised block. The projection itself holds ~5 (xa, ya, t_0, dy_v, dz_v)
#: and ``apply_nf_tilt`` adds ~10 more (p0*, P1*, ABC*, safe) on top of the
#: rel_*/v_*_px pairs, so a 5x estimate under-counts by ~3x and OOMs inside
#: the tilt correction.
_SCREEN_LIVE_TENSORS = 16


def _auto_chunks(
    T: int, V: int, *, device, dtype, k_pixels: int = 1, safety: float = 0.40,
) -> tuple[int, int]:
    """Pick ``(voxels_per_chunk, spots_per_tile)`` so the working set fits.

    The block they bound holds ``(Vc, Tt, 3)`` projection tensors and
    ``(Vc, Tt, K)`` per-pixel tensors, so the cost is proportional to the
    product ``Vc * Tt``. Solve for that product once, then split it: take as
    many whole voxels as fit, and give the spot tile whatever is left.

    Tiling the spot axis is what makes a correctly-tiled coarse grid possible
    at all -- see the note in ``screen``. Override either axis with
    ``MIDAS_NF_SCREEN_VOXEL_CHUNK`` / ``MIDAS_NF_SCREEN_SPOT_TILE``.
    """
    itemsize = torch.empty((), dtype=dtype).element_size()
    k = max(1, int(k_pixels))
    per_pair = _SCREEN_LIVE_TENSORS * 3 * itemsize + k * 24
    dev = torch.device(device)
    if dev.type == "cuda":
        free, _total = torch.cuda.mem_get_info(dev)
    else:
        free = 8 << 30
    max_pairs = max(1, int((free * safety) // max(per_pair, 1)))

    env_v = os.environ.get("MIDAS_NF_SCREEN_VOXEL_CHUNK")
    env_t = os.environ.get("MIDAS_NF_SCREEN_SPOT_TILE")
    v_chunk = int(env_v) if env_v else max(1, min(V, max_pairs // max(T, 1)))
    v_chunk = max(1, v_chunk)
    t_tile = int(env_t) if env_t else max(1, min(T, max_pairs // v_chunk))
    return v_chunk, max(1, t_tile)


def _auto_voxel_chunk(
    T: int, *, device, dtype, k_pixels: int = 1, safety: float = 0.40,
) -> int:
    """Voxels per chunk such that the working set fits in memory.

    Two families of tensors live simultaneously inside the chunk body:

    * ``(Vc, T, 3)`` -- the projection and the three triangle vertices
      (xa/ya/t_0/dy_v/dz_v, rel_*, v_*_px, plus apply_nf_tilt's temporaries).
    * ``(Vc, T, K)`` -- the per-PIXEL rasterisation, where ``K`` is the number
      of pixels a projected triangle covers: ``offsets_y``/``offsets_z``
      (int64), ``valid_mask``/``hits_per_pixel``/``bounds_per_pixel`` (bool).

    ``K`` scales with triangle AREA, so it is ~1 for a sub-pixel triangle and
    ~(edge/px)^2 for a large one. Sizing on the vertex term alone (the old
    behaviour) silently under-counts by that factor: raising ``EdgeLength``
    from 1 um to 10 um at px=1.48 takes K from ~1 to ~80 and the chunk that
    used to fit then OOMs inside ``orient2d``.

    Override with ``MIDAS_NF_SCREEN_VOXEL_CHUNK`` if the heuristic is wrong for
    a particular grid/seed-list combination.
    """
    env = os.environ.get("MIDAS_NF_SCREEN_VOXEL_CHUNK")
    if env:
        return max(1, int(env))
    itemsize = torch.empty((), dtype=dtype).element_size()
    k = max(1, int(k_pixels))
    # vertex-shaped working set + per-pixel working set
    # (2 int64 offset maps @8B + 3 bool masks @1B = 19 B per pixel; round to 24
    #  for the transient copies torch makes during reshape/expand).
    per_voxel = T * (_SCREEN_LIVE_TENSORS * 3 * itemsize + k * 24)
    dev = torch.device(device)
    if dev.type == "cuda":
        free, _total = torch.cuda.mem_get_info(dev)
    else:
        free = 8 << 30  # assume 8 GiB of host budget on CPU
    n = int((free * safety) // max(per_voxel, 1))
    return max(1, n)


def _estimate_k_pixels(grid, voxel_indices_arr, px: float) -> int:
    """Upper bound on pixels covered by one projected voxel triangle.

    Measured from the grid's own geometry rather than assumed: take a real
    triangle, get its bounding box in um, convert to pixels and pad. NF is
    effectively a 1:1 projection for a small triangle, so the detector-plane
    footprint is the sample-plane footprint.
    """
    try:
        verts = np.asarray(grid.triangle_vertices(int(voxel_indices_arr[0])))
    except Exception:
        return 1
    if verts.size == 0:
        return 1
    span_um = max(
        float(verts[:, 0].max() - verts[:, 0].min()),
        float(verts[:, 1].max() - verts[:, 1].min()),
    )
    side_px = int(math.ceil(span_um / max(px, 1e-9))) + 2
    return max(1, side_px * side_px)


# ---------------------------------------------------------------------------
#  Per-voxel fallback (mixed-gs grids)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _screen_per_voxel(
    *,
    grid: GridTable,
    voxel_indices: np.ndarray,
    cos_w: torch.Tensor, sin_w: torch.Tensor,
    yl_0: torch.Tensor, zl_0: torch.Tensor,
    Lsd: torch.Tensor, Lsd_0: torch.Tensor,
    ybc: torch.Tensor, zbc: torch.Tensor,
    R_tilt: Optional[torch.Tensor], has_tilts: bool, px: float,
    n_y: int, n_z: int, D: int, T: int,
    frame_clamped: torch.Tensor, frame_in_range: torch.Tensor,
    spot_to_orient: torch.Tensor, n_orient: int,
    obs: ObsVolume, p: FitParams,
    device: torch.device, dtype: torch.dtype,
    counts: np.ndarray, winners: List[Winner],
) -> ScreenResult:
    """Legacy per-voxel screen kernel; used only when ``gs`` varies
    across the batch.  See the docstring of :func:`screen` for the
    physics — this is a literal copy of the pre-vectorised loop body."""
    for vi_local, vi in enumerate(voxel_indices):
        verts = grid.triangle_vertices(int(vi))
        XG = torch.tensor(verts[:, 0], device=device, dtype=dtype)
        YG = torch.tensor(verts[:, 1], device=device, dtype=dtype)

        cos_w_ = cos_w.unsqueeze(1)
        sin_w_ = sin_w.unsqueeze(1)
        XG_ = XG.unsqueeze(0)
        YG_ = YG.unsqueeze(0)
        xa = XG_ * cos_w_ - YG_ * sin_w_
        ya = XG_ * sin_w_ + YG_ * cos_w_
        t_0 = 1.0 - xa / Lsd_0
        dy_v = ya + yl_0.unsqueeze(1) * t_0
        dz_v = zl_0.unsqueeze(1) * t_0
        if has_tilts:
            dy_v, dz_v = apply_nf_tilt(dy_v, dz_v, Lsd_0, R_tilt)
        v_y_px = dy_v / px + ybc[0]
        v_z_px = dz_v / px + zbc[0]

        cy_lab = yl_0
        cz_lab = zl_0
        if has_tilts:
            cy_lab, cz_lab = apply_nf_tilt(cy_lab, cz_lab, Lsd_0, R_tilt)
        c_y_px = cy_lab / px + ybc[0]
        c_z_px = cz_lab / px + zbc[0]

        vert_in_bounds = (
            (v_y_px >= 0) & (v_y_px < n_y)
            & (v_z_px >= 0) & (v_z_px < n_z)
        ).all(dim=1) & frame_in_range

        rel_y = v_y_px - c_y_px.unsqueeze(1)
        rel_z = v_z_px - c_z_px.unsqueeze(1)

        gs_um = float(grid.gs[vi])
        super_pixel = (2.0 * gs_um) > px

        if super_pixel:
            offsets_y, offsets_z, valid_mask = rasterize_triangles(rel_y, rel_z)
            valid_mask = valid_mask & vert_in_bounds.unsqueeze(1)
        else:
            cent_y = ((rel_y[:, 0] + rel_y[:, 1] + rel_y[:, 2]) / 3.0).round().long()
            cent_z = ((rel_z[:, 0] + rel_z[:, 1] + rel_z[:, 2]) / 3.0).round().long()
            offsets_y = cent_y.unsqueeze(1)
            offsets_z = cent_z.unsqueeze(1)
            valid_mask = vert_in_bounds.unsqueeze(1)

        K = offsets_y.shape[1]
        hits_per_pixel = torch.ones((T, K), device=device, dtype=dtype)
        bounds_per_pixel = valid_mask.clone()
        for d in range(D):
            scale = (Lsd[d] / Lsd_0)
            cy_d = (c_y_px - ybc[0]) * scale + ybc[d]
            cz_d = (c_z_px - zbc[0]) * scale + zbc[d]
            abs_y = cy_d.floor().long().unsqueeze(1) + offsets_y
            abs_z = cz_d.floor().long().unsqueeze(1) + offsets_z
            in_bd = (
                (abs_y >= 0) & (abs_y < n_y)
                & (abs_z >= 0) & (abs_z < n_z)
            )
            bounds_per_pixel = bounds_per_pixel & in_bd
            ay = abs_y.clamp(0, n_y - 1)
            az = abs_z.clamp(0, n_z - 1)
            f_idx = frame_clamped.unsqueeze(1).expand_as(ay)
            d_idx = torch.full_like(f_idx, d)
            hit = obs.lookup(d_idx, f_idx, ay, az)
            hits_per_pixel = hits_per_pixel * hit.to(dtype)

        hits_per_pixel = hits_per_pixel * bounds_per_pixel.to(dtype)
        total_per_spot = bounds_per_pixel.to(dtype).sum(dim=1)
        hits_per_spot = hits_per_pixel.sum(dim=1)

        if spot_weight is not None:
            _w = spot_weight.to(dtype)
            hits_per_spot = hits_per_spot.to(dtype) * _w
            total_per_spot = total_per_spot.to(dtype) * _w

        hits_per_orient = torch.zeros(n_orient, device=device, dtype=dtype)
        total_per_orient = torch.zeros(n_orient, device=device, dtype=dtype)
        hits_per_orient.scatter_add_(0, spot_to_orient, hits_per_spot)
        total_per_orient.scatter_add_(0, spot_to_orient, total_per_spot)
        frac = hits_per_orient / total_per_orient.clamp(min=1.0)

        keep = frac >= p.min_frac_accept
        keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
        if keep_idx.numel() > 0:
            keep_frac = frac[keep_idx].cpu().numpy()
            keep_orient = keep_idx.cpu().numpy().astype(np.int64)
            for f, oi in zip(keep_frac, keep_orient):
                winners.append(Winner(int(vi), int(oi), float(f)))
                counts[vi] += 1

    return ScreenResult(winners=winners, n_winners_per_voxel=counts)


# ---------------------------------------------------------------------------
#  Diagnostic CSV (matches C screen_cpu.csv)
# ---------------------------------------------------------------------------

def write_screen_csv(result: ScreenResult, path: str) -> None:
    """Dump winners sorted ``(voxel_idx, orient_idx)`` for diff vs. C.

    Output format matches the C ``screen_cpu.csv``::

        voxelIdx,orientIdx,fracOverlap
        0,12,0.842311
        ...
    """
    sorted_winners = sorted(
        result.winners,
        key=lambda w: (w.voxel_idx, w.orient_idx),
    )
    with open(path, "w") as f:
        f.write("voxelIdx,orientIdx,fracOverlap\n")
        for w in sorted_winners:
            f.write(f"{w.voxel_idx},{w.orient_idx},{w.frac_overlap:.6f}\n")
