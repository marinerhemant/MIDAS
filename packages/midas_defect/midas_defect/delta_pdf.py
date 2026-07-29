"""P7 — 3D-ΔPDF (defect pair-displacement map).

Take a sparse q-space voxel cloud, densify onto a regular grid, optionally
subtract a Bragg model (from `asterism_fit`), then inverse-3-D-FFT to
get a real-space pair-displacement correlation map `Δρ(r)`. The map's
dominant features should agree with the rod direction (in real-space, the
defect-plane normal) and with the Hendricks-Teller `d_layer` (the
inter-fault spacing).

Conventions (these are load-bearing — keep aligned with the rest of MIDAS):
  * **q-convention**: `q = 2π/d` (crystallographic), units 1/Å. A Bragg
    comb at planar spacing `d_layer` lives at `q_n = 2π · n / d_layer` along
    the rod direction, **not** at `n / d_layer`.
  * **r-convention**: `Δρ(r)` is in Å (real-space pair-displacement). FFT
    bin spacing `Δr = 2π / (N · Δq)`, derived directly from the q-comb
    aliasing of a periodic structure with spacing `2π / Δq`.
  * **Friedel symmetry**: `I(-q) = I(q)` is enforced by default
    (kinematic scattering, no anomalous correction at the moment).
  * **FFT placement**: `ifftshift` before the forward FFT, `fftshift`
    after, so the real-space origin (`r=0`) sits at index `n_grid // 2`
    in `delta_rho`.

Implementation is intentionally simple for v0.1:
  * Nearest-neighbor (intensity-conserving) accumulation onto the q-grid.
  * Optional Bragg subtraction using a list of `AsterismFit` (subtracts each
    fitted 3-D Gaussian from the dense grid).
  * Cosine-taper window before FFT (suppresses cube-edge ringing).
  * Optional Friedel symmetrization (`I(-q) = I(q)`).
  * `torch.fft.fftn` for autograd safety (so downstream P6 can backprop
    through the FFT if needed).
  * Mask deconvolution: NOT implemented in v0.1 (caveat in docstring).
    The user can read the unmasked FFT result with the standard caveat that
    features within a few unit-cells of any masked region are smeared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import math
import numpy as np
import torch

from midas_transforms.device import resolve_device, resolve_dtype


__all__ = [
    "DeltaPDFResult",
    "PerGrainDeltaPDF",
    "bragg_mask_from_fits",
    "compute_delta_pdf",
    "compute_delta_pdf_per_grain",
    "densify_to_qgrid",
    "profile_along_crystal_direction",
    "profiles_for_population",
    "subtract_bragg_from_qgrid",
    "variant_profile_bands",
    "wiener_deconvolve_mask",
    "radial_delta_pdf_profile",
]


@dataclass
class DeltaPDFResult:
    """Output of `compute_delta_pdf`."""
    delta_rho: np.ndarray         # (Nx, Ny, Nz) real-space scalar field
    r_axes: tuple                  # (rx, ry, rz) coordinate vectors in Å
    q_axes: tuple                  # the input q-grid
    diff_volume_max: float         # max(|Δρ|) for normalization later
    n_voxels_in: int               # how many sparse points went in
    bragg_subtracted: bool


@dataclass
class PerGrainDeltaPDF:
    """Output of :func:`compute_delta_pdf_per_grain`.

    Stores the per-grain `DeltaPDFResult` map keyed by grain id, plus a
    sorted list of grain ids for deterministic iteration and a count of
    voxels assigned to each grain (useful for downstream weighting and
    for spotting near-empty grains).
    """
    by_grain: dict             # {gID: DeltaPDFResult}
    grain_ids: np.ndarray      # (n_grains,) sorted unique grain ids
    n_voxels_per_grain: np.ndarray  # (n_grains,) voxel counts in input order
    q_axes: tuple              # shared q-grid (all grains use the same cube)
    r_axes: tuple              # shared r-grid

    def __iter__(self):
        # Iterate over grains that survived `min_voxels_per_grain` only;
        # `grain_ids` retains the full population for accounting, but
        # under-threshold grains are absent from `by_grain` by design.
        for gid in sorted(self.by_grain.keys()):
            yield int(gid), self.by_grain[int(gid)]


def densify_to_qgrid(
    qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, intensity: np.ndarray,
    *,
    q_max: float,
    n_grid: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor accumulation of (q, I) points into a regular cube.

    Returns `(grid, q_axis)`. `grid` is shape `(n_grid,)*3`. `q_axis` is
    shape `(n_grid,)` with the cell centers in 1/Å, spanning [-q_max, q_max].

    Out-of-range points are dropped (no clamping → no edge pile-up).
    """
    q_axis = np.linspace(-q_max, q_max, n_grid, endpoint=False)
    dq = q_axis[1] - q_axis[0]
    # cell index
    ix = np.floor((qx + q_max) / dq).astype(np.int64)
    iy = np.floor((qy + q_max) / dq).astype(np.int64)
    iz = np.floor((qz + q_max) / dq).astype(np.int64)
    valid = ((ix >= 0) & (ix < n_grid) &
             (iy >= 0) & (iy < n_grid) &
             (iz >= 0) & (iz < n_grid))
    ix = ix[valid]; iy = iy[valid]; iz = iz[valid]; I = intensity[valid]
    grid = np.zeros((n_grid, n_grid, n_grid), dtype=np.float32)
    np.add.at(grid, (ix, iy, iz), I.astype(np.float32))
    return grid, q_axis


def subtract_bragg_from_qgrid(
    grid: np.ndarray,
    fits: Sequence,        # list[AsterismFit]
    q_axis: np.ndarray,
    *,
    clip_negative: bool = True,
) -> np.ndarray:
    """Subtract each fitted 3-D Gaussian (no baseline) from the dense grid."""
    qx_g, qy_g, qz_g = np.meshgrid(q_axis, q_axis, q_axis, indexing="ij")
    out = grid.astype(np.float32).copy()
    for f in fits:
        Sigma = f.sigma_axes @ np.diag(f.sigma_eig ** 2) @ f.sigma_axes.T
        Sigma_inv = np.linalg.inv(Sigma)
        dx = qx_g - f.q_fit[0]
        dy = qy_g - f.q_fit[1]
        dz = qz_g - f.q_fit[2]
        v = np.stack([dx, dy, dz], axis=-1)
        quad = np.einsum("...i,ij,...j->...", v, Sigma_inv, v)
        bragg = f.amplitude * np.exp(-0.5 * quad)
        out -= bragg.astype(np.float32)
    if clip_negative:
        out = np.clip(out, 0.0, None)
    return out


def bragg_mask_from_fits(
    q_axis: np.ndarray,
    fits: Sequence,        # list[AsterismFit]
    *,
    sigma_scale: float = 3.0,
) -> np.ndarray:
    """Build a boolean q-cube mask that EXCLUDES the Bragg-fit ellipsoids.

    The mask is suitable as the second argument to
    :func:`wiener_deconvolve_mask`: `True` cells are kept (diffuse-only),
    Bragg-region cells are zeroed.

    Parameters
    ----------
    q_axis
        1-D cell-center axis used for densification (output of
        :func:`densify_to_qgrid`).
    fits
        Iterable of `AsterismFit`-like objects. Each must expose
        `q_fit (3,)`, `sigma_axes (3, 3)`, `sigma_eig (3,)`.
    sigma_scale
        Half-width of the excluded ellipsoid in units of the fitted Σ
        principal half-widths. The Mahalanobis distance threshold is
        `sigma_scale` (so 3.0 ⇒ all cells within 3σ along every axis are
        masked out).

    Returns
    -------
    mask : np.ndarray of float64, shape (n, n, n)
        1.0 where the cell is kept, 0.0 where it falls inside a Bragg
        ellipsoid. Returned as float so it can multiply intensity grids
        and feed directly into the Wiener inverse.
    """
    n = len(q_axis)
    qx_g, qy_g, qz_g = np.meshgrid(q_axis, q_axis, q_axis, indexing="ij")
    mask = np.ones((n, n, n), dtype=np.float64)
    thresh_sq = float(sigma_scale) ** 2
    for f in fits:
        Sigma = f.sigma_axes @ np.diag(f.sigma_eig ** 2) @ f.sigma_axes.T
        Sigma_inv = np.linalg.inv(Sigma)
        dx = qx_g - f.q_fit[0]
        dy = qy_g - f.q_fit[1]
        dz = qz_g - f.q_fit[2]
        v = np.stack([dx, dy, dz], axis=-1)
        d2 = np.einsum("...i,ij,...j->...", v, Sigma_inv, v)
        mask[d2 < thresh_sq] = 0.0
    return mask


def _cosine_taper_3d(n: int, frac: float = 0.1) -> np.ndarray:
    """Build a separable cosine-taper window of shape (n, n, n)."""
    edge = max(1, int(frac * n))
    win1 = np.ones(n, dtype=np.float32)
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(edge) / edge))
    win1[:edge] = ramp; win1[-edge:] = ramp[::-1]
    return np.einsum("i,j,k->ijk", win1, win1, win1)


def wiener_deconvolve_mask(
    grid: np.ndarray, mask: np.ndarray, *, lam: float = 0.01,
) -> np.ndarray:
    """Wiener-deconvolve the mask convolution from a Bragg-subtracted q-grid.

    Forward model: I_obs(q) = M(q) · I_true(q), where M ∈ {0, 1}.
    In real space: Δρ_obs = h ⊛ Δρ_true with h = FT⁻¹(M).

    A regularised Wiener inverse is:
        Δρ_deconv = FT⁻¹{ |H|² / (|H|² + λ) · FT{Δρ_obs} / H }
    where H = FT(h) = M. Stable as M ∈ {0, 1}: at measured cells the
    Wiener gain is 1/(1+λ); at unmeasured cells (M=0) the gain is 0,
    suppressing the mask's autocorrelation ringing.
    """
    import torch
    H = torch.as_tensor(mask, dtype=torch.float64)
    Y = torch.as_tensor(grid, dtype=torch.float64)
    Y_shift = torch.fft.ifftshift(Y)
    F_Y = torch.fft.fftn(Y_shift, norm="ortho")
    H_shift = torch.fft.ifftshift(H)
    F_H = torch.fft.fftn(H_shift, norm="ortho")
    # Wiener filter
    denom = (F_H.abs() ** 2) + lam
    F_X = (F_H.conj() / denom) * F_Y
    x = torch.fft.fftshift(torch.fft.ifftn(F_X, norm="ortho").real)
    return x.cpu().numpy()


def radial_delta_pdf_profile(
    delta_rho: np.ndarray, r_axis: np.ndarray,
    *, direction: np.ndarray, half_width: float = 1.0,
    n_samples: int = 200,
) -> dict:
    """Sample Δρ along a 1-D radial line through the origin, in a given direction.

    Uses trilinear interpolation. The line spans [-r_max, r_max] in
    real-space distance along the given direction; samples are returned as
    (r, Δρ) for plotting.

    `half_width` chooses how many lateral cells to average over for a
    tube-style profile (averages a small disc perpendicular to `direction`).
    """
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    n = delta_rho.shape[0]
    dr = r_axis[1] - r_axis[0]
    r_max = float(r_axis.max())
    t = np.linspace(-r_max, r_max, n_samples)
    # Trilinear interpolation helper
    def _interp(pts_xyz):
        """pts_xyz shape (M, 3) in physical r coordinates."""
        idx = (pts_xyz - r_axis[0]) / dr
        ix = np.clip(idx[:, 0].astype(int), 0, n - 2)
        iy = np.clip(idx[:, 1].astype(int), 0, n - 2)
        iz = np.clip(idx[:, 2].astype(int), 0, n - 2)
        fx = idx[:, 0] - ix; fy = idx[:, 1] - iy; fz = idx[:, 2] - iz
        c000 = delta_rho[ix,     iy,     iz    ]
        c100 = delta_rho[ix + 1, iy,     iz    ]
        c010 = delta_rho[ix,     iy + 1, iz    ]
        c110 = delta_rho[ix + 1, iy + 1, iz    ]
        c001 = delta_rho[ix,     iy,     iz + 1]
        c101 = delta_rho[ix + 1, iy,     iz + 1]
        c011 = delta_rho[ix,     iy + 1, iz + 1]
        c111 = delta_rho[ix + 1, iy + 1, iz + 1]
        c00 = c000 * (1 - fx) + c100 * fx
        c10 = c010 * (1 - fx) + c110 * fx
        c01 = c001 * (1 - fx) + c101 * fx
        c11 = c011 * (1 - fx) + c111 * fx
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        return c0 * (1 - fz) + c1 * fz

    # Sample along the line + a small lateral disc for averaging
    if half_width > 0:
        # Perpendicular basis
        u = direction
        if abs(u[2]) < 0.9:
            v = np.cross(u, np.array([0, 0, 1.0]))
        else:
            v = np.cross(u, np.array([1.0, 0, 0]))
        v /= np.linalg.norm(v)
        w = np.cross(u, v)
        offsets = []
        for du in np.linspace(-half_width, half_width, 5):
            for dv in np.linspace(-half_width, half_width, 5):
                offsets.append(du * v + dv * w)
        offsets = np.stack(offsets)
        profile = np.zeros(n_samples)
        for off in offsets:
            pts = t[:, None] * direction[None, :] + off[None, :]
            profile += _interp(pts)
        profile /= len(offsets)
    else:
        pts = t[:, None] * direction[None, :]
        profile = _interp(pts)
    return dict(t=t, profile=profile)


def profile_along_crystal_direction(
    res: DeltaPDFResult,
    OM: np.ndarray,
    direction_cry: np.ndarray,
    *,
    half_width: float = 1.0,
    n_samples: int = 200,
) -> dict:
    """Sample Δρ(r) along a crystal-frame direction mapped through OM.

    The Δρ map lives in the SAMPLE frame (because the input q-cloud is in
    the sample frame). To compare variants by a crystallographic axis
    (e.g. [111]_cry, the 9R lamella normal), rotate the crystal-frame
    direction into the sample frame with `OM` first and then sample.

    Parameters
    ----------
    res
        A single grain's `DeltaPDFResult` (or the population result if the
        grain orientations have already been aligned).
    OM
        Orientation matrix mapping crystal-frame vectors to sample-frame
        vectors: ``v_sample = OM @ v_cry``. Use the same convention as
        the rest of midas_defect / midas_stress.
    direction_cry
        3-vector in the crystal frame (e.g. ``[1, 1, 1]`` for [111]_cry).
    half_width, n_samples
        Forwarded to :func:`radial_delta_pdf_profile`.

    Returns
    -------
    dict
        Same shape as :func:`radial_delta_pdf_profile`: ``{"t", "profile"}``,
        plus ``"direction_sample"`` (the rotated unit vector) for plotting.
    """
    OM = np.asarray(OM, dtype=np.float64)
    direction_cry = np.asarray(direction_cry, dtype=np.float64)
    direction_sample = OM @ direction_cry
    direction_sample /= np.linalg.norm(direction_sample)
    profile = radial_delta_pdf_profile(
        res.delta_rho, res.r_axes[0],
        direction=direction_sample,
        half_width=half_width, n_samples=n_samples,
    )
    profile["direction_sample"] = direction_sample
    return profile


def profiles_for_population(
    per_grain: PerGrainDeltaPDF,
    OM_per_grain: dict,
    direction_cry: np.ndarray,
    *,
    half_width: float = 1.0,
    n_samples: int = 200,
) -> dict:
    """Crystal-direction Δρ profile for every grain in a `PerGrainDeltaPDF`.

    Wraps :func:`profile_along_crystal_direction` over the grain map.

    Parameters
    ----------
    per_grain
        Output of :func:`compute_delta_pdf_per_grain`.
    OM_per_grain
        ``{gID: OM (3, 3)}``. Grains missing from this map are skipped.
    direction_cry
        Crystal-frame direction (e.g. ``[1, 1, 1]``).

    Returns
    -------
    dict
        ``{gID: {"t", "profile", "direction_sample"}}``.
    """
    out: dict = {}
    for gid, res in per_grain:
        if int(gid) not in OM_per_grain:
            continue
        out[int(gid)] = profile_along_crystal_direction(
            res, OM_per_grain[int(gid)], direction_cry,
            half_width=half_width, n_samples=n_samples,
        )
    return out


def variant_profile_bands(
    profiles: dict,
    variant_of_grain: dict,
    *,
    n_boot: int = 500,
    rng_seed: int = 0,
    boot_unit: str = "grain",
) -> dict:
    """Per-variant bootstrap CI bands of crystal-direction Δρ profiles.

    Groups per-grain profiles (e.g. from :func:`profiles_for_population`)
    by variant label and runs an independent grain-resampling bootstrap
    inside each variant population. Returns one
    :class:`~midas_defect.bootstrap.aggregators.ProfileBand` per variant.

    Parameters
    ----------
    profiles
        ``{gID: {"t": (n_r,), "profile": (n_r,), ...}}`` as produced by
        :func:`profiles_for_population`. All grains' ``t`` arrays must be
        identical (shared r-axis from a shared q-cube — the default in
        :func:`compute_delta_pdf_per_grain`).
    variant_of_grain
        ``{gID: variant_label}``. Grains missing from this map are dropped.
        Variant labels are stored verbatim on the returned dict (any
        hashable: int, str, tuple, ...).
    n_boot
        Bootstrap draws per variant. Each draw resamples grains with
        replacement (full population size).
    rng_seed
        Base seed; variant ``i`` uses ``rng_seed + i`` so different
        variants are independent yet reproducible.

    Returns
    -------
    dict
        ``{variant_label: ProfileBand}``.
    """
    # Imported here to keep the file's top-level imports light and avoid
    # a circular dependency at module-import time.
    from .bootstrap import bootstrap_profile_band

    # Group grain ids by variant label, dropping any without a label.
    by_variant: dict = {}
    for gid, prof in profiles.items():
        if gid not in variant_of_grain:
            continue
        by_variant.setdefault(variant_of_grain[gid], []).append((gid, prof))

    out: dict = {}
    for i, (label, items) in enumerate(by_variant.items()):
        # All profiles must share the same r-axis.
        t0 = np.asarray(items[0][1]["t"], dtype=float)
        stacked = np.stack(
            [np.asarray(prof["profile"], dtype=float) for _, prof in items],
            axis=0,
        )
        out[label] = bootstrap_profile_band(
            stacked, t0,
            n_boot=n_boot, rng_seed=int(rng_seed) + i,
            boot_unit=boot_unit,
        )
    return out


def compute_delta_pdf(
    qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, intensity: np.ndarray,
    *,
    q_max: float,
    n_grid: int = 128,
    bragg_fits: Optional[Sequence] = None,
    symmetrize_friedel: bool = True,
    taper_frac: float = 0.1,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
) -> DeltaPDFResult:
    """Full pipeline: densify → optional Bragg subtract → window → 3D FFT.

    Parameters
    ----------
    q_max
        Half-extent of the q-cube in 1/Å. Anything outside [-q_max, q_max]³
        is dropped during densification.
    n_grid
        Grid points per axis. 128³ ≈ 8 MB float32, 256³ ≈ 64 MB.
    bragg_fits
        If given, subtract the per-hkl Gaussian models before FFT.
    symmetrize_friedel
        Average `I(q)` with `I(-q)`; helps fill the Ewald gap for free
        (kinematic scattering is centrosymmetric).
    taper_frac
        Width of cosine taper near the cube edges, as a fraction of `n_grid`.
    """
    device_ = resolve_device(device)
    dtype_  = resolve_dtype(device_, dtype)

    grid, q_axis = densify_to_qgrid(qx, qy, qz, intensity,
                                     q_max=q_max, n_grid=n_grid)
    if bragg_fits:
        grid = subtract_bragg_from_qgrid(grid, bragg_fits, q_axis)

    if symmetrize_friedel:
        grid_flip = grid[::-1, ::-1, ::-1].copy()
        grid = 0.5 * (grid + grid_flip)

    grid *= _cosine_taper_3d(n_grid, frac=taper_frac)

    g_t = torch.as_tensor(grid, dtype=dtype_, device=device_)
    # FFT-shift so the zero-frequency (= zero-q = origin in real space) is at
    # the center of the output.
    g_t_shift = torch.fft.ifftshift(g_t)
    Fr = torch.fft.fftn(g_t_shift, norm="ortho")
    delta_rho = torch.fft.fftshift(Fr).real

    # real-space axis: Δr = π / q_max  (= 2π / (2 q_max))
    dq = q_axis[1] - q_axis[0]
    dr = 2.0 * math.pi / (n_grid * dq)
    r_axis = (np.arange(n_grid) - n_grid // 2) * dr

    delta_rho_np = delta_rho.detach().cpu().numpy()
    return DeltaPDFResult(
        delta_rho=delta_rho_np,
        r_axes=(r_axis, r_axis, r_axis),
        q_axes=(q_axis, q_axis, q_axis),
        diff_volume_max=float(np.abs(delta_rho_np).max()),
        n_voxels_in=int(len(qx)),
        bragg_subtracted=bragg_fits is not None,
    )


def compute_delta_pdf_per_grain(
    qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, intensity: np.ndarray,
    grain_of_voxel: np.ndarray,
    *,
    q_max: float,
    n_grid: int = 128,
    bragg_fits_per_grain: Optional[dict] = None,
    symmetrize_friedel: bool = True,
    taper_frac: float = 0.1,
    min_voxels_per_grain: int = 50,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
) -> PerGrainDeltaPDF:
    """Compute one 3D-ΔPDF per labelled grain on a shared q-cube.

    The q-cube extent and grid are shared across all grains so that the
    per-grain Δρ(r) maps are directly comparable on the same (r_x, r_y, r_z)
    coordinates. Grains with fewer than `min_voxels_per_grain` voxels are
    skipped (their entry is absent from `by_grain` but their id appears in
    `grain_ids` with the corresponding `n_voxels_per_grain` count, so the
    caller can audit attrition).

    Parameters
    ----------
    grain_of_voxel
        Integer label per voxel. Same length as `qx`. Negative ids are
        treated as "unassigned" and dropped.
    bragg_fits_per_grain
        Optional `{gID: list[AsterismFit]}` map. If given, the corresponding
        grain's grid has its Bragg fits subtracted before FFT. Grains not
        present in the map run without subtraction. Pass an empty dict to
        suppress subtraction everywhere while keeping the routing.
    min_voxels_per_grain
        Lower cutoff on the per-grain voxel count. Falls in line with the
        densify resolution: below this you typically can't fill enough
        cells to make the FFT meaningful, and the cosine taper will dwarf
        what little signal you have.
    """
    qx = np.asarray(qx); qy = np.asarray(qy); qz = np.asarray(qz)
    intensity = np.asarray(intensity)
    grain_of_voxel = np.asarray(grain_of_voxel)
    if not (qx.shape == qy.shape == qz.shape == intensity.shape
            == grain_of_voxel.shape):
        raise ValueError(
            "qx, qy, qz, intensity, grain_of_voxel must have matching shape; "
            f"got {qx.shape}, {qy.shape}, {qz.shape}, {intensity.shape}, "
            f"{grain_of_voxel.shape}"
        )
    valid = grain_of_voxel >= 0
    qx_v = qx[valid]; qy_v = qy[valid]; qz_v = qz[valid]
    I_v = intensity[valid]; g_v = grain_of_voxel[valid]

    grain_ids = np.unique(g_v)
    n_per = np.array(
        [int((g_v == gid).sum()) for gid in grain_ids], dtype=np.int64,
    )

    by_grain: dict = {}
    fits_map = bragg_fits_per_grain or {}
    last_q_axes: tuple = ()
    last_r_axes: tuple = ()
    for gid, n in zip(grain_ids, n_per):
        if int(n) < int(min_voxels_per_grain):
            continue
        sel = (g_v == gid)
        fits = fits_map.get(int(gid))
        res = compute_delta_pdf(
            qx_v[sel], qy_v[sel], qz_v[sel], I_v[sel],
            q_max=q_max, n_grid=n_grid,
            bragg_fits=fits,
            symmetrize_friedel=symmetrize_friedel,
            taper_frac=taper_frac,
            device=device, dtype=dtype,
        )
        by_grain[int(gid)] = res
        last_q_axes = res.q_axes
        last_r_axes = res.r_axes

    if not by_grain:
        # No grain met the voxel-count cutoff. Synthesise the shared axes
        # from a trivial pass so the caller still gets consistent axes.
        _, q_axis = densify_to_qgrid(
            qx_v[:0], qy_v[:0], qz_v[:0], I_v[:0],
            q_max=q_max, n_grid=n_grid,
        )
        dq = q_axis[1] - q_axis[0]
        dr = 2.0 * math.pi / (n_grid * dq)
        r_axis = (np.arange(n_grid) - n_grid // 2) * dr
        last_q_axes = (q_axis, q_axis, q_axis)
        last_r_axes = (r_axis, r_axis, r_axis)

    return PerGrainDeltaPDF(
        by_grain=by_grain,
        grain_ids=grain_ids.astype(np.int64),
        n_voxels_per_grain=n_per,
        q_axes=last_q_axes,
        r_axes=last_r_axes,
    )
