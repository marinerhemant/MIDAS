"""Theoretical -> observed spot matching.

Mirrors `CompareSpots` from `FF_HEDM/src/IndexerOMP.c:460` and the GPU kernel
`gpu_CompareSpots` from `FF_HEDM/src/IndexerGPU.cu:297`.

Tie-break: smallest |delta_omega| match per theoretical spot. **Must match C
semantics exactly** for byte-identical regression — see dev/implementation_plan.md
§6.4. C uses strict `<` on diffOme so the lowest-index candidate wins on ties;
torch's `argmin` returns the first minimum, matching that behaviour.

Implementation strategy: build a dense `[..., max_nspots]` candidate gather
with masking. For typical bins this fits comfortably in memory; the
matching iteration becomes one vectorized pass across the (..., n_T) grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from .binning import get_bin_indices, lookup_bin_counts

# Numba is optional — if absent the CPU dispatch falls back to the torch
# m-iter path (still correct, just slower at PF scale). On chiltepin /
# alleppey / copland and the dev environments it's a stable transitive
# dep (used by find_grains, merge_scans, potts).
try:
    from numba import njit, prange  # type: ignore
    _NUMBA_AVAILABLE = True
except ImportError:
    njit = None
    prange = None
    _NUMBA_AVAILABLE = False

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


@dataclass
class MatchResult:
    """Per-evaluation-tuple match outcome.

    The integer fields ``n_matches``, ``n_matches_frac``, ``n_t_frac`` and
    the binary ``matched`` field always reflect the **hard** match decision
    (back-compat with the bit-exact C-parity gate).  When
    :func:`compare_spots` is called with a ``soft_beam_weight_fn``, the
    optional ``weighted_*`` fields are populated with the soft-attribution
    analogues:

    * ``weighted_n_matches``      = Σ_t  has_match(t) · w(best_cand(t))
    * ``weighted_n_matches_frac`` = same, restricted to non-rejected rings
    * ``weighted_frac_matches``   = weighted_n_matches_frac / n_t_frac

    Soft-mode consumers (e.g. ``midas_pipeline.stages.indexing`` with
    ``soft_beam_attribution=True``) score seeds by ``weighted_frac_matches``;
    the hard scoring path remains identical to today.
    """

    n_matches: torch.Tensor          # (N,) int64 — total matched theor spots
    n_matches_frac: torch.Tensor     # (N,) int64 — matches excluding rings_to_reject (denom of frac)
    n_t_frac: torch.Tensor           # (N,) int64 — valid theor spots excluding rings_to_reject
    frac_matches: torch.Tensor       # (N,) float — n_matches_frac / n_t_frac
    avg_ia: torch.Tensor             # (N,) float — IA average; placeholder for v0.1.0
    matched_obs_id: torch.Tensor     # (N, T) int64 — best obs spot id per theor spot, -1 if none
    matched_obs_row: torch.Tensor    # (N, T) int64 — row index in `obs` for each match, -1 if none
    delta_omega: torch.Tensor        # (N, T) float — |Δomega| for the best match, +inf if none
    matched: torch.Tensor            # (N, T) bool — match found per theor spot
    # Optional soft-attribution outputs (populated only when soft_beam_weight_fn is provided)
    weighted_n_matches:      torch.Tensor | None = None  # (N,) float
    weighted_n_matches_frac: torch.Tensor | None = None  # (N,) float
    weighted_frac_matches:   torch.Tensor | None = None  # (N,) float


def build_eta_margins(
    ring_radii: dict[int, float],
    margin_eta: float,
    stepsize_orient_deg: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_n_rings: int = 500,
) -> torch.Tensor:
    """Per-ring eta margin: `etamargins[r] = atan(MarginEta / R[r]) + 0.5 * StepOrient`.

    Mirrors IndexerOMP.c:1773-1779. Sparse rings (no radius) get 0.
    """
    arr = torch.zeros(max_n_rings, device=device, dtype=dtype)
    rad2deg = 180.0 / torch.pi
    half_step = 0.5 * stepsize_orient_deg
    for r, rad in ring_radii.items():
        if 0 < r < max_n_rings and rad > 0:
            arr[r] = (rad2deg * torch.atan(torch.tensor(margin_eta / rad, dtype=dtype, device=device))) + half_step
    return arr


def build_ome_margins(
    margin_ome: float,
    stepsize_orient_deg: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """181-element omega margin LUT keyed by `int(floor(|eta|))`.

    Mirrors IndexerOMP.c:1768-1772:
        omemargins[i] = MarginOme + 0.5 * StepOrient / |sin(i*deg2rad)|     for i in 1..179
        omemargins[0] = omemargins[180] = omemargins[1]
    """
    arr = torch.empty(181, device=device, dtype=dtype)
    deg2rad = torch.pi / 180.0
    for i in range(1, 180):
        arr[i] = margin_ome + 0.5 * stepsize_orient_deg / abs(torch.sin(torch.tensor(i * deg2rad, dtype=dtype)).item())
    arr[0] = arr[1]
    arr[180] = arr[1]
    return arr


# ---------------------------------------------------------------------------
# Auto-strategy picker — decide dense vs jagged based on predicted peak vs
# available device memory. Lets the GPU launch sites avoid OOM on dense
# datasets (e.g. Ti-7Al: 12 rings + crowded bins ⇒ huge (N, T, M) tensors)
# without forcing the caller to know the dataset shape ahead of time.
# ---------------------------------------------------------------------------

# Conservative per-cell bookkeeping in the dense path. Inside compare_spots,
# the (N, T, M) candidate gather builds roughly:
#   - 5 int64 tensors (cand_arange_b, rows, rows_clamped, spot_rows, cand_id)
#   - 7 bool   tensors (in_bin, rad_ok, eta_ok, radial_pass, radial_ok, ok,
#                       scan_ok in PF mode)
#   - 7 float  tensors (cand_ome, cand_eta, cand_rad, cand_ringrad,
#                       ref_rad_b, diff_ome, diff_ome_masked)
# Per-cell cost: 5*8 + 7*1 + 7*bytes_per_float
#   fp64 → 103 bytes/cell    fp32 →  75 bytes/cell
# Round up to leave headroom for short-lived intermediates the allocator
# may not free in time.
_PEAK_BYTES_PER_CELL_FP64 = 128
_PEAK_BYTES_PER_CELL_FP32 = 96
_JAGGED_CHUNK_MAX = 65536        # match _compare_spots_jagged default
_JAGGED_CHUNK_MIN = 64


def _per_cell_bytes(dtype: torch.dtype) -> int:
    return (_PEAK_BYTES_PER_CELL_FP64 if dtype == torch.float64
            else _PEAK_BYTES_PER_CELL_FP32)


def pick_compare_strategy(
    theor: torch.Tensor,
    max_n_cap: int | None,
    *,
    safety: float = 0.5,
    free_bytes: int | None = None,
) -> tuple[str, int]:
    """Pick ``compare_spots`` strategy + chunk_size from peak-memory prediction.

    Returns ``("dense", chunk_size)`` when the predicted peak of the (N, T, M)
    candidate-gather stack fits in ``free_bytes × safety``, else
    ``("jagged", chunk_size)`` with a chunk_size sized so each chunk stays
    inside the same budget.

    `chunk_size` is always returned (used only in the jagged path) so the
    caller can pass it through unconditionally.

    On non-CUDA devices, or when ``free_bytes`` cannot be probed, defaults
    to ``("dense", _JAGGED_CHUNK_MAX)`` — i.e. let the caller use the dense
    path it would have used before this picker existed.
    """
    if max_n_cap is None or max_n_cap <= 0:
        return "dense", _JAGGED_CHUNK_MAX
    if theor.ndim < 2:
        return "dense", _JAGGED_CHUNK_MAX
    N = theor.shape[0]
    T = theor.shape[1] if theor.ndim >= 2 else 1
    if N == 0 or T == 0:
        return "dense", _JAGGED_CHUNK_MAX

    on_cpu = theor.device.type != "cuda"
    if free_bytes is None:
        if on_cpu:
            # CPU / MPS: probe host RAM. Without this the dense path runs
            # unbounded and crashes the box on real-scale PF data (e.g.
            # Wenxi-class 91x91 voxels × 217k seeds where N×T×M can hit
            # 100s of GB per compare_spots call).
            try:
                import psutil
                free_bytes = int(psutil.virtual_memory().available)
            except Exception:
                # Last resort: /proc/meminfo on Linux, or assume 8 GiB.
                try:
                    with open("/proc/meminfo") as f:
                        for line in f:
                            if line.startswith("MemAvailable:"):
                                free_bytes = int(line.split()[1]) * 1024
                                break
                except Exception:
                    pass
                if free_bytes is None:
                    free_bytes = 8 * (1 << 30)  # 8 GiB conservative
        else:
            try:
                free_bytes, _total = torch.cuda.mem_get_info(theor.device)
            except Exception:
                return "dense", _JAGGED_CHUNK_MAX

    bytes_per_cell = _per_cell_bytes(theor.dtype)
    # CPU torch's allocator doesn't release back to the OS as eagerly as
    # CUDA's caching allocator, and the compare_spots call sites stack
    # several (N, T, M) intermediates. Use a much tighter safety factor on
    # CPU so chunking actually bounds per-call memory; on big-memory hosts
    # (chiltepin, 1.5 TB) a 0.5 safety still picks chunk = N which doesn't
    # chunk. Also hard-cap CPU chunk size to keep per-call peaks well under
    # 8 GiB regardless of host size.
    effective_safety = safety if not on_cpu else min(safety, 0.02)
    max_call_bytes_cpu = 8 * (1 << 30)
    budget = max(1, int(free_bytes * effective_safety))
    if on_cpu:
        budget = min(budget, max_call_bytes_cpu)
    predicted_dense = N * T * max_n_cap * bytes_per_cell

    if predicted_dense <= budget:
        return "dense", _JAGGED_CHUNK_MAX

    # Size the chunk so chunk_size × T × M × bytes_per_cell ≤ budget.
    per_n_bytes = max(1, T * max_n_cap * bytes_per_cell)
    chunk = max(_JAGGED_CHUNK_MIN, min(_JAGGED_CHUNK_MAX, budget // per_n_bytes))
    chunk = min(chunk, N)
    return "jagged", int(chunk)


def compare_spots(
    theor: torch.Tensor,             # (N, T, 14) float
    valid: torch.Tensor,             # (N, T) bool
    obs: torch.Tensor,               # (n_obs, 9) float — or (n_obs, 10) when scanning
    bin_data: torch.Tensor,          # int32 flat — Data.bin
    bin_ndata: torch.Tensor,         # int32 flat — nData.bin (interleaved count, offset)
    *,
    ref_rad: torch.Tensor,           # (N,) float — per-tuple reference radius
    margin_rad: float,
    margin_radial: float,
    eta_margins: torch.Tensor,       # (max_n_rings,) per-ring eta margin
    ome_margins: torch.Tensor,       # (181,) eta-keyed omega margin LUT
    eta_bin_size: float,
    ome_bin_size: float,
    n_eta_bins: int,
    n_ome_bins: int,
    rings_to_reject: torch.Tensor,   # (n_reject,) int — for skipRadialFilter + nMatchesFracCalc
    distance: float | None = None,    # Lsd in um — required for avg_ia computation
    pos: torch.Tensor | None = None,  # (N, 3) sample-frame (ga,gb,gc) — required for avg_ia
    strategy: str = "dense",          # "dense" or "jagged"
    chunk_size: int = 65536,          # for "jagged": rows of N processed per chunk
    max_n_cap: int | None = None,     # if known, skip the per-call n_per.max() sync
    # --- Scan-aware (pf-HEDM) extensions ---
    scan_positions: torch.Tensor | None = None,  # (n_scans,) 1-D Y values (µm)
    voxel_xy: torch.Tensor | None = None,         # (N, 2) per-tuple (x, y) in µm
    scan_pos_tol_um: float = 0.0,                 # 0 ⇒ filter disabled (FF default)
    friedel_symmetric_scan_filter: bool = False,  # single-sided default = matches C + correct physics
    obs_scan_nr_int64: torch.Tensor | None = None,  # cached obs[..., 9].long() from IndexerContext
    # --- Soft attribution (P6 of the V-map plan) ---
    soft_beam_weight_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> MatchResult:
    """Vectorized binned matching. See module docstring for tie-break semantics.

    `strategy="dense"` (default): one allocation of the candidate gather
    `(N, T, max_n)` across all tuples. Fastest when memory permits.

    `strategy="jagged"`: split N into chunks of `chunk_size` and process
    each chunk sequentially, then concatenate. Identical numerics to
    "dense" but bounds peak memory by `chunk_size * T * max_n`. Use when
    the dense path OOMs (huge N or unbalanced bins).
    """
    if strategy not in ("dense", "jagged"):
        raise ValueError(
            f"strategy must be 'dense' or 'jagged'; got {strategy!r}"
        )
    if strategy == "jagged" and theor.shape[0] > chunk_size:
        return _compare_spots_jagged(
            theor=theor, valid=valid, obs=obs,
            bin_data=bin_data, bin_ndata=bin_ndata,
            ref_rad=ref_rad,
            margin_rad=margin_rad, margin_radial=margin_radial,
            eta_margins=eta_margins, ome_margins=ome_margins,
            eta_bin_size=eta_bin_size, ome_bin_size=ome_bin_size,
            n_eta_bins=n_eta_bins, n_ome_bins=n_ome_bins,
            rings_to_reject=rings_to_reject,
            distance=distance, pos=pos,
            chunk_size=chunk_size,
            max_n_cap=max_n_cap,
            scan_positions=scan_positions,
            voxel_xy=voxel_xy,
            scan_pos_tol_um=scan_pos_tol_um,
            friedel_symmetric_scan_filter=friedel_symmetric_scan_filter,
            obs_scan_nr_int64=obs_scan_nr_int64,
            soft_beam_weight_fn=soft_beam_weight_fn,
        )

    # CPU dispatch: per-cell numba loop with early-exit when bin is empty.
    # The dense (N, T, M) torch path scales O(N·T·M_global) regardless of
    # actual bin occupancy. On PF data (Wenxi) 99% of (N, T) cells point
    # to empty bins — numba's per-cell loop skips them entirely, matching
    # C IndexerScanningOMP's algorithm. Soft-attribution falls back to the
    # torch m-iter path because @njit can't call arbitrary Python callbacks.
    if theor.device.type != "cuda":
        if soft_beam_weight_fn is None:
            return _compare_spots_numba(
                theor=theor, valid=valid, obs=obs,
                bin_data=bin_data, bin_ndata=bin_ndata,
                ref_rad=ref_rad,
                margin_rad=margin_rad, margin_radial=margin_radial,
                eta_margins=eta_margins, ome_margins=ome_margins,
                eta_bin_size=eta_bin_size, ome_bin_size=ome_bin_size,
                n_eta_bins=n_eta_bins, n_ome_bins=n_ome_bins,
                rings_to_reject=rings_to_reject,
                distance=distance, pos=pos,
                scan_positions=scan_positions,
                voxel_xy=voxel_xy,
                scan_pos_tol_um=scan_pos_tol_um,
                friedel_symmetric_scan_filter=friedel_symmetric_scan_filter,
                obs_scan_nr_int64=obs_scan_nr_int64,
            )
        return _compare_spots_m_iter(
            theor=theor, valid=valid, obs=obs,
            bin_data=bin_data, bin_ndata=bin_ndata,
            ref_rad=ref_rad,
            margin_rad=margin_rad, margin_radial=margin_radial,
            eta_margins=eta_margins, ome_margins=ome_margins,
            eta_bin_size=eta_bin_size, ome_bin_size=ome_bin_size,
            n_eta_bins=n_eta_bins, n_ome_bins=n_ome_bins,
            rings_to_reject=rings_to_reject,
            distance=distance, pos=pos,
            max_n_cap=max_n_cap,
            scan_positions=scan_positions,
            voxel_xy=voxel_xy,
            scan_pos_tol_um=scan_pos_tol_um,
            friedel_symmetric_scan_filter=friedel_symmetric_scan_filter,
            obs_scan_nr_int64=obs_scan_nr_int64,
            soft_beam_weight_fn=soft_beam_weight_fn,
        )

    device = theor.device
    dtype = theor.dtype
    N, T, _ = theor.shape

    ring_nr = theor[..., 9].to(torch.int64).clamp(min=0)        # (N, T)
    eta_post = theor[..., 12]                                    # (N, T)
    omega = theor[..., 6]                                        # (N, T)
    rad_diff = theor[..., 13]                                    # (N, T)

    # 1. Bin lookup
    bin_pos = get_bin_indices(
        ring_nr, eta_post, omega, eta_bin_size, ome_bin_size, n_eta_bins, n_ome_bins,
    )                                                            # (N, T)
    # Out-of-range pos → bin count 0 (no candidates).
    bin_pos = bin_pos.clamp(0, max(0, (bin_ndata.numel() // 2) - 1))
    n_per, data_offset = lookup_bin_counts(bin_pos, bin_ndata)   # (N, T) each

    # 2. Dense candidate gather. max_nspots = max bin occupancy.
    # When `max_n_cap` is provided (precomputed at IndexerContext init from
    # the bin_ndata global max), we avoid the per-call `n_per.max().item()`
    # GPU sync — letting `compare_spots` enqueue all kernels async, which is
    # needed for the prefetch overlap in `run_block`.
    if max_n_cap is not None:
        max_n = int(max_n_cap)
    else:
        max_n = int(n_per.max().item()) if n_per.numel() else 0
    if max_n == 0:
        zeros = torch.zeros((N, T), dtype=torch.bool, device=device)
        soft_zeros = (
            torch.zeros(N, dtype=dtype, device=device)
            if soft_beam_weight_fn is not None else None
        )
        return MatchResult(
            n_matches=torch.zeros(N, dtype=torch.int64, device=device),
            n_matches_frac=torch.zeros(N, dtype=torch.int64, device=device),
            n_t_frac=valid.sum(dim=-1).to(torch.int64).clamp_min(1),
            frac_matches=torch.zeros(N, dtype=dtype, device=device),
            avg_ia=torch.zeros(N, dtype=dtype, device=device),
            matched_obs_id=torch.full((N, T), -1, dtype=torch.int64, device=device),
            matched_obs_row=torch.full((N, T), -1, dtype=torch.int64, device=device),
            delta_omega=torch.full((N, T), float("inf"), dtype=dtype, device=device),
            matched=zeros,
            weighted_n_matches=soft_zeros,
            weighted_n_matches_frac=soft_zeros.clone() if soft_zeros is not None else None,
            weighted_frac_matches=soft_zeros.clone() if soft_zeros is not None else None,
        )

    # arange over candidate axis, masked by per-cell n_per.
    cand_arange = torch.arange(max_n, device=device, dtype=torch.int64)        # (M,)
    cand_arange_b = cand_arange.expand(N, T, max_n)                            # (N, T, M)
    in_bin = cand_arange_b < n_per.unsqueeze(-1)                                # (N, T, M)
    rows = data_offset.unsqueeze(-1) + cand_arange_b
    rows_clamped = rows.clamp(0, bin_data.numel() - 1)
    spot_rows = bin_data[rows_clamped].to(torch.int64)                          # (N, T, M)
    # When out-of-bin, point to row 0 — masked off below.

    # 3. Pull observed values per candidate
    obs_y = obs[..., 0]
    obs_z = obs[..., 1]
    obs_ome = obs[..., 2]
    obs_ringrad = obs[..., 3]
    obs_id = obs[..., 4].to(torch.int64)
    obs_eta = obs[..., 6]
    obs_rad = obs[..., 8]

    cand_ome = obs_ome[spot_rows]                       # (N, T, M)
    cand_eta = obs_eta[spot_rows]
    cand_rad = obs_rad[spot_rows]
    cand_ringrad = obs_ringrad[spot_rows]
    cand_id = obs_id[spot_rows]

    # 4. Margin checks
    rad_ok = (rad_diff.unsqueeze(-1) - cand_rad).abs() < margin_radial
    eta_margin_per = eta_margins[ring_nr.clamp(0, eta_margins.numel() - 1)]    # (N, T)
    eta_ok = (eta_post.unsqueeze(-1) - cand_eta).abs() < eta_margin_per.unsqueeze(-1)

    # skipRadialFilter for rings in rings_to_reject. Otherwise enforce
    # |RefRad - obs[3]| < MarginRad.
    if rings_to_reject.numel() > 0:
        skip_radial = (
            ring_nr.unsqueeze(-1) == rings_to_reject.view(1, 1, -1)
        ).any(dim=-1)                                  # (N, T)
    else:
        skip_radial = torch.zeros((N, T), dtype=torch.bool, device=device)
    # ref_rad expands to (N, 1) since it's per-tuple
    ref_rad_b = ref_rad.view(N, 1, 1).expand(N, T, max_n)
    radial_pass = (ref_rad_b - cand_ringrad).abs() < margin_rad
    radial_ok = skip_radial.unsqueeze(-1) | radial_pass

    ok = in_bin & rad_ok & radial_ok & eta_ok & valid.unsqueeze(-1)            # (N, T, M)

    # 4b. Scan-position filter (pf-HEDM mode). Default-off (tol == 0) — FF
    # behavior is unchanged. When active, drop candidates whose observed
    # scan position is inconsistent with the voxel's lab-frame (x, y).
    #
    # Filter expression (from IndexerScanningOMP.c:453-459):
    #     s_proj = xThis * cos(omega) + yThis * sin(omega)
    #     keep if |s_proj − ypos[scannrobs]| < scan_pos_tol_um
    #
    # Production default is **Friedel-symmetric** per plan §1b:
    #     keep if (|s_proj − ypos| < tol) OR (|−s_proj − ypos| < tol)
    # The single-sided form (Friedel OFF) is required for the bit-exact
    # parity gate against IndexerScanningOMP.
    if (
        scan_pos_tol_um > 0
        and scan_positions is not None
        and voxel_xy is not None
    ):
        if obs.shape[-1] < 10:
            raise ValueError(
                "scan-aware mode requires obs with 10 columns (Spots.bin PF "
                "layout); got %d." % obs.shape[-1]
            )
        # Per-tuple voxel (x, y) — shape (N, 1) for broadcast across T.
        v_x = voxel_xy[..., 0].view(N, 1).to(dtype=dtype, device=device)
        v_y = voxel_xy[..., 1].view(N, 1).to(dtype=dtype, device=device)
        # Project voxel onto rotated scan axis per-theor-spot using omega (deg → rad).
        # Convention matches IndexerScanningOMP.c:440 + spots[][14/15] = sin/cos:
        #     yRot = xThis * sin(omega) + yThis * cos(omega)
        omega_rad = omega * DEG2RAD                                            # (N, T)
        s_proj = v_x * torch.sin(omega_rad) + v_y * torch.cos(omega_rad)        # (N, T)

        # Per-candidate scan position (col 9 of obs is scanNr, indexes
        # scan_positions). Prefer the IndexerContext-cached int64 tensor
        # when available — avoids re-casting on every compare_spots call
        # (the dominant per-voxel hot path in scan mode).
        if obs_scan_nr_int64 is not None:
            obs_scan_idx = obs_scan_nr_int64
        else:
            obs_scan_idx = obs[..., 9].to(torch.int64)                          # (n_obs,)
        scan_pos_arr = scan_positions.to(dtype=dtype, device=device)
        cand_scan_idx = obs_scan_idx[spot_rows]                                 # (N, T, M)
        cand_scan_pos = scan_pos_arr[cand_scan_idx.clamp(0, scan_pos_arr.numel() - 1)]

        diff = (s_proj.unsqueeze(-1) - cand_scan_pos).abs()
        scan_weights: torch.Tensor | None = None
        if soft_beam_weight_fn is not None:
            scan_weights = soft_beam_weight_fn(diff).to(dtype=dtype)
            if friedel_symmetric_scan_filter:
                diff_f = (s_proj.unsqueeze(-1) + cand_scan_pos).abs()
                scan_weights = torch.maximum(
                    scan_weights,
                    soft_beam_weight_fn(diff_f).to(dtype=dtype),
                )
            scan_ok = scan_weights > 0
        else:
            scan_ok = diff < scan_pos_tol_um
            if friedel_symmetric_scan_filter:
                # Friedel pair: matching spot may appear at +scan or -scan offset.
                diff_friedel = (s_proj.unsqueeze(-1) + cand_scan_pos).abs()
                scan_ok = scan_ok | (diff_friedel < scan_pos_tol_um)
        ok = ok & scan_ok                                                       # (N, T, M)
    else:
        scan_weights = None

    # 5. Tie-break on smallest |Δomega|
    diff_ome = (omega.unsqueeze(-1) - cand_ome).abs()
    diff_ome_masked = torch.where(
        ok, diff_ome, torch.full_like(diff_ome, float("inf"))
    )
    best_idx = diff_ome_masked.argmin(dim=-1)                                  # (N, T)
    has_match = ok.any(dim=-1)                                                  # (N, T)

    # gather chosen candidate id + row + delta
    matched_id = cand_id.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)
    matched_row = spot_rows.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)
    delta_ome = diff_ome.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)
    matched_id = torch.where(
        has_match, matched_id, torch.full_like(matched_id, -1)
    )
    matched_row = torch.where(
        has_match, matched_row, torch.full_like(matched_row, -1)
    )
    delta_ome = torch.where(
        has_match, delta_ome, torch.full_like(delta_ome, float("inf"))
    )

    # 6. Counts. n_matches counts all matches; n_matches_frac_calc excludes
    # matches in rings_to_reject (matches `nMatchesFracCalc` in C:535-541).
    matched_for_frac = has_match & ~skip_radial
    n_matches = has_match.sum(dim=-1).to(torch.int64)
    n_matches_frac = matched_for_frac.sum(dim=-1).to(torch.int64)

    # nTspotsFracCalc denominator: count valid theor spots not in rejected rings.
    valid_for_frac = valid & ~skip_radial
    n_t_frac = valid_for_frac.sum(dim=-1).to(torch.int64).clamp_min(1)
    frac = n_matches_frac.to(dtype) / n_t_frac.to(dtype)

    # Soft-attribution weighted counts (only when soft_beam_weight_fn given AND
    # we're in scan-aware mode that populated scan_weights). For every theor
    # spot with a match, gather the weight at the best (Δω-minimizing)
    # candidate and sum across the T axis.
    if scan_weights is not None:
        best_weight = scan_weights.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)  # (N, T)
        best_weight = torch.where(has_match, best_weight, torch.zeros_like(best_weight))
        weighted_n_matches = (best_weight * has_match.to(dtype)).sum(dim=-1)
        weighted_n_matches_frac = (
            best_weight * matched_for_frac.to(dtype)
        ).sum(dim=-1)
        weighted_frac_matches = weighted_n_matches_frac / n_t_frac.to(dtype)
    else:
        weighted_n_matches = None
        weighted_n_matches_frac = None
        weighted_frac_matches = None

    # avg_ia: per-tuple internal-angle average between matched theor/obs g-vectors.
    # Mirrors `CalcIA` from FF_HEDM/src/IndexerOMP.c:1654.
    if distance is not None and pos is not None:
        avg_ia = _compute_avg_ia(
            theor=theor,                            # (N, T, 14)
            obs=obs,
            matched_obs_row=matched_row,
            has_match=has_match,
            distance=distance,
            pos=pos,
        )
    else:
        avg_ia = torch.zeros(N, dtype=dtype, device=device)

    return MatchResult(
        n_matches=n_matches,
        n_matches_frac=n_matches_frac,
        n_t_frac=n_t_frac,
        frac_matches=frac,
        avg_ia=avg_ia,
        matched_obs_id=matched_id,
        matched_obs_row=matched_row,
        delta_omega=delta_ome,
        matched=has_match,
        weighted_n_matches=weighted_n_matches,
        weighted_n_matches_frac=weighted_n_matches_frac,
        weighted_frac_matches=weighted_frac_matches,
    )


def _compare_spots_jagged(
    theor: torch.Tensor,
    valid: torch.Tensor,
    obs: torch.Tensor,
    bin_data: torch.Tensor,
    bin_ndata: torch.Tensor,
    *,
    ref_rad: torch.Tensor,
    margin_rad: float,
    margin_radial: float,
    eta_margins: torch.Tensor,
    ome_margins: torch.Tensor,
    eta_bin_size: float,
    ome_bin_size: float,
    n_eta_bins: int,
    n_ome_bins: int,
    rings_to_reject: torch.Tensor,
    distance: float | None,
    pos: torch.Tensor | None,
    chunk_size: int,
    max_n_cap: int | None = None,
    scan_positions: torch.Tensor | None = None,
    voxel_xy: torch.Tensor | None = None,
    scan_pos_tol_um: float = 0.0,
    friedel_symmetric_scan_filter: bool = False,
    obs_scan_nr_int64: torch.Tensor | None = None,
    soft_beam_weight_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> MatchResult:
    """Memory-bounded variant of `compare_spots`: chunks N axis into
    `chunk_size` slabs and concatenates per-slab MatchResults.

    Same numerics as the dense path. Use when N is large enough that
    `(N, T, max_n)` doesn't fit in memory.
    """
    N = theor.shape[0]
    chunks: list[MatchResult] = []
    for start in range(0, N, chunk_size):
        stop = min(start + chunk_size, N)
        sl = slice(start, stop)
        chunk_pos = pos[sl] if pos is not None else None
        chunk_voxel_xy = voxel_xy[sl] if voxel_xy is not None else None
        chunks.append(
            compare_spots(
                theor=theor[sl], valid=valid[sl], obs=obs,
                bin_data=bin_data, bin_ndata=bin_ndata,
                ref_rad=ref_rad[sl],
                margin_rad=margin_rad, margin_radial=margin_radial,
                eta_margins=eta_margins, ome_margins=ome_margins,
                eta_bin_size=eta_bin_size, ome_bin_size=ome_bin_size,
                n_eta_bins=n_eta_bins, n_ome_bins=n_ome_bins,
                rings_to_reject=rings_to_reject,
                distance=distance, pos=chunk_pos,
                strategy="dense",
                max_n_cap=max_n_cap,
                scan_positions=scan_positions,
                voxel_xy=chunk_voxel_xy,
                scan_pos_tol_um=scan_pos_tol_um,
                friedel_symmetric_scan_filter=friedel_symmetric_scan_filter,
                obs_scan_nr_int64=obs_scan_nr_int64,
                soft_beam_weight_fn=soft_beam_weight_fn,
            )
        )

    def _cat_opt(field: str):
        vals = [getattr(c, field) for c in chunks]
        if any(v is None for v in vals):
            return None
        return torch.cat(vals, dim=0)

    return MatchResult(
        n_matches=torch.cat([c.n_matches for c in chunks], dim=0),
        n_matches_frac=torch.cat([c.n_matches_frac for c in chunks], dim=0),
        n_t_frac=torch.cat([c.n_t_frac for c in chunks], dim=0),
        frac_matches=torch.cat([c.frac_matches for c in chunks], dim=0),
        avg_ia=torch.cat([c.avg_ia for c in chunks], dim=0),
        matched_obs_id=torch.cat([c.matched_obs_id for c in chunks], dim=0),
        matched_obs_row=torch.cat([c.matched_obs_row for c in chunks], dim=0),
        delta_omega=torch.cat([c.delta_omega for c in chunks], dim=0),
        matched=torch.cat([c.matched for c in chunks], dim=0),
        weighted_n_matches=_cat_opt("weighted_n_matches"),
        weighted_n_matches_frac=_cat_opt("weighted_n_matches_frac"),
        weighted_frac_matches=_cat_opt("weighted_frac_matches"),
    )


def _spot_to_gv_pos(
    xi: torch.Tensor,           # (..,)
    yi: torch.Tensor,
    zi: torch.Tensor,
    omega_deg: torch.Tensor,
    cx: torch.Tensor,           # broadcastable
    cy: torch.Tensor,
    cz: torch.Tensor,
) -> torch.Tensor:
    """Vectorized port of `spot_to_gv_pos` from IndexerOMP.c:748.

    Returns g-vectors as (.., 3). Subtracts the omega-rotated grain position
    from the spot coordinates, then converts to a unit-sphere reciprocal
    vector via `spot_to_gv` (line 726).
    """
    # RotateAroundZ(c, omega): vr = (cos*cx - sin*cy, sin*cx + cos*cy, cz)
    # The function then rotates by -omega for the final transform; by parity
    # cos(-x)=cos(x), sin(-x)=-sin(x) — reuse (co, so) instead of recomputing.
    omega_rad = omega_deg * DEG2RAD
    co = torch.cos(omega_rad)
    so = torch.sin(omega_rad)
    vr_x = co * cx - so * cy
    vr_y = so * cx + co * cy
    vr_z = cz
    xi = xi - vr_x
    yi = yi - vr_y
    zi = zi - vr_z
    L = torch.sqrt(xi * xi + yi * yi + zi * zi).clamp_min(1e-30)
    xn = xi / L
    yn = yi / L
    zn = zi / L
    g1r = -1.0 + xn
    g2r = yn
    # Rotate by -omega: cos(-ω)=co, sin(-ω)=-so.
    g1 = g1r * co + g2r * so       # was: g1r*cos_neg - g2r*sin_neg
    g2 = -g1r * so + g2r * co      # was: g1r*sin_neg + g2r*cos_neg
    g3 = zn
    return torch.stack([g1, g2, g3], dim=-1)


def _compute_avg_ia(
    theor: torch.Tensor,            # (N, T, 14)
    obs: torch.Tensor,              # (n_obs, 9)
    matched_obs_row: torch.Tensor,  # (N, T) int64; row index into `obs`, -1 if no match
    has_match: torch.Tensor,        # (N, T) bool
    distance: float,
    pos: torch.Tensor,              # (N, 3) (ga, gb, gc)
) -> torch.Tensor:
    """Compute mean internal-angle (degrees) between theor/obs g-vectors per tuple.

    Mirrors `CalcIA` from IndexerOMP.c:1654 + `CalcInternalAngle` from line 253.
    Unmatched theor spots contribute nothing (C uses 999 sentinel + skip).
    """
    N, T, _ = theor.shape
    device = theor.device
    dtype = theor.dtype

    # Theor side: y = col 10 (yl_disp), z = col 11 (zl_disp), omega = col 6.
    theor_y = theor[..., 10]                                          # (N, T)
    theor_z = theor[..., 11]
    theor_ome = theor[..., 6]
    xi_t = torch.full_like(theor_y, distance)

    # Obs side: gather by matched_obs_row (clamp -1 to 0; mask later).
    safe_row = matched_obs_row.clamp_min(0)
    obs_y = obs[..., 0][safe_row]                                      # (N, T)
    obs_z = obs[..., 1][safe_row]
    obs_ome = obs[..., 2][safe_row]
    xi_o = torch.full_like(obs_y, distance)

    # Grain position broadcasts to (N, T): cx, cy, cz from `pos[N, 3]`.
    cx = pos[:, 0:1].expand(N, T)
    cy = pos[:, 1:2].expand(N, T)
    cz = pos[:, 2:3].expand(N, T)

    gv1 = _spot_to_gv_pos(xi_t, theor_y, theor_z, theor_ome, cx, cy, cz)  # (N, T, 3)
    gv2 = _spot_to_gv_pos(xi_o, obs_y, obs_z, obs_ome, cx, cy, cz)

    n1 = torch.linalg.vector_norm(gv1, dim=-1).clamp_min(1e-30)
    n2 = torch.linalg.vector_norm(gv2, dim=-1).clamp_min(1e-30)
    cos_ia = (gv1 * gv2).sum(dim=-1) / (n1 * n2)
    cos_ia = cos_ia.clamp(-1.0, 1.0)
    ia_deg = torch.acos(cos_ia) * RAD2DEG                              # (N, T)

    # Average over matched spots only. Unmatched -> contribute 0 to numerator
    # and 0 to denominator (matches C's 999 sentinel skip).
    ia_abs = ia_deg.abs()
    masked = torch.where(has_match, ia_abs, torch.zeros_like(ia_abs))
    n_match = has_match.sum(dim=-1).to(dtype).clamp_min(1.0)
    return masked.sum(dim=-1) / n_match


# ---------------------------------------------------------------------------
# M-iterating compare path (CPU-fast)
# ---------------------------------------------------------------------------


def _compare_spots_m_iter(
    theor: torch.Tensor,
    valid: torch.Tensor,
    obs: torch.Tensor,
    bin_data: torch.Tensor,
    bin_ndata: torch.Tensor,
    *,
    ref_rad: torch.Tensor,
    margin_rad: float,
    margin_radial: float,
    eta_margins: torch.Tensor,
    ome_margins: torch.Tensor,
    eta_bin_size: float,
    ome_bin_size: float,
    n_eta_bins: int,
    n_ome_bins: int,
    rings_to_reject: torch.Tensor,
    distance: float | None,
    pos: torch.Tensor | None,
    max_n_cap: int | None = None,
    scan_positions: torch.Tensor | None = None,
    voxel_xy: torch.Tensor | None = None,
    scan_pos_tol_um: float = 0.0,
    friedel_symmetric_scan_filter: bool = False,
    obs_scan_nr_int64: torch.Tensor | None = None,
    soft_beam_weight_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> MatchResult:
    """M-iterating compare path: same numerics as ``compare_spots`` dense path,
    but loops over the bin-occupancy axis ``m`` one slice at a time.

    Why this matters on CPU at PF scale:

    The dense path allocates a ``(N, T, M)`` bool stack where ``M`` is the
    global max bin occupancy. On Wenxi-class data, ``M = 177`` but the
    median bin holds 0.2 spots and the 99th-percentile holds 1 — so the
    dense gather wastes ~99% of its budget processing empty cells.

    This path:
      1. Computes ``n_per`` and ``data_offset`` once on the (N, T) grid.
      2. Computes a per-call ``actual_max_n = n_per.max()`` (opt C — uses
         the per-voxel maximum, not the global cap).
      3. Iterates ``m`` from 0 to ``actual_max_n``: each iteration builds
         only (N, T) tensors, runs the margin checks, and updates a
         running ``best_delta_ome``/``best_matched_id`` per (N, T) cell.

    Memory per iter: ``O(N·T)`` instead of ``O(N·T·M)`` — for Wenxi that's
    ~14 MB vs ~4.8 GB. Total ops are the same as dense, but each iter
    fits in cache and the allocator never sees a (N, T, M) blob.

    See ``compare_spots`` for the full inputs/outputs contract; this
    function returns an equivalent ``MatchResult``.
    """
    device = theor.device
    dtype = theor.dtype
    N, T, _ = theor.shape

    # ── 0. Per-theor-spot ring + post-displacement eta + omega (same as dense)
    ring_nr = theor[..., 9].to(torch.int64).clamp(min=0)         # (N, T)
    eta_post = theor[..., 12]                                     # (N, T)
    omega = theor[..., 6]                                         # (N, T)
    rad_diff = theor[..., 13]                                     # (N, T)

    # ── 1. Bin lookup (single pass, same for every m)
    bin_pos = get_bin_indices(
        ring_nr, eta_post, omega, eta_bin_size, ome_bin_size, n_eta_bins, n_ome_bins,
    )                                                             # (N, T) int64
    bin_pos = bin_pos.clamp(0, max(0, (bin_ndata.numel() // 2) - 1))
    n_per, data_offset = lookup_bin_counts(bin_pos, bin_ndata)    # (N, T) each

    # ── 2. Per-voxel max_n (opt C). One small sync, amortised across the
    # whole compare call.
    actual_max_n_t = n_per.max() if n_per.numel() else torch.zeros((), dtype=n_per.dtype, device=device)
    actual_max_n = int(actual_max_n_t.item())
    if max_n_cap is not None:
        actual_max_n = min(actual_max_n, int(max_n_cap))

    soft_zeros = (
        torch.zeros(N, dtype=dtype, device=device)
        if soft_beam_weight_fn is not None else None
    )

    if actual_max_n == 0:
        return MatchResult(
            n_matches=torch.zeros(N, dtype=torch.int64, device=device),
            n_matches_frac=torch.zeros(N, dtype=torch.int64, device=device),
            n_t_frac=valid.sum(dim=-1).to(torch.int64).clamp_min(1),
            frac_matches=torch.zeros(N, dtype=dtype, device=device),
            avg_ia=torch.zeros(N, dtype=dtype, device=device),
            matched_obs_id=torch.full((N, T), -1, dtype=torch.int64, device=device),
            matched_obs_row=torch.full((N, T), -1, dtype=torch.int64, device=device),
            delta_omega=torch.full((N, T), float("inf"), dtype=dtype, device=device),
            matched=torch.zeros((N, T), dtype=torch.bool, device=device),
            weighted_n_matches=soft_zeros,
            weighted_n_matches_frac=soft_zeros.clone() if soft_zeros is not None else None,
            weighted_frac_matches=soft_zeros.clone() if soft_zeros is not None else None,
        )

    # ── 3. Pre-compute (N, T) constants used in every m iteration
    eta_margin_per = eta_margins[ring_nr.clamp(0, eta_margins.numel() - 1)]      # (N, T)
    if rings_to_reject.numel() > 0:
        skip_radial = (
            ring_nr.unsqueeze(-1) == rings_to_reject.view(1, 1, -1)
        ).any(dim=-1)                                                            # (N, T)
    else:
        skip_radial = torch.zeros((N, T), dtype=torch.bool, device=device)

    # Scan-pos filter prep (PF mode). Computed once if active.
    scan_active = (
        scan_pos_tol_um > 0
        and scan_positions is not None
        and voxel_xy is not None
    )
    if scan_active:
        if obs.shape[-1] < 10:
            raise ValueError(
                "scan-aware mode requires obs with 10 columns (Spots.bin PF layout)."
            )
        v_x = voxel_xy[..., 0].view(N, 1).to(dtype=dtype, device=device)
        v_y = voxel_xy[..., 1].view(N, 1).to(dtype=dtype, device=device)
        omega_rad = omega * DEG2RAD
        s_proj = v_x * torch.sin(omega_rad) + v_y * torch.cos(omega_rad)         # (N, T)
        if obs_scan_nr_int64 is not None:
            obs_scan_idx_full = obs_scan_nr_int64
        else:
            obs_scan_idx_full = obs[..., 9].to(torch.int64)
        scan_pos_arr = scan_positions.to(dtype=dtype, device=device)
    else:
        s_proj = None
        obs_scan_idx_full = None
        scan_pos_arr = None

    # Obs columns (1-D over n_obs). Match dense-path semantics exactly:
    #   col 2 = omega
    #   col 3 = ringrad (used by radial_pass vs ref_rad)
    #   col 4 = spot_id
    #   col 6 = eta
    #   col 8 = rad_diff (used by rad_ok vs theor.col13 — note: confusingly
    #           also called "obs_rad" in the dense path; semantically it's
    #           the radial-displacement, NOT the ring radius).
    obs_ome_col = obs[..., 2]
    obs_ringrad_col = obs[..., 3]
    obs_id_col = obs[..., 4].to(torch.int64)
    obs_eta_col = obs[..., 6]
    obs_rad_col = obs[..., 8]

    # ── 4. Running state per (N, T) cell
    best_delta_ome = torch.full((N, T), float("inf"), dtype=dtype, device=device)
    best_matched_id = torch.full((N, T), -1, dtype=torch.int64, device=device)
    best_matched_row = torch.full((N, T), -1, dtype=torch.int64, device=device)
    has_match = torch.zeros((N, T), dtype=torch.bool, device=device)
    best_weight: torch.Tensor | None = None
    if soft_beam_weight_fn is not None:
        best_weight = torch.zeros((N, T), dtype=dtype, device=device)

    # ── 5. M-iter loop
    valid_b = valid                                                              # alias for clarity
    n_obs = obs.shape[0]
    bin_data_max = bin_data.numel() - 1
    scan_pos_max_idx = (scan_pos_arr.numel() - 1) if scan_pos_arr is not None else 0

    for m in range(actual_max_n):
        in_bin = (m < n_per) & valid_b                                           # (N, T)
        if not bool(in_bin.any()):
            continue

        rows_m = (data_offset + m).clamp(0, bin_data_max)
        spot_rows_m = bin_data[rows_m].to(torch.int64)                           # (N, T)
        # Safe clamp (n_obs - 1) — masked off by in_bin anyway.
        spot_rows_safe = spot_rows_m.clamp(0, max(0, n_obs - 1))

        cand_ome = obs_ome_col[spot_rows_safe]
        cand_eta = obs_eta_col[spot_rows_safe]
        cand_ringrad = obs_ringrad_col[spot_rows_safe]
        cand_rad = obs_rad_col[spot_rows_safe]
        cand_id = obs_id_col[spot_rows_safe]

        # rad_ok compares theor's rad_diff (col 13) to obs's rad_diff (col 8).
        # radial_pass compares per-tuple ref_rad to obs's ringrad (col 3).
        # Distinct quantities — must NOT collapse to one cand_* tensor.
        rad_ok = (rad_diff - cand_rad).abs() < margin_radial
        eta_ok = (eta_post - cand_eta).abs() < eta_margin_per
        radial_pass = (ref_rad.view(N, 1) - cand_ringrad).abs() < margin_rad
        radial_ok = skip_radial | radial_pass

        ok = in_bin & rad_ok & radial_ok & eta_ok                                # (N, T)

        weight_m = None
        if scan_active:
            cand_scan_idx = obs_scan_idx_full[spot_rows_safe]
            cand_scan_pos = scan_pos_arr[cand_scan_idx.clamp(0, scan_pos_max_idx)]
            diff = (s_proj - cand_scan_pos).abs()
            if soft_beam_weight_fn is not None:
                weight_m = soft_beam_weight_fn(diff).to(dtype=dtype)
                if friedel_symmetric_scan_filter:
                    diff_f = (s_proj + cand_scan_pos).abs()
                    weight_m = torch.maximum(weight_m, soft_beam_weight_fn(diff_f).to(dtype=dtype))
                scan_ok = weight_m > 0
            else:
                scan_ok = diff < scan_pos_tol_um
                if friedel_symmetric_scan_filter:
                    diff_f = (s_proj + cand_scan_pos).abs()
                    scan_ok = scan_ok | (diff_f < scan_pos_tol_um)
            ok = ok & scan_ok

        if not bool(ok.any()):
            continue

        # |Δω| for tie-break — only meaningful where ok.
        diff_ome = (omega - cand_ome).abs()
        better = ok & (diff_ome < best_delta_ome)
        if better.any():
            best_delta_ome = torch.where(better, diff_ome, best_delta_ome)
            best_matched_id = torch.where(better, cand_id, best_matched_id)
            best_matched_row = torch.where(better, spot_rows_safe, best_matched_row)
            if best_weight is not None and weight_m is not None:
                best_weight = torch.where(better, weight_m, best_weight)
        has_match = has_match | ok

    # ── 6. Reductions (same as dense path)
    matched_for_frac = has_match & ~skip_radial
    n_matches = has_match.sum(dim=-1).to(torch.int64)
    n_matches_frac = matched_for_frac.sum(dim=-1).to(torch.int64)
    valid_for_frac = valid & ~skip_radial
    n_t_frac = valid_for_frac.sum(dim=-1).to(torch.int64).clamp_min(1)
    frac = n_matches_frac.to(dtype) / n_t_frac.to(dtype)

    if best_weight is not None:
        weighted_n_matches = (best_weight * has_match.to(dtype)).sum(dim=-1)
        weighted_n_matches_frac = (best_weight * matched_for_frac.to(dtype)).sum(dim=-1)
        weighted_frac_matches = weighted_n_matches_frac / n_t_frac.to(dtype)
    else:
        weighted_n_matches = None
        weighted_n_matches_frac = None
        weighted_frac_matches = None

    if distance is not None and pos is not None:
        avg_ia = _compute_avg_ia(
            theor=theor, obs=obs,
            matched_obs_row=best_matched_row,
            has_match=has_match,
            distance=distance, pos=pos,
        )
    else:
        avg_ia = torch.zeros(N, dtype=dtype, device=device)

    # Sentinel for unmatched (matches dense path final cleanup).
    best_matched_id = torch.where(has_match, best_matched_id, torch.full_like(best_matched_id, -1))
    best_matched_row = torch.where(has_match, best_matched_row, torch.full_like(best_matched_row, -1))
    best_delta_ome = torch.where(has_match, best_delta_ome, torch.full_like(best_delta_ome, float("inf")))

    return MatchResult(
        n_matches=n_matches,
        n_matches_frac=n_matches_frac,
        n_t_frac=n_t_frac,
        frac_matches=frac,
        avg_ia=avg_ia,
        matched_obs_id=best_matched_id,
        matched_obs_row=best_matched_row,
        delta_omega=best_delta_ome,
        matched=has_match,
        weighted_n_matches=weighted_n_matches,
        weighted_n_matches_frac=weighted_n_matches_frac,
        weighted_frac_matches=weighted_frac_matches,
    )


# ---------------------------------------------------------------------------
# Numba CPU-fast path
# ---------------------------------------------------------------------------
#
# Why a separate path: the dense torch implementation allocates an
# ``(N, T, M)`` bool stack and does bulk ops over it. At PF scale (Wenxi:
# N≈54000 per seed, T≈500, M_global=177) this is 4.8 billion cells per
# compare call, even though ~99% of (N, T) cells point to empty bins
# (median bin occupancy = 0.2). torch can't naturally skip empty cells —
# it has to allocate and mask. Numba CAN: a per-cell loop tests
# ``n_per[n, t]`` and continues if it's 0. That's the same algorithm C
# IndexerScanningOMP uses to achieve ~0.7s/voxel single-thread.
#
# GPU path stays torch (already fast on GPU — brute-force parallelism
# absorbs the 99% empty cells trivially). Dispatch is by
# ``theor.device.type == 'cuda'`` in ``compare_spots``.


if _NUMBA_AVAILABLE:

    @njit(parallel=True, cache=True, fastmath=False)
    def _compute_avg_ia_numba_inner(
        theor_y,                  # (N, T) float64
        theor_z,                  # (N, T) float64
        theor_ome,                # (N, T) float64
        matched_obs_row,          # (N, T) int64
        has_match,                # (N, T) bool
        obs_y,                    # (n_obs,) float64
        obs_z,                    # (n_obs,) float64
        obs_ome,                  # (n_obs,) float64
        pos,                      # (N, 3) float64
        distance,                 # scalar
    ):
        """Per-(n, t) cell loop computing mean internal-angle (degrees).

        Mirrors the torch ``_compute_avg_ia`` exactly: for each matched
        (n, t) cell, builds the theor and obs g-vectors (each is a 3-vector
        derived from yl/zl/omega + grain position), takes the cosine of
        their angle, accumulates into a per-row sum + count, returns mean
        per row. Unmatched cells contribute 0/0 (becomes 0 after the
        clamp_min(1) on the denominator).
        """
        DEG2RAD_ = math.pi / 180.0
        RAD2DEG_ = 180.0 / math.pi
        N, T = theor_y.shape
        out = np.zeros(N, dtype=np.float64)

        for n in prange(N):
            cx = pos[n, 0]
            cy = pos[n, 1]
            cz = pos[n, 2]
            sum_ia = 0.0
            n_match = 0
            for t in range(T):
                if not has_match[n, t]:
                    continue
                row = matched_obs_row[n, t]
                if row < 0:
                    continue

                # ── gv1 from theor (xi=distance, yi=theor_y, zi=theor_z, ω=theor_ome[n, t])
                om = theor_ome[n, t] * DEG2RAD_
                co = math.cos(om)
                so = math.sin(om)
                # RotateAroundZ(c, omega): vr = (co*cx - so*cy, so*cx + co*cy, cz)
                vr_x = co * cx - so * cy
                vr_y = so * cx + co * cy
                # Subtract from spot coords
                xi = distance - vr_x
                yi = theor_y[n, t] - vr_y
                zi = theor_z[n, t] - cz
                Linv = 1.0 / math.sqrt(xi * xi + yi * yi + zi * zi + 1e-60)
                xn = xi * Linv
                yn = yi * Linv
                zn = zi * Linv
                # Pre-rotation g-vec from (xn, yn, zn)
                g1r = -1.0 + xn
                g2r = yn
                # Rotate by -omega (co, -so)
                g1_theor = g1r * co + g2r * so
                g2_theor = -g1r * so + g2r * co
                g3_theor = zn

                # ── gv2 from obs (same formula, with obs values)
                om_o = obs_ome[row] * DEG2RAD_
                co_o = math.cos(om_o)
                so_o = math.sin(om_o)
                vr_xo = co_o * cx - so_o * cy
                vr_yo = so_o * cx + co_o * cy
                xi2 = distance - vr_xo
                yi2 = obs_y[row] - vr_yo
                zi2 = obs_z[row] - cz
                Linv2 = 1.0 / math.sqrt(xi2 * xi2 + yi2 * yi2 + zi2 * zi2 + 1e-60)
                xn2 = xi2 * Linv2
                yn2 = yi2 * Linv2
                zn2 = zi2 * Linv2
                g1r2 = -1.0 + xn2
                g2r2 = yn2
                g1_obs = g1r2 * co_o + g2r2 * so_o
                g2_obs = -g1r2 * so_o + g2r2 * co_o
                g3_obs = zn2

                # cos(IA) = dot(g_theor, g_obs) / (|g_theor| * |g_obs|)
                n1 = math.sqrt(g1_theor * g1_theor + g2_theor * g2_theor + g3_theor * g3_theor)
                n2 = math.sqrt(g1_obs * g1_obs + g2_obs * g2_obs + g3_obs * g3_obs)
                if n1 < 1e-30 or n2 < 1e-30:
                    continue
                cos_ia = (g1_theor * g1_obs + g2_theor * g2_obs + g3_theor * g3_obs) / (n1 * n2)
                if cos_ia > 1.0:
                    cos_ia = 1.0
                elif cos_ia < -1.0:
                    cos_ia = -1.0
                ia = math.acos(cos_ia) * RAD2DEG_
                sum_ia += abs(ia)
                n_match += 1

            if n_match > 0:
                out[n] = sum_ia / n_match
        return out

    @njit(parallel=True, cache=True, fastmath=False)
    def _compare_spots_numba_inner(
        # Theor (N, T) views
        ring_nr,                  # (N, T) int64
        eta_post,                 # (N, T) float64
        omega,                    # (N, T) float64
        rad_diff,                 # (N, T) float64
        valid,                    # (N, T) bool
        # Bin lookups
        n_per,                    # (N, T) int64
        data_offset,              # (N, T) int64
        bin_data,                 # (n_bin_data,) int32
        # Obs columns
        obs_ome,                  # (n_obs,) float64
        obs_eta,                  # (n_obs,) float64
        obs_ringrad,              # (n_obs,) float64
        obs_rad,                  # (n_obs,) float64
        obs_id,                   # (n_obs,) int64
        # Margins
        eta_margin_per,           # (N, T) float64 — per-ring eta margin
        margin_radial,            # scalar
        margin_rad,               # scalar
        skip_radial,              # (N, T) bool
        ref_rad,                  # (N,) float64
        # Scan-aware (PF). When scan_active=False the rest of these are unused.
        scan_active,              # bool
        s_proj,                   # (N, T) float64; ignored if not scan_active
        obs_scan_idx,             # (n_obs,) int64; ignored if not scan_active
        scan_pos_arr,             # (n_scans,) float64; ignored if not scan_active
        scan_pos_tol,             # scalar
        friedel_sym,              # bool
    ):
        """Per-(n, t) cell loop with bin early-exit. Returns
        ``(best_delta_ome, best_matched_id, best_matched_row, has_match)``.

        Mirrors ``CalcCompareSpots`` from IndexerOMP.c:1700-1830 with the
        Friedel-symmetric scan filter from IndexerScanningOMP.c.
        """
        N, T = omega.shape
        best_delta_ome = np.full((N, T), np.inf, dtype=np.float64)
        best_matched_id = np.full((N, T), -1, dtype=np.int64)
        best_matched_row = np.full((N, T), -1, dtype=np.int64)
        has_match = np.zeros((N, T), dtype=np.bool_)

        n_bin = bin_data.shape[0]
        n_obs = obs_ome.shape[0]
        n_scans_pos = scan_pos_arr.shape[0]

        for n in prange(N):
            ref_rad_n = ref_rad[n]
            for t in range(T):
                if not valid[n, t]:
                    continue
                n_p = n_per[n, t]
                if n_p == 0:
                    # 99% of cells on PF data — primary early-exit.
                    continue
                offset = data_offset[n, t]
                rad_diff_nt = rad_diff[n, t]
                eta_post_nt = eta_post[n, t]
                omega_nt = omega[n, t]
                eta_marg_nt = eta_margin_per[n, t]
                skip_rad_nt = skip_radial[n, t]
                s_proj_nt = s_proj[n, t] if scan_active else 0.0

                local_best_dome = np.inf
                local_best_id = np.int64(-1)
                local_best_row = np.int64(-1)
                local_has_match = False

                for m in range(n_p):
                    row_idx = offset + m
                    if row_idx < 0 or row_idx >= n_bin:
                        continue
                    row = np.int64(bin_data[row_idx])
                    if row < 0 or row >= n_obs:
                        continue

                    # Tolerance gates (C IndexerOMP.c order).
                    if abs(rad_diff_nt - obs_rad[row]) >= margin_radial:
                        continue
                    if abs(eta_post_nt - obs_eta[row]) >= eta_marg_nt:
                        continue
                    if not skip_rad_nt:
                        if abs(ref_rad_n - obs_ringrad[row]) >= margin_rad:
                            continue

                    # Scan-position filter (PF mode only).
                    if scan_active:
                        scan_idx = obs_scan_idx[row]
                        if scan_idx < 0:
                            scan_idx = 0
                        elif scan_idx >= n_scans_pos:
                            scan_idx = n_scans_pos - 1
                        scan_pos = scan_pos_arr[scan_idx]
                        diff = abs(s_proj_nt - scan_pos)
                        if diff >= scan_pos_tol:
                            if friedel_sym:
                                diff_f = abs(s_proj_nt + scan_pos)
                                if diff_f >= scan_pos_tol:
                                    continue
                            else:
                                continue

                    # Matched. Tie-break by min |Δω|.
                    d_ome = abs(omega_nt - obs_ome[row])
                    if d_ome < local_best_dome:
                        local_best_dome = d_ome
                        local_best_id = obs_id[row]
                        local_best_row = row
                    local_has_match = True

                if local_has_match:
                    best_delta_ome[n, t] = local_best_dome
                    best_matched_id[n, t] = local_best_id
                    best_matched_row[n, t] = local_best_row
                    has_match[n, t] = True

        return best_delta_ome, best_matched_id, best_matched_row, has_match

else:  # pragma: no cover — numba absent
    def _compare_spots_numba_inner(*args, **kwargs):  # type: ignore
        raise ImportError(
            "numba is required for the CPU fast path. Install with "
            "`pip install numba` or use compare_spots() with strategy='dense'."
        )


def _compare_spots_numba(
    theor: torch.Tensor,
    valid: torch.Tensor,
    obs: torch.Tensor,
    bin_data: torch.Tensor,
    bin_ndata: torch.Tensor,
    *,
    ref_rad: torch.Tensor,
    margin_rad: float,
    margin_radial: float,
    eta_margins: torch.Tensor,
    ome_margins: torch.Tensor,
    eta_bin_size: float,
    ome_bin_size: float,
    n_eta_bins: int,
    n_ome_bins: int,
    rings_to_reject: torch.Tensor,
    distance: float | None,
    pos: torch.Tensor | None,
    scan_positions: torch.Tensor | None = None,
    voxel_xy: torch.Tensor | None = None,
    scan_pos_tol_um: float = 0.0,
    friedel_symmetric_scan_filter: bool = False,
    obs_scan_nr_int64: torch.Tensor | None = None,
) -> MatchResult:
    """CPU-fast compare_spots via a numba per-cell loop.

    Falls back to the torch m-iter path if numba is not installed.
    """
    if not _NUMBA_AVAILABLE:
        return _compare_spots_m_iter(
            theor=theor, valid=valid, obs=obs,
            bin_data=bin_data, bin_ndata=bin_ndata,
            ref_rad=ref_rad,
            margin_rad=margin_rad, margin_radial=margin_radial,
            eta_margins=eta_margins, ome_margins=ome_margins,
            eta_bin_size=eta_bin_size, ome_bin_size=ome_bin_size,
            n_eta_bins=n_eta_bins, n_ome_bins=n_ome_bins,
            rings_to_reject=rings_to_reject,
            distance=distance, pos=pos,
            scan_positions=scan_positions, voxel_xy=voxel_xy,
            scan_pos_tol_um=scan_pos_tol_um,
            friedel_symmetric_scan_filter=friedel_symmetric_scan_filter,
            obs_scan_nr_int64=obs_scan_nr_int64,
            soft_beam_weight_fn=None,
        )

    device = theor.device
    dtype = theor.dtype
    N, T, _ = theor.shape

    # ── 0. Per-theor-spot features (same as dense path)
    ring_nr_t = theor[..., 9].to(torch.int64).clamp(min=0)
    eta_post_t = theor[..., 12]
    omega_t = theor[..., 6]
    rad_diff_t = theor[..., 13]

    # ── 1. Bin lookup (torch — vectorised; no per-cell math needed)
    bin_pos = get_bin_indices(
        ring_nr_t, eta_post_t, omega_t,
        eta_bin_size, ome_bin_size, n_eta_bins, n_ome_bins,
    )
    bin_pos = bin_pos.clamp(0, max(0, (bin_ndata.numel() // 2) - 1))
    n_per_t, data_offset_t = lookup_bin_counts(bin_pos, bin_ndata)

    # ── 2. Pre-compute per-cell static tensors
    eta_margin_per_t = eta_margins[ring_nr_t.clamp(0, eta_margins.numel() - 1)]
    if rings_to_reject.numel() > 0:
        skip_radial_t = (
            ring_nr_t.unsqueeze(-1) == rings_to_reject.view(1, 1, -1)
        ).any(dim=-1)
    else:
        skip_radial_t = torch.zeros((N, T), dtype=torch.bool, device=device)

    # ── 3. Scan-aware prep (PF mode)
    scan_active = (
        scan_pos_tol_um > 0
        and scan_positions is not None
        and voxel_xy is not None
    )
    if scan_active:
        if obs.shape[-1] < 10:
            raise ValueError(
                "scan-aware mode requires obs with 10 columns (Spots.bin PF)."
            )
        v_x = voxel_xy[..., 0].view(N, 1).to(dtype=dtype, device=device)
        v_y = voxel_xy[..., 1].view(N, 1).to(dtype=dtype, device=device)
        omega_rad_t = omega_t * DEG2RAD
        s_proj_t = v_x * torch.sin(omega_rad_t) + v_y * torch.cos(omega_rad_t)
        obs_scan_idx_t = (
            obs_scan_nr_int64
            if obs_scan_nr_int64 is not None
            else obs[..., 9].to(torch.int64)
        )
        scan_pos_arr_t = scan_positions.to(dtype=dtype, device=device)
    else:
        s_proj_t = torch.zeros((N, T), dtype=dtype, device=device)
        obs_scan_idx_t = torch.zeros(1, dtype=torch.int64, device=device)
        scan_pos_arr_t = torch.zeros(1, dtype=dtype, device=device)

    # ── 4. Marshal to contiguous numpy (np.ascontiguousarray copies if needed)
    def _np_f64(t):
        return np.ascontiguousarray(t.detach().cpu().numpy().astype(np.float64, copy=False))

    def _np_i64(t):
        return np.ascontiguousarray(t.detach().cpu().numpy().astype(np.int64, copy=False))

    def _np_bool(t):
        return np.ascontiguousarray(t.detach().cpu().numpy().astype(np.bool_, copy=False))

    ring_nr_np = _np_i64(ring_nr_t)
    eta_post_np = _np_f64(eta_post_t)
    omega_np = _np_f64(omega_t)
    rad_diff_np = _np_f64(rad_diff_t)
    valid_np = _np_bool(valid)
    n_per_np = _np_i64(n_per_t)
    data_offset_np = _np_i64(data_offset_t)
    bin_data_np = np.ascontiguousarray(bin_data.detach().cpu().numpy().astype(np.int32, copy=False))
    obs_np = obs.detach().cpu().numpy()
    obs_ome_np = np.ascontiguousarray(obs_np[..., 2].astype(np.float64, copy=False))
    obs_eta_np = np.ascontiguousarray(obs_np[..., 6].astype(np.float64, copy=False))
    obs_ringrad_np = np.ascontiguousarray(obs_np[..., 3].astype(np.float64, copy=False))
    obs_rad_np = np.ascontiguousarray(obs_np[..., 8].astype(np.float64, copy=False))
    obs_id_np = np.ascontiguousarray(obs_np[..., 4].astype(np.int64, copy=False))
    eta_margin_per_np = _np_f64(eta_margin_per_t)
    skip_radial_np = _np_bool(skip_radial_t)
    ref_rad_np = _np_f64(ref_rad)
    s_proj_np = _np_f64(s_proj_t)
    obs_scan_idx_np = _np_i64(obs_scan_idx_t)
    scan_pos_arr_np = _np_f64(scan_pos_arr_t)

    # ── 5. Numba kernel
    best_delta_ome_np, best_matched_id_np, best_matched_row_np, has_match_np = (
        _compare_spots_numba_inner(
            ring_nr_np, eta_post_np, omega_np, rad_diff_np, valid_np,
            n_per_np, data_offset_np, bin_data_np,
            obs_ome_np, obs_eta_np, obs_ringrad_np, obs_rad_np, obs_id_np,
            eta_margin_per_np,
            float(margin_radial), float(margin_rad),
            skip_radial_np,
            ref_rad_np,
            bool(scan_active),
            s_proj_np,
            obs_scan_idx_np,
            scan_pos_arr_np,
            float(scan_pos_tol_um),
            bool(friedel_symmetric_scan_filter),
        )
    )

    # ── 6. Back to torch + compute reductions (same as dense path)
    best_delta_ome = torch.from_numpy(best_delta_ome_np).to(device=device, dtype=dtype)
    best_matched_id = torch.from_numpy(best_matched_id_np).to(device=device)
    best_matched_row = torch.from_numpy(best_matched_row_np).to(device=device)
    has_match = torch.from_numpy(has_match_np).to(device=device)

    matched_for_frac = has_match & ~skip_radial_t
    n_matches = has_match.sum(dim=-1).to(torch.int64)
    n_matches_frac = matched_for_frac.sum(dim=-1).to(torch.int64)
    valid_for_frac = valid & ~skip_radial_t
    n_t_frac = valid_for_frac.sum(dim=-1).to(torch.int64).clamp_min(1)
    frac = n_matches_frac.to(dtype) / n_t_frac.to(dtype)

    if distance is not None and pos is not None:
        # Numba avg_ia: per-(n, t) scalar work. Was the largest remaining
        # torch hot spot (~90 ms/call from _spot_to_gv_pos + torch.stack).
        theor_y_np = np.ascontiguousarray(theor[..., 10].detach().cpu().numpy().astype(np.float64, copy=False))
        theor_z_np = np.ascontiguousarray(theor[..., 11].detach().cpu().numpy().astype(np.float64, copy=False))
        theor_ome_np = np.ascontiguousarray(theor[..., 6].detach().cpu().numpy().astype(np.float64, copy=False))
        obs_y_np = np.ascontiguousarray(obs_np[..., 0].astype(np.float64, copy=False))
        obs_z_np = np.ascontiguousarray(obs_np[..., 1].astype(np.float64, copy=False))
        pos_np = np.ascontiguousarray(pos.detach().cpu().numpy().astype(np.float64, copy=False))
        avg_ia_np = _compute_avg_ia_numba_inner(
            theor_y_np, theor_z_np, theor_ome_np,
            best_matched_row_np, has_match_np,
            obs_y_np, obs_z_np, obs_ome_np,
            pos_np, float(distance),
        )
        avg_ia = torch.from_numpy(avg_ia_np).to(device=device, dtype=dtype)
    else:
        avg_ia = torch.zeros(N, dtype=dtype, device=device)

    return MatchResult(
        n_matches=n_matches,
        n_matches_frac=n_matches_frac,
        n_t_frac=n_t_frac,
        frac_matches=frac,
        avg_ia=avg_ia,
        matched_obs_id=best_matched_id,
        matched_obs_row=best_matched_row,
        delta_omega=best_delta_ome,
        matched=has_match,
        weighted_n_matches=None,
        weighted_n_matches_frac=None,
        weighted_frac_matches=None,
    )

