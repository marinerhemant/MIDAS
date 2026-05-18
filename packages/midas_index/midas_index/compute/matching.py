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

import torch

from .binning import get_bin_indices, lookup_bin_counts

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


@dataclass
class MatchResult:
    """Per-evaluation-tuple match outcome."""

    n_matches: torch.Tensor          # (N,) int64 — total matched theor spots
    n_matches_frac: torch.Tensor     # (N,) int64 — matches excluding rings_to_reject (denom of frac)
    n_t_frac: torch.Tensor           # (N,) int64 — valid theor spots excluding rings_to_reject
    frac_matches: torch.Tensor       # (N,) float — n_matches_frac / n_t_frac
    avg_ia: torch.Tensor             # (N,) float — IA average; placeholder for v0.1.0
    matched_obs_id: torch.Tensor     # (N, T) int64 — best obs spot id per theor spot, -1 if none
    matched_obs_row: torch.Tensor    # (N, T) int64 — row index in `obs` for each match, -1 if none
    delta_omega: torch.Tensor        # (N, T) float — |Δomega| for the best match, +inf if none
    matched: torch.Tensor            # (N, T) bool — match found per theor spot


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

    if free_bytes is None:
        if theor.device.type != "cuda":
            return "dense", _JAGGED_CHUNK_MAX
        try:
            free_bytes, _total = torch.cuda.mem_get_info(theor.device)
        except Exception:
            return "dense", _JAGGED_CHUNK_MAX

    bytes_per_cell = _per_cell_bytes(theor.dtype)
    budget = max(1, int(free_bytes * safety))
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
        scan_ok = diff < scan_pos_tol_um
        if friedel_symmetric_scan_filter:
            # Friedel pair: matching spot may appear at +scan or -scan offset.
            diff_friedel = (s_proj.unsqueeze(-1) + cand_scan_pos).abs()
            scan_ok = scan_ok | (diff_friedel < scan_pos_tol_um)
        ok = ok & scan_ok                                                       # (N, T, M)

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
            )
        )
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

