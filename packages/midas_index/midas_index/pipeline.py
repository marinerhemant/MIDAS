"""High-level pipeline orchestration.

Mirrors `FF_HEDM/src/IndexerOMP.c::main` flow:

  1. Parse argv -> (param_file, block_nr, n_blocks, n_spots_to_index, num_procs)
  2. ReadParams(param_file)                              -> IndexerParams
  3. read hkls.csv                                        -> hkls table
  4. read Bins.bin / nData.bin                            -> binned spot index
  5. read Spots.bin                                       -> ObsSpotsLab
  6. if isGrainsInput: build SpotsToIndex.csv from Grains.csv (mode A)
     load SpotsToIndex.csv                                -> seed spot IDs
  7. compute startRowNr, endRowNr from block sharding
  8. for spot_id in seeds[startRowNr:endRowNr]:
         process_seed(...)                                <- the per-seed kernel
         write best result to IndexBest.bin
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from .compute import (
    binning,
    forward_adapter,
    matching,
    orientation_grid,
    position_grid,
    reduce as reduce_,
    seeds as seeds_module,
)
from .params import IndexerParams
from .result import IndexerResult, SeedResult

if TYPE_CHECKING:
    pass

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

_LOG = logging.getLogger(__name__)

# Warn at most once per (chunk_size bucket) per process so a chunked dataset
# doesn't spam the log on every group call.
_jagged_warned: set[int] = set()


def _log_jagged_fallback(theor: torch.Tensor, max_n_cap: int,
                          chunk_size: int) -> None:
    """One-time INFO when the auto picker switches to the chunked path."""
    if chunk_size in _jagged_warned:
        return
    _jagged_warned.add(chunk_size)
    N = int(theor.shape[0]) if theor.ndim >= 1 else 0
    T = int(theor.shape[1]) if theor.ndim >= 2 else 0
    _LOG.info(
        "compare_spots: auto strategy=jagged chunk_size=%d "
        "(N=%d, T=%d, max_n_cap=%d, dtype=%s) — dense path would exceed "
        "memory budget",
        chunk_size, N, T, max_n_cap, str(theor.dtype),
    )


# ---------------------------------------------------------------------------
# Pre-computed indexer context (per-Indexer; reused across seeds)
# ---------------------------------------------------------------------------


class IndexerContext:
    """Shared, immutable context built once per `Indexer.run`.

    Holds device-resident tensors for hkls, observed spots, bin index, margin
    LUTs, and the forward adapter. One per Indexer invocation.
    """

    def __init__(
        self,
        params: IndexerParams,
        hkls_real: np.ndarray | torch.Tensor,
        hkls_int: np.ndarray | torch.Tensor,
        obs: np.ndarray | torch.Tensor,
        bin_data: np.ndarray | torch.Tensor,
        bin_ndata: np.ndarray | torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.params = params
        self.device = device
        self.dtype = dtype

        self.hkls_real = torch.as_tensor(np.asarray(hkls_real), device=device, dtype=dtype)
        self.hkls_int = torch.as_tensor(np.asarray(hkls_int), device=device, dtype=torch.long)
        self.obs = torch.as_tensor(np.asarray(obs), device=device, dtype=dtype)
        self.bin_data = torch.as_tensor(np.asarray(bin_data), device=device, dtype=torch.int32)
        self.bin_ndata = torch.as_tensor(np.asarray(bin_ndata), device=device, dtype=torch.int32)

        # Per-ring cached HKL row index. For seed processing we need to find
        # the HKL Cartesian (g1, g2, g3) for a given ring number. C code stores
        # this in `RingHKL[ringnr]` (last-seen wins per IndexerOMP.c:2202-2205).
        # We replicate by iterating in order; later rings overwrite if duplicate.
        self.ring_hkl: dict[int, torch.Tensor] = {}
        self.ring_ttheta: dict[int, float] = {}
        for i in range(self.hkls_real.shape[0]):
            rn = int(self.hkls_real[i, 3].item())
            self.ring_hkl[rn] = self.hkls_real[i, :3].clone()
            self.ring_ttheta[rn] = float(self.hkls_real[i, 5].item() * 2 * RAD2DEG)

        # Per-ring integer (h,k,l) cached as a Python tuple so the per-seed
        # loop avoids one `.tolist()`-shaped GPU sync per seed. C indexer also
        # uses a static int table (HKLints) keyed on ring nr.
        hkls_int_cpu = self.hkls_int.cpu().numpy()
        self.ring_hkl_int: dict[int, tuple[int, int, int]] = {}
        for i in range(hkls_int_cpu.shape[0]):
            rn = int(hkls_int_cpu[i, 3])
            self.ring_hkl_int[rn] = (
                int(hkls_int_cpu[i, 0]),
                int(hkls_int_cpu[i, 1]),
                int(hkls_int_cpu[i, 2]),
            )

        # Bin geometry
        self.n_eta_bins = int(math.ceil(360.0 / params.EtaBinSize))
        self.n_ome_bins = int(math.ceil(360.0 / params.OmeBinSize))

        # Margin LUTs
        self.eta_margins = matching.build_eta_margins(
            ring_radii=params.RingRadii,
            margin_eta=params.MarginEta,
            stepsize_orient_deg=params.StepsizeOrient,
            device=device, dtype=dtype,
        )
        self.ome_margins = matching.build_ome_margins(
            margin_ome=params.MarginOme,
            stepsize_orient_deg=params.StepsizeOrient,
            device=device, dtype=dtype,
        )

        self.rings_to_reject = torch.tensor(
            params.RingsToReject if params.RingsToReject else [],
            device=device, dtype=torch.int64,
        )

        # CPU mirror of the obs table. Used by `_setup_group_cpu` to do the
        # per-group obs-id lookup without touching GPU memory — critical for
        # the prefetch overlap, since any GPU access from CPU code (e.g.
        # `.cpu()`) would force a sync on the in-flight GPU stream.
        self.obs_cpu = self.obs.detach().cpu().numpy()

        # Pre-cast obs columns 4 (spot_id) and 9 (scan_nr when PF) to int64
        # once. ``compare_spots`` reaches for these on every call, and the
        # naive ``obs[..., 4].to(torch.int64)`` allocates a new n_obs tensor
        # each time. Cached on the context so the scan-aware path (in
        # particular) doesn't pay this per-voxel.
        self.obs_id_int64 = self.obs[..., 4].to(torch.int64).contiguous()
        if self.obs.shape[-1] >= 10:
            self.obs_scan_nr_int64 = self.obs[..., 9].to(torch.int64).contiguous()
        else:
            self.obs_scan_nr_int64 = None

        # Global max bin occupancy from nData.bin (interleaved as [count, off]).
        # Precomputed once so `compare_spots` doesn't need a per-call
        # `n_per.max().item()` sync — the dense candidate gather is sized to
        # this fixed cap. Empirically the bin distribution is heavily skewed
        # (median ~1, max ~5) so the wasted memory is negligible.
        if self.bin_ndata.numel() >= 2:
            counts_view = self.bin_ndata.view(-1, 2)[:, 0]
            self.bin_max_count = int(counts_view.max().item())
        else:
            self.bin_max_count = 0

        # Forward adapter (constructs HEDMForwardModel internally)
        self.adapter = forward_adapter.IndexerForwardAdapter(
            params=params,
            hkls_real=self.hkls_real,
            hkls_int=self.hkls_int,
            device=device,
            dtype=dtype,
        )

        # --- Scan-aware (pf-HEDM) state — default-off, no FF impact ---
        # Populated by ``Indexer.run_scanning()`` per voxel; FF runs leave
        # these as None / 0 and compare_spots receives no extra kwargs.
        self.scan_positions: torch.Tensor | None = None
        self.current_voxel_xy: torch.Tensor | None = None
        self.scan_pos_tol_um: float = float(params.scan_pos_tol_um)
        self.friedel_symmetric_scan_filter: bool = bool(
            params.friedel_symmetric_scan_filter
        )
        # Soft beam attribution (P6/P8 of the V-map plan).  When set, the
        # scan_kwargs() dict passes it to compare_spots, which populates
        # the optional weighted_* fields on MatchResult.  None ⇒ legacy
        # binary scan_pos_tol_um filter (back-compat default).
        self.soft_beam_weight_fn = None  # type: ignore[var-annotated]

    def scan_kwargs(self, n_tuples: int) -> dict:
        """Return scan-aware kwargs for :func:`matching.compare_spots`.

        Empty in FF mode (``scan_positions`` is None) → caller's existing
        compare_spots call is byte-identical to today. In scan mode,
        returns the per-tuple voxel_xy + scan_positions + tol + Friedel
        flag.
        """
        if self.scan_positions is None or self.current_voxel_xy is None:
            return {}
        out = {
            "scan_positions": self.scan_positions,
            "voxel_xy": self.current_voxel_xy.view(1, 2).expand(n_tuples, 2),
            "scan_pos_tol_um": self.scan_pos_tol_um,
            "friedel_symmetric_scan_filter": self.friedel_symmetric_scan_filter,
            # Pre-cached int64 ScanNr column so compare_spots' per-call
            # ``obs[..., 9].to(int64)`` cast becomes a noop.
            "obs_scan_nr_int64": self.obs_scan_nr_int64,
        }
        if self.soft_beam_weight_fn is not None:
            out["soft_beam_weight_fn"] = self.soft_beam_weight_fn
        return out

    def find_obs_row_by_id(self, spot_id: int) -> int:
        """Return the row index of the observed spot whose column 4 equals `spot_id`."""
        col = self.obs[:, 4]
        idx = torch.where(col == float(spot_id))[0]
        if idx.numel() == 0:
            return -1
        return int(idx[0].item())


# ---------------------------------------------------------------------------
# Per-seed kernel
# ---------------------------------------------------------------------------


def _normalize(v: torch.Tensor) -> torch.Tensor:
    return v / torch.linalg.vector_norm(v).clamp_min(1e-30)


def _spot_to_gv(
    distance: float, y0: float, z0: float, omega_deg: float,
    *, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """G-vector for a Bragg spot at (Lsd, y0, z0) measured at omega.

    Mirrors `MakeUnitLength` + `spot_to_gv` from `FF_HEDM/src/IndexerOMP.c:726`,
    used as the seed's `hklnormal` in GenerateCandidateOrientationsF (line 1849).
    """
    L = math.sqrt(distance * distance + y0 * y0 + z0 * z0)
    xn = distance / L
    yn = y0 / L
    zn = z0 / L
    g1r = -1.0 + xn
    g2r = yn
    co = math.cos(-omega_deg * DEG2RAD)
    so = math.sin(-omega_deg * DEG2RAD)
    g1 = g1r * co - g2r * so
    g2 = g1r * so + g2r * co
    g3 = zn
    return torch.tensor([g1, g2, g3], device=device, dtype=dtype)


def _spot_to_gv_batch(
    distance: float,
    y0_all: torch.Tensor,    # (B,)
    z0_all: torch.Tensor,    # (B,)
    omega_deg: float,
) -> torch.Tensor:
    """Batched plane normals for B (y0, z0) candidates of one seed.

    Returns (B, 3) on the same device/dtype as y0_all.
    """
    L = torch.sqrt(distance * distance + y0_all * y0_all + z0_all * z0_all)
    xn = distance / L
    yn = y0_all / L
    zn = z0_all / L
    g1r = -1.0 + xn
    g2r = yn
    co = math.cos(-omega_deg * DEG2RAD)
    so = math.sin(-omega_deg * DEG2RAD)
    g1 = g1r * co - g2r * so
    g2 = g1r * so + g2r * co
    g3 = zn
    return torch.stack([g1, g2, g3], dim=-1)


def process_seed(
    spot_id: int, ctx: IndexerContext,
) -> SeedResult | None:
    """Run the full per-seed indexing kernel for a single spot ID."""
    p = ctx.params
    obs_row = ctx.find_obs_row_by_id(spot_id)
    if obs_row < 0:
        return None

    seed_obs = ctx.obs[obs_row]                            # (9,)
    ys = float(seed_obs[0].item())
    zs = float(seed_obs[1].item())
    seed_omega = float(seed_obs[2].item())
    seed_ring_rad = float(seed_obs[3].item())              # observed radial position
    seed_eta = float(seed_obs[6].item())
    seed_ring_nr = int(seed_obs[5].item())

    if seed_ring_nr not in ctx.ring_hkl:
        return None
    hkl = ctx.ring_hkl[seed_ring_nr]                       # (3,)
    ring_rad_user = p.get_ring_radius(seed_ring_nr)        # canonical ring radius from paramstest
    ring_rad = ring_rad_user if ring_rad_user > 0 else seed_ring_rad
    ttheta = ctx.ring_ttheta[seed_ring_nr]

    # 1. Seed candidates (y0, z0)
    if p.UseFriedelPairs == 1:
        seed_yz = seeds_module.generate_ideal_spots_friedel(
            ys=ys, zs=zs, ttheta_deg=ttheta, eta_deg=seed_eta, omega_deg=seed_omega,
            ring_nr=seed_ring_nr, ring_rad=ring_rad,
            rsample=p.Rsample, hbeam=p.Hbeam,
            ome_tol=p.MarginOme, radius_tol=p.MarginRadial,
            obs_spots=ctx.obs, device=ctx.device, dtype=ctx.dtype,
        )
        # C parity: when the strict friedel-pair search finds no obs
        # partner (e.g., partner was unobserved or fell outside the
        # geometric band at eta near 90 ± eta_equator), fall back to
        # the "mixed" search that doesn't require an obs partner.
        # See IndexerOMP.c:1808-1822 — `if (nPlaneNormals == 0)` branch.
        if seed_yz.shape[0] == 0:
            seed_yz = seeds_module.generate_ideal_spots_friedel_mixed(
                ys=ys, zs=zs, ttheta_deg=ttheta, eta_deg=seed_eta,
                omega_deg=seed_omega,
                ring_nr=seed_ring_nr, ring_rad=ring_rad, lsd=p.Distance,
                rsample=p.Rsample, hbeam=p.Hbeam,
                step_size_pos=p.StepsizePos,
                ome_tol=p.MarginOme, radial_tol=p.MarginRadial,
                eta_tol_um=p.MarginEta,
                obs_spots=ctx.obs, device=ctx.device, dtype=ctx.dtype,
            )
    elif p.UseFriedelPairs == 2:
        seed_yz = seeds_module.generate_ideal_spots_friedel_mixed(
            ys=ys, zs=zs, ttheta_deg=ttheta, eta_deg=seed_eta, omega_deg=seed_omega,
            ring_nr=seed_ring_nr, ring_rad=ring_rad, lsd=p.Distance,
            rsample=p.Rsample, hbeam=p.Hbeam,
            step_size_pos=p.StepsizePos,
            ome_tol=p.MarginOme, radial_tol=p.MarginRad, eta_tol_um=p.MarginEta,
            obs_spots=ctx.obs, device=ctx.device, dtype=ctx.dtype,
        )
    else:
        seed_yz = seeds_module.generate_ideal_spots(
            ys=ys, zs=zs, ttheta_deg=ttheta, eta_deg=seed_eta,
            ring_rad=ring_rad, rsample=p.Rsample, hbeam=p.Hbeam,
            step_size=p.StepsizePos, device=ctx.device, dtype=ctx.dtype,
        )
    if seed_yz.shape[0] == 0:
        return None

    # 2. Build the global cartesian product of (y0_z0) × orientations × positions
    #    in one shot, so the forward sim and match run as a single batched
    #    tensor op (instead of one Python iteration per y0_z0 candidate).
    hkl_int_for_ring = ctx.ring_hkl_int[seed_ring_nr]

    R_chunks: list[torch.Tensor] = []
    pos_chunks: list[torch.Tensor] = []
    for k in range(seed_yz.shape[0]):
        y0 = float(seed_yz[k, 0].item())
        z0 = float(seed_yz[k, 1].item())
        plane_normal = _spot_to_gv(
            p.Distance, y0, z0, seed_omega,
            device=ctx.device, dtype=ctx.dtype,
        )
        Rs = orientation_grid.generate_candidate_orientations(
            hkl=hkl, plane_normal=plane_normal,
            stepsize_orient_deg=p.StepsizeOrient,
            ring_nr=seed_ring_nr, space_group=p.SpaceGroup,
            hkl_int=hkl_int_for_ring, abcabg=p.LatticeConstant,
        )
        if Rs.shape[0] == 0:
            continue
        positions_k, _ = position_grid.build_position_grid(
            seed_y0=torch.tensor([y0], device=ctx.device, dtype=ctx.dtype),
            seed_z0=torch.tensor([z0], device=ctx.device, dtype=ctx.dtype),
            ys=ys, zs=zs, omega_deg=seed_omega,
            distance=p.Distance, r_sample=p.Rsample, step_size=p.StepsizePos,
            h_beam=p.Hbeam,
        )
        if positions_k.shape[0] == 0:
            continue

        n_or = Rs.shape[0]
        n_pos = positions_k.shape[0]
        N_k = n_or * n_pos
        R_chunks.append(
            Rs.unsqueeze(1).expand(n_or, n_pos, 3, 3).reshape(N_k, 3, 3)
        )
        pos_chunks.append(
            positions_k.unsqueeze(0).expand(n_or, n_pos, 3).reshape(N_k, 3)
        )

    if not R_chunks:
        return None

    R_all = torch.cat(R_chunks, dim=0)            # (N_total, 3, 3)
    pos_all = torch.cat(pos_chunks, dim=0)        # (N_total, 3)
    N = R_all.shape[0]

    # 3. Single batched forward + match across all candidate tuples.
    theor, valid = ctx.adapter.simulate(R_all, pos_all, lattice=None)
    # Match-time radial check (C IndexerOMP.c:1797 + 502): the comparator is
    # |seed_obs[col 3] - cand_obs[col 3]| < MarginRad, so `ref_rad` must be
    # the *seed's* observed radial offset, NOT the full ring radius from
    # paramstest (the latter is only correct for friedel-pair seed search).
    # Using `ring_rad` here breaks matching: full ring radii (~10^5 um)
    # never come within MarginRad of obs col 3 (~10^2 um).
    ref_rad = torch.full((N,), seed_ring_rad, device=ctx.device, dtype=ctx.dtype)
    result = matching.compare_spots(
        theor=theor, valid=valid, obs=ctx.obs,
        bin_data=ctx.bin_data, bin_ndata=ctx.bin_ndata,
        ref_rad=ref_rad,
        margin_rad=p.MarginRad, margin_radial=p.MarginRadial,
        eta_margins=ctx.eta_margins, ome_margins=ctx.ome_margins,
        eta_bin_size=p.EtaBinSize, ome_bin_size=p.OmeBinSize,
        n_eta_bins=ctx.n_eta_bins, n_ome_bins=ctx.n_ome_bins,
        rings_to_reject=ctx.rings_to_reject,
        distance=p.Distance, pos=pos_all,
        max_n_cap=ctx.bin_max_count,
        **ctx.scan_kwargs(N),
    )

    # 4. Reduce: per-seed best tuple via packed-score argmax.
    keys = reduce_.pack_score(result.frac_matches, result.avg_ia)
    if keys.numel() == 0:
        return None
    idx = int(reduce_.best_tuple(keys).item())

    # 5. Compose SeedResult for the winning tuple.
    n_t = int(valid[idx].sum().item())
    n_t_frac = n_t
    if ctx.rings_to_reject.numel() > 0:
        in_reject = (
            theor[idx, :, 9].long().unsqueeze(-1) == ctx.rings_to_reject
        ).any(dim=-1)
        n_t_frac = int((valid[idx] & ~in_reject).sum().item())
    # Compact (matched-only) layout — mirrors C IndexerOMP.c::WriteBestMatchBin
    # (line 1635-1640): C writes ONE row per MATCHED theor spot, not per
    # theor spot with zeros for unmatched. Downstream consumers
    # (midas-fit-grain) read the first n_observed pairs and expect every
    # one to carry a valid spot ID; the older `(T, 2)` layout interleaved
    # zeros for unmatched theor spots and broke the refiner.
    obs_ids_for_best = result.matched_obs_id[idx]
    delta_for_best = result.delta_omega[idx]
    matched_mask = result.matched[idx]
    pairs = torch.stack(
        [obs_ids_for_best[matched_mask].to(ctx.dtype),
         delta_for_best[matched_mask].to(ctx.dtype)],
        dim=-1,
    )

    weighted_n = (
        float(result.weighted_n_matches[idx].item())
        if result.weighted_n_matches is not None else None
    )
    weighted_frac = (
        float(result.weighted_frac_matches[idx].item())
        if result.weighted_frac_matches is not None else None
    )
    return SeedResult(
        spot_id=spot_id,
        best_or_mat=R_all[idx].detach().clone(),
        best_pos=pos_all[idx].detach().clone(),
        n_matches=int(result.n_matches[idx].item()),
        n_t_spots=n_t,
        n_t_frac_calc=n_t_frac,
        frac_matches=float(result.frac_matches[idx].item()),
        avg_ia=float(result.avg_ia[idx].item()),
        matched_ids=result.matched_obs_id[idx][result.matched[idx]].clone(),
        matched_pairs=pairs.detach().cpu(),
        weighted_n_matches=weighted_n,
        weighted_frac_matches=weighted_frac,
    )


# ---------------------------------------------------------------------------
# Block driver
# ---------------------------------------------------------------------------


def _default_use_c_compat() -> bool:
    """Default ON. Set `MIDAS_INDEX_EXHAUSTIVE=1` to disable the C-compat
    post-filter and pick the global argmax over all (R, pos) tuples.

    The C-compat filter replays IndexerOMP.c's adaptive `nDelta` position
    skip + `MinMatchesToAccept` threshold so that midas-index reaches the
    same winner as C IndexerOMP. Without it, midas finds higher-frac
    optima that C's heuristic skips, but ~22 of those (in the 500-grain
    Cu reference) are spurious — high frac coming from a smaller `nT`
    denominator (4 spots dropped by the pole filter), not a real grain.
    """
    import os
    return os.environ.get("MIDAS_INDEX_EXHAUSTIVE", "0") not in ("1", "true", "yes")


def _c_compat_visited_mask(
    frac_cpu,                        # np.ndarray (N_total,) float64
    n_m_frac_cpu,                    # np.ndarray (N_total,) int64
    n_t_frac_cpu,                    # np.ndarray (N_total,) int64
    seed_meta_valid: list[dict],     # only the valid seeds, in order
    seg_starts: list[int],           # per-valid-seed start index in flat block
    min_matches_to_accept_frac: float,
):
    """Compute the per-tuple "C would have visited and accepted" mask.

    Replays the adaptive search in IndexerOMP.c:1873-1947:

      for orient o in [0, n_or):
        n = n_min
        while n <= n_max:
          eval(o, n)  -> frac, nM, nT
          if frac >= 0.5: nDelta = 1
          else: nDelta = 5 - round_half_away_from_zero(frac * 4 / 0.5)
          n += nDelta

    Then applies C's threshold: nM_frac >= int(nT_frac * MinMatchesToAcceptFrac).

    Vectorized across the n_or orientations of each (seed, y0z0) — each
    walk is bounded by `n_pos_per_y0z0[k]` iterations (≤ ~5 in practice),
    so total work is tiny (~few ms per group on CPU).

    Returns
    -------
    mask : np.ndarray (N_total,) bool
        True for tuples C would have considered as a winner candidate.
    """
    import numpy as np
    n_total = len(frac_cpu)
    visited = np.zeros(n_total, dtype=bool)

    for s_idx, m in enumerate(seed_meta_valid):
        seg_start = seg_starts[s_idx]
        n_or = m["n_or"]
        n_pos_list = m["n_pos_per_y0z0"]
        cum_pos = 0
        for n_pos_k in n_pos_list:
            if n_pos_k == 0:
                continue
            # Vectorized walk across all orientations of this (seed, y0z0).
            # Each orient steps independently along its position axis.
            p_per_o = np.zeros(n_or, dtype=np.int64)
            # At most n_pos_k iterations needed (each step advances by >=1).
            for _ in range(n_pos_k + 1):
                active = p_per_o < n_pos_k
                if not active.any():
                    break
                o_active = np.nonzero(active)[0]
                p_active = p_per_o[o_active]
                flat = seg_start + (cum_pos + p_active) * n_or + o_active
                visited[flat] = True
                f = frac_cpu[flat]
                # `round_c`: round-half-away-from-zero (C's `round`), not
                # numpy's banker's rounding. Differs only at exact halves,
                # but match C semantics.
                scaled = f * 8.0
                rounded = np.floor(np.abs(scaled) + 0.5) * np.sign(scaled)
                nDelta = np.where(f >= 0.5, 1, 5 - rounded.astype(np.int64))
                p_per_o[o_active] += nDelta
            cum_pos += n_pos_k

    # MinMatchesToAccept threshold (C uses int truncation):
    #   MinMatchesToAccept = (int)(nT_frac * MinMatchesToAcceptFrac)
    #   accept = nM_frac >= MinMatchesToAccept
    min_accept = (n_t_frac_cpu.astype(np.float64) * min_matches_to_accept_frac).astype(np.int64)
    above_thresh = n_m_frac_cpu >= min_accept

    return visited & above_thresh


def _default_seed_group_size(device: torch.device | None = None) -> int:
    """Resolve `seed_group_size` from env (`MIDAS_INDEX_GROUP_SIZE`) or default.

    Larger groups amortize per-group Python/kernel-launch overhead and pack the
    forward+match into fewer big batches (better SM utilization), at the cost
    of more device memory. ``MIDAS_INDEX_GROUP_SIZE`` overrides the auto-pick.

    Auto-pick by device:
      * cuda  : 64  (sized for an 80 GB H100)
      * mps   : 1   (Apple-Silicon shared 64-128 GB; ``_compute_avg_ia``
                     materialises ~18× (N, T) tensors per group, so even
                     2 seeds blow past the MPS 75 % watermark on
                     ~1500-grain decks)
      * cpu   : 8   (CPU has page-cache headroom that MPS lacks)
    """
    import os
    raw = os.environ.get("MIDAS_INDEX_GROUP_SIZE")
    if raw is not None:
        return max(1, int(raw))
    if device is not None:
        if device.type == "cuda":
            return 64
        if device.type == "mps":
            return 1
    return 8


def run_block(
    ctx: IndexerContext,
    spot_ids: torch.Tensor,           # int64 (n_total_seeds,)
    block_nr: int,
    n_blocks: int,
    seed_group_size: int | None = None,
) -> IndexerResult:
    """Process one block's slice of spot_ids and return seed results.

    Mirrors IndexerOMP.c:2287-2347. Per-block sharding:
        startRowNr = ceil(n_total / n_blocks) * block_nr
        endRowNr   = min(ceil(n_total / n_blocks) * (block_nr+1) - 1, n_total - 1)

    Cross-seed batched in groups of `seed_group_size`: builds the full
    cartesian product of (y0_z0 × R × pos) for that group into ONE tensor,
    runs a single forward sim plus a single match call, then does per-seed
    argmax via async kernel-launch loop. Eliminates per-seed Python `.item()`
    syncs that one-seed-at-a-time processing would pay. Group size caps
    peak GPU memory: ~32 seeds × ~1500 tuples × ~100 spots × 14 cols × 4 B
    ≈ 270 MB for the theor tensor, well within H100 limits even with the
    dense candidate gather in compare_spots layered on top.
    """
    n_total = int(spot_ids.numel())
    block_size = math.ceil(n_total / max(1, n_blocks))
    start = block_size * block_nr
    end_inclusive = min(block_size * (block_nr + 1) - 1, n_total - 1)
    if start > end_inclusive or start >= n_total:
        return IndexerResult(block_nr=block_nr, n_blocks=n_blocks, seeds=[])

    if seed_group_size is None:
        seed_group_size = _default_seed_group_size(ctx.device)

    seeds_block = spot_ids[start:end_inclusive + 1]
    n_seeds = int(seeds_block.numel())

    group_ranges: list[tuple[int, int]] = []
    for grp_start in range(0, n_seeds, seed_group_size):
        group_ranges.append((grp_start, min(grp_start + seed_group_size, n_seeds)))
    if not group_ranges:
        return IndexerResult(block_nr=block_nr, n_blocks=n_blocks, seeds=[])

    # Prefetch pattern: while group i's GPU work is in flight, build group
    # i+1's CPU setup. Eliminates ~40 ms/group of CPU setup from the wall
    # clock except for the first group's. Only effective on CUDA — on CPU,
    # there's no parallel stream so we just call the sequential wrapper.
    seeds_out: list[SeedResult] = []
    if ctx.device.type != "cuda":
        for s, e in group_ranges:
            seeds_out.extend(_process_seed_group(ctx, seeds_block[s:e]))
        return IndexerResult(block_nr=block_nr, n_blocks=n_blocks, seeds=seeds_out)

    # CUDA path: prefetch next group's CPU setup during current group's GPU work.
    next_setup = _setup_group_cpu(ctx, seeds_block[group_ranges[0][0]:group_ranges[0][1]])
    for i, _ in enumerate(group_ranges):
        cur_setup = next_setup
        next_setup = None
        if cur_setup is None:
            # Even when current group has no valid seeds, we still need to
            # prefetch the next group so we don't drop it.
            if i + 1 < len(group_ranges):
                ns, ne = group_ranges[i + 1]
                next_setup = _setup_group_cpu(ctx, seeds_block[ns:ne])
            continue

        if i + 1 < len(group_ranges):
            # Launch GPU kernels for current group; the returned closure
            # blocks for the .cpu() syncs only when called.
            finalize = _compute_group_gpu_launch(ctx, cur_setup)
            # Setup NEXT group on CPU. This is the overlap window: GPU is
            # crunching simulate+compare while CPU builds the next batch.
            ns, ne = group_ranges[i + 1]
            next_setup = _setup_group_cpu(ctx, seeds_block[ns:ne])
            seeds_out.extend(finalize())
        else:
            # Last group: nothing to prefetch.
            seeds_out.extend(_compute_group_gpu(ctx, cur_setup))

    return IndexerResult(block_nr=block_nr, n_blocks=n_blocks, seeds=seeds_out)


def _setup_group_cpu(
    ctx: IndexerContext,
    seeds_block: torch.Tensor,        # (n_seeds,) CPU int64
) -> dict | None:
    """CPU-only phase of `_process_seed_group`: builds (R, pos, ref_rad) on CPU.

    Returns a dict of CPU tensors + per-seed metadata, or `None` if no
    valid seeds in this group. The companion `_compute_group_gpu` consumes
    this. Splitting the phases lets `run_block` overlap this CPU work with
    the previous group's GPU compute (prefetch pattern).
    """
    p = ctx.params
    n_seeds = int(seeds_block.numel())

    # Pure-CPU obs lookup (avoids any GPU sync — needed so this function can
    # run on the CPU thread while a previous group's GPU kernels are still
    # in flight). Builds: `obs_rows` (which row of obs each seed_id maps to)
    # and `has_match_cpu` (which seeds have a corresponding obs).
    import numpy as np
    obs_id_col = ctx.obs_cpu[:, 4].astype(np.int64)
    seed_ids = seeds_block.cpu().numpy().astype(np.int64)
    # Build a hash map seed_id -> obs row. Reuse across calls if cached.
    if not hasattr(ctx, "_obs_id_to_row"):
        ctx._obs_id_to_row = {int(v): i for i, v in enumerate(obs_id_col)}
    obs_id_to_row = ctx._obs_id_to_row
    obs_rows_list = [obs_id_to_row.get(int(sid), -1) for sid in seed_ids]
    has_match_cpu = np.array([r >= 0 for r in obs_rows_list], dtype=bool)
    safe_rows = np.array([max(0, r) for r in obs_rows_list], dtype=np.int64)
    seed_obs_cpu = ctx.obs_cpu[safe_rows]

    cpu = torch.device("cpu")
    R_chunks_cpu: list[torch.Tensor] = []
    pos_chunks_cpu: list[torch.Tensor] = []
    ref_rad_chunks_cpu: list[torch.Tensor] = []
    seed_meta: list[dict] = []

    if not hasattr(ctx, "_ring_hkl_cpu"):
        ctx._ring_hkl_cpu = {
            rn: t.detach().to(device=cpu) for rn, t in ctx.ring_hkl.items()
        }

    for i in range(n_seeds):
        sid = int(seeds_block[i].item())
        if not bool(has_match_cpu[i]):
            seed_meta.append({"sid": sid, "valid": False})
            continue
        seed_obs = seed_obs_cpu[i]
        ys = float(seed_obs[0])
        zs = float(seed_obs[1])
        seed_omega = float(seed_obs[2])
        seed_ring_rad = float(seed_obs[3])
        seed_eta = float(seed_obs[6])
        seed_ring_nr = int(seed_obs[5])
        if seed_ring_nr not in ctx.ring_hkl:
            seed_meta.append({"sid": sid, "valid": False})
            continue

        hkl_cpu = ctx._ring_hkl_cpu[seed_ring_nr]
        ring_rad_user = p.get_ring_radius(seed_ring_nr)
        ring_rad = ring_rad_user if ring_rad_user > 0 else seed_ring_rad
        ttheta = ctx.ring_ttheta[seed_ring_nr]
        hkl_int_for_ring = ctx.ring_hkl_int[seed_ring_nr]

        if p.UseFriedelPairs == 1:
            seed_yz = seeds_module.generate_ideal_spots_friedel(
                ys=ys, zs=zs, ttheta_deg=ttheta, eta_deg=seed_eta, omega_deg=seed_omega,
                ring_nr=seed_ring_nr, ring_rad=ring_rad,
                rsample=p.Rsample, hbeam=p.Hbeam,
                ome_tol=p.MarginOme, radius_tol=p.MarginRadial,
                obs_spots=ctx.obs, device=cpu, dtype=ctx.dtype,
            )
            # C parity: friedel-mixed fallback when no obs partner found
            # (matches IndexerOMP.c:1815-1822).
            if seed_yz.shape[0] == 0:
                seed_yz = seeds_module.generate_ideal_spots_friedel_mixed(
                    ys=ys, zs=zs, ttheta_deg=ttheta, eta_deg=seed_eta,
                    omega_deg=seed_omega,
                    ring_nr=seed_ring_nr, ring_rad=ring_rad, lsd=p.Distance,
                    rsample=p.Rsample, hbeam=p.Hbeam,
                    step_size_pos=p.StepsizePos,
                    ome_tol=p.MarginOme, radial_tol=p.MarginRadial,
                    eta_tol_um=p.MarginEta,
                    obs_spots=ctx.obs, device=cpu, dtype=ctx.dtype,
                )
        elif p.UseFriedelPairs == 2:
            seed_yz = seeds_module.generate_ideal_spots_friedel_mixed(
                ys=ys, zs=zs, ttheta_deg=ttheta, eta_deg=seed_eta, omega_deg=seed_omega,
                ring_nr=seed_ring_nr, ring_rad=ring_rad, lsd=p.Distance,
                rsample=p.Rsample, hbeam=p.Hbeam,
                step_size_pos=p.StepsizePos,
                ome_tol=p.MarginOme, radial_tol=p.MarginRad, eta_tol_um=p.MarginEta,
                obs_spots=ctx.obs, device=cpu, dtype=ctx.dtype,
            )
        else:
            seed_yz = seeds_module.generate_ideal_spots(
                ys=ys, zs=zs, ttheta_deg=ttheta, eta_deg=seed_eta,
                ring_rad=ring_rad, rsample=p.Rsample, hbeam=p.Hbeam,
                step_size=p.StepsizePos, device=cpu, dtype=ctx.dtype,
            )
        if seed_yz.shape[0] == 0:
            seed_meta.append({"sid": sid, "valid": False})
            continue

        y0_all = seed_yz[:, 0]
        z0_all = seed_yz[:, 1]
        plane_normals = _spot_to_gv_batch(p.Distance, y0_all, z0_all, seed_omega)
        Rs_B = orientation_grid.generate_candidate_orientations_batched(
            hkl=hkl_cpu, plane_normals=plane_normals,
            stepsize_orient_deg=p.StepsizeOrient,
            ring_nr=seed_ring_nr, space_group=p.SpaceGroup,
            hkl_int=hkl_int_for_ring, abcabg=p.LatticeConstant,
        )
        n_or = Rs_B.shape[1]
        if n_or == 0:
            seed_meta.append({"sid": sid, "valid": False})
            continue

        # Scanning mode (PF): the voxel center is the FIXED grain position
        # (mirrors IndexerScanningOMP.c:1156: ``ga=xThis, gb=yThis, gc=zThis``).
        # NO position grid — the voxel grid IS the position grid. Skipping
        # build_position_grid here is the dominant perf win for PF: the
        # legacy FF call expands N by ~Rsample/StepsizePos (50-100×) per
        # seed, which crushes downstream compare_spots / simulate.
        if ctx.scan_positions is not None and ctx.current_voxel_xy is not None:
            n_seed_yz = int(y0_all.numel())
            voxel_x = float(ctx.current_voxel_xy[0].item())
            voxel_y = float(ctx.current_voxel_xy[1].item())
            positions_seed = torch.tensor(
                [[voxel_x, voxel_y, 0.0]] * n_seed_yz,
                device=cpu, dtype=ctx.dtype,
            )
            candidate_idx = torch.arange(n_seed_yz, device=cpu, dtype=torch.int64)
        else:
            positions_seed, candidate_idx = position_grid.build_position_grid(
                seed_y0=y0_all, seed_z0=z0_all,
                ys=ys, zs=zs, omega_deg=seed_omega,
                distance=p.Distance, r_sample=p.Rsample, step_size=p.StepsizePos,
                h_beam=p.Hbeam,
            )
        if positions_seed.shape[0] == 0:
            seed_meta.append({"sid": sid, "valid": False})
            continue

        Rs_per_pos = Rs_B[candidate_idx]
        pos_expanded = positions_seed.unsqueeze(1).expand(-1, n_or, -1)
        local_R_cat = Rs_per_pos.reshape(-1, 3, 3)
        local_pos_cat = pos_expanded.reshape(-1, 3)
        n_local = local_R_cat.shape[0]

        # Per-(y0,z0) position counts — used by the C-compat post-filter to
        # replay IndexerOMP's adaptive `nDelta` skip walk. See
        # `_c_compat_visited_mask` for the algorithm. (In PF mode this is
        # all-ones since each seed_yz contributes exactly one position.)
        n_pos_per_y0z0 = torch.bincount(candidate_idx, minlength=y0_all.numel()).tolist()

        R_chunks_cpu.append(local_R_cat)
        pos_chunks_cpu.append(local_pos_cat)
        ref_rad_chunks_cpu.append(
            # See note above (~line 318): ref_rad is the seed's obs col 3
            # for radial check parity with C IndexerOMP.c:1797.
            torch.full((n_local,), seed_ring_rad, device=cpu, dtype=ctx.dtype)
        )
        seed_meta.append({
            "sid": sid, "valid": True,
            "ring_nr": seed_ring_nr,
            "n_local": n_local,
            "n_or": n_or,
            "n_pos_per_y0z0": n_pos_per_y0z0,
        })

    if not R_chunks_cpu:
        return None

    R_all_cpu = torch.cat(R_chunks_cpu, dim=0).pin_memory() if ctx.device.type == "cuda" else torch.cat(R_chunks_cpu, dim=0)
    pos_all_cpu = torch.cat(pos_chunks_cpu, dim=0).pin_memory() if ctx.device.type == "cuda" else torch.cat(pos_chunks_cpu, dim=0)
    ref_rad_all_cpu = torch.cat(ref_rad_chunks_cpu, dim=0).pin_memory() if ctx.device.type == "cuda" else torch.cat(ref_rad_chunks_cpu, dim=0)

    return {
        "R_cpu": R_all_cpu,
        "pos_cpu": pos_all_cpu,
        "ref_rad_cpu": ref_rad_all_cpu,
        "seed_meta": seed_meta,
    }


def _compute_group_gpu(
    ctx: IndexerContext,
    setup: dict,
) -> list[SeedResult]:
    """GPU phase: transfer, forward sim, compare, argmax, build SeedResults.

    Consumes the CPU-resident output of `_setup_group_cpu`. Kernels are
    enqueued async so that any CPU work scheduled between this call and the
    final `.cpu()` syncs (e.g. setting up the next group) overlaps with
    GPU compute on a CUDA device.
    """
    p = ctx.params
    seed_meta = setup["seed_meta"]
    use_c_compat = _default_use_c_compat()

    # CPU fast path: single fused numba kernel does simulate + compare + avg_ia.
    # Skips all torch round-trips between phases (was ~50% of wall time in
    # Phase 3 from numpy.ascontiguousarray on theor columns).
    if ctx.device.type != "cuda" and (
        ctx.soft_beam_weight_fn is None
        and not ctx.adapter._has_panel_coverage
    ):
        try:
            return _compute_group_cpu_fused(ctx, setup, use_c_compat)
        except _FusedFallback:
            pass  # fall through to legacy torch path

    # Async transfer to GPU.
    R_all = setup["R_cpu"].to(device=ctx.device, non_blocking=True)
    pos_all = setup["pos_cpu"].to(device=ctx.device, non_blocking=True)
    ref_rad_all = setup["ref_rad_cpu"].to(device=ctx.device, non_blocking=True)

    # Forward simulation across all tuples.
    theor, valid = ctx.adapter.simulate(R_all, pos_all, lattice=None)

    # Predict dense-path peak vs free device memory and chunk if needed.
    strategy, chunk_size = matching.pick_compare_strategy(theor, ctx.bin_max_count)
    if strategy == "jagged":
        _log_jagged_fallback(theor, ctx.bin_max_count, chunk_size)

    # Match. With `max_n_cap`, this returns async — no internal `.item()` sync.
    result = matching.compare_spots(
        theor=theor, valid=valid, obs=ctx.obs,
        bin_data=ctx.bin_data, bin_ndata=ctx.bin_ndata,
        ref_rad=ref_rad_all,
        margin_rad=p.MarginRad, margin_radial=p.MarginRadial,
        eta_margins=ctx.eta_margins, ome_margins=ctx.ome_margins,
        eta_bin_size=p.EtaBinSize, ome_bin_size=p.OmeBinSize,
        n_eta_bins=ctx.n_eta_bins, n_ome_bins=ctx.n_ome_bins,
        rings_to_reject=ctx.rings_to_reject,
        distance=p.Distance, pos=pos_all,
        max_n_cap=ctx.bin_max_count,
        strategy=strategy, chunk_size=chunk_size,
        **ctx.scan_kwargs(theor.shape[0]),
    )

    keys = reduce_.pack_score(result.frac_matches, result.avg_ia)
    seed_meta_valid = [m for m in seed_meta if m["valid"]]
    n_per_valid = [m["n_local"] for m in seed_meta_valid]
    seg_starts: list[int] = []
    cur = 0
    for nl in n_per_valid:
        seg_starts.append(cur)
        cur += nl
    if not seed_meta_valid:
        return []

    if use_c_compat:
        # Apply C IndexerOMP's adaptive nDelta skip + MinMatchesToAccept
        # threshold so we land on the same winner as C. See
        # `_c_compat_visited_mask` for details. CPU-side post-process of
        # the existing exhaustive results — no GPU rework.
        import numpy as np
        frac_cpu = result.frac_matches.cpu().numpy()
        n_m_frac_cpu = result.n_matches_frac.cpu().numpy()
        n_t_frac_cpu = result.n_t_frac.cpu().numpy()
        mask_cpu = _c_compat_visited_mask(
            frac_cpu, n_m_frac_cpu, n_t_frac_cpu,
            seed_meta_valid, seg_starts, p.MinMatchesToAcceptFrac,
        )
        keys_cpu = keys.cpu().numpy()
        # Mark non-eligible tuples with INT64_MIN so argmax skips them.
        keys_cpu[~mask_cpu] = np.iinfo(np.int64).min
        best_global_idx_list: list[int] = []
        # Per-seed: track which seeds have ZERO eligible tuples (matches
        # C's `bestMatchFound == 0` early return — no result for that seed).
        seed_has_match: list[bool] = []
        for s_start, s_len in zip(seg_starts, n_per_valid):
            seg = keys_cpu[s_start:s_start + s_len]
            seg_mask = mask_cpu[s_start:s_start + s_len]
            if not seg_mask.any():
                seed_has_match.append(False)
                best_global_idx_list.append(s_start)  # placeholder
            else:
                seed_has_match.append(True)
                best_global_idx_list.append(s_start + int(seg.argmax()))
        best_global_idx = torch.tensor(best_global_idx_list, device=ctx.device, dtype=torch.int64)
    else:
        # Exhaustive: pure GPU per-seed argmax. Every seed produces a result.
        seed_has_match = [True] * len(seed_meta_valid)
        best_global_idx_list_gpu: list[torch.Tensor] = []
        for s_start, s_len in zip(seg_starts, n_per_valid):
            local_argmax = keys[s_start:s_start + s_len].argmax()
            best_global_idx_list_gpu.append(s_start + local_argmax)
        best_global_idx = torch.stack(best_global_idx_list_gpu)

    # Batched gather to single CPU transfer.
    best_R = R_all[best_global_idx].detach().cpu()
    best_pos = pos_all[best_global_idx].detach().cpu()
    best_n_match = result.n_matches[best_global_idx].cpu()
    best_frac = result.frac_matches[best_global_idx].cpu()
    best_ia = result.avg_ia[best_global_idx].cpu()
    best_valid = valid[best_global_idx]
    best_n_t = best_valid.sum(dim=-1).cpu()
    best_obs_id = result.matched_obs_id[best_global_idx].cpu()
    best_delta = result.delta_omega[best_global_idx].cpu()
    best_match_mask = result.matched[best_global_idx].cpu()
    if ctx.rings_to_reject.numel() > 0:
        best_theor_rings = theor[best_global_idx, :, 9].long()
        in_reject = (
            best_theor_rings.unsqueeze(-1) == ctx.rings_to_reject
        ).any(dim=-1)
        best_n_t_frac = (best_valid & ~in_reject).sum(dim=-1).cpu()
    else:
        best_n_t_frac = best_n_t

    T = theor.shape[1]
    seeds_out: list[SeedResult] = []
    valid_idx = 0
    for m in seed_meta:
        if not m["valid"]:
            continue
        # In c_compat mode, a "valid" seed (one with R/pos tuples to evaluate)
        # may still produce no result if no tuple passed C's MinMatchesToAccept
        # threshold. Mirrors C's `bestMatchFound == 0` early return.
        if not seed_has_match[valid_idx]:
            valid_idx += 1
            continue
        # Compact (matched-only) layout — see note in `process_seed`.
        mm = best_match_mask[valid_idx]
        pairs = torch.stack(
            [best_obs_id[valid_idx][mm].to(ctx.dtype),
             best_delta[valid_idx][mm].to(ctx.dtype)],
            dim=-1,
        )
        seeds_out.append(SeedResult(
            spot_id=m["sid"],
            best_or_mat=best_R[valid_idx].clone(),
            best_pos=best_pos[valid_idx].clone(),
            n_matches=int(best_n_match[valid_idx].item()),
            n_t_spots=int(best_n_t[valid_idx].item()),
            n_t_frac_calc=int(best_n_t_frac[valid_idx].item()),
            frac_matches=float(best_frac[valid_idx].item()),
            avg_ia=float(best_ia[valid_idx].item()),
            matched_ids=best_obs_id[valid_idx][mm].clone(),
            matched_pairs=pairs,
        ))
        valid_idx += 1
    return seeds_out


def _compute_group_gpu_launch(ctx: IndexerContext, setup: dict):
    """Async launch of GPU work; returns a `finalize()` closure.

    Splits `_compute_group_gpu` into:
      1. Kernel-launch portion (this function) — enqueues simulate, compare,
         pack_score, argmax. All async — no `.cpu()` or `.item()` sync.
      2. Finalize closure — runs the per-tuple gather + `.cpu()` syncs +
         SeedResult construction. Calling it blocks until the GPU finishes.

    The split lets `run_block` run setup_cpu(group i+1) on the CPU thread
    *between* steps 1 and 2, so the CPU work overlaps with the group-i
    kernels still executing on the CUDA stream.
    """
    p = ctx.params
    seed_meta = setup["seed_meta"]

    R_all = setup["R_cpu"].to(device=ctx.device, non_blocking=True)
    pos_all = setup["pos_cpu"].to(device=ctx.device, non_blocking=True)
    ref_rad_all = setup["ref_rad_cpu"].to(device=ctx.device, non_blocking=True)

    theor, valid = ctx.adapter.simulate(R_all, pos_all, lattice=None)

    # Predict dense-path peak vs free device memory and chunk if needed.
    strategy, chunk_size = matching.pick_compare_strategy(theor, ctx.bin_max_count)
    if strategy == "jagged":
        _log_jagged_fallback(theor, ctx.bin_max_count, chunk_size)

    result = matching.compare_spots(
        theor=theor, valid=valid, obs=ctx.obs,
        bin_data=ctx.bin_data, bin_ndata=ctx.bin_ndata,
        ref_rad=ref_rad_all,
        margin_rad=p.MarginRad, margin_radial=p.MarginRadial,
        eta_margins=ctx.eta_margins, ome_margins=ctx.ome_margins,
        eta_bin_size=p.EtaBinSize, ome_bin_size=p.OmeBinSize,
        n_eta_bins=ctx.n_eta_bins, n_ome_bins=ctx.n_ome_bins,
        rings_to_reject=ctx.rings_to_reject,
        distance=p.Distance, pos=pos_all,
        max_n_cap=ctx.bin_max_count,
        strategy=strategy, chunk_size=chunk_size,
        **ctx.scan_kwargs(theor.shape[0]),
    )
    keys = reduce_.pack_score(result.frac_matches, result.avg_ia)
    seed_meta_valid = [m for m in seed_meta if m["valid"]]
    n_per_valid = [m["n_local"] for m in seed_meta_valid]
    seg_starts: list[int] = []
    cur = 0
    for nl in n_per_valid:
        seg_starts.append(cur)
        cur += nl
    if not seed_meta_valid:
        return lambda: []

    use_c_compat = _default_use_c_compat()
    if not use_c_compat:
        # Async GPU per-seed argmax (kernels enqueued, sync deferred to finalize).
        best_global_idx_list_gpu: list[torch.Tensor] = []
        for s_start, s_len in zip(seg_starts, n_per_valid):
            local_argmax = keys[s_start:s_start + s_len].argmax()
            best_global_idx_list_gpu.append(s_start + local_argmax)
        best_global_idx_pre = torch.stack(best_global_idx_list_gpu)
    else:
        best_global_idx_pre = None  # computed in finalize after sync

    def finalize() -> list[SeedResult]:
        nonlocal best_global_idx_pre
        seed_has_match: list[bool]
        if use_c_compat:
            # Sync small score tensors to CPU, build C-compat mask, argmax on CPU.
            # The previous group's GPU kernels (and this group's setup_cpu)
            # already overlapped during the launch's enqueue phase.
            import numpy as np
            frac_cpu = result.frac_matches.cpu().numpy()
            n_m_frac_cpu = result.n_matches_frac.cpu().numpy()
            n_t_frac_cpu = result.n_t_frac.cpu().numpy()
            mask_cpu = _c_compat_visited_mask(
                frac_cpu, n_m_frac_cpu, n_t_frac_cpu,
                seed_meta_valid, seg_starts, p.MinMatchesToAcceptFrac,
            )
            keys_cpu = keys.cpu().numpy()
            keys_cpu[~mask_cpu] = np.iinfo(np.int64).min
            best_idx_cpu: list[int] = []
            seed_has_match = []
            for s_start, s_len in zip(seg_starts, n_per_valid):
                seg_mask = mask_cpu[s_start:s_start + s_len]
                if not seg_mask.any():
                    seed_has_match.append(False)
                    best_idx_cpu.append(s_start)  # placeholder
                else:
                    seed_has_match.append(True)
                    best_idx_cpu.append(s_start + int(keys_cpu[s_start:s_start + s_len].argmax()))
            best_global_idx = torch.tensor(best_idx_cpu, device=ctx.device, dtype=torch.int64)
        else:
            best_global_idx = best_global_idx_pre
            seed_has_match = [True] * len(seed_meta_valid)

        best_R = R_all[best_global_idx].detach().cpu()
        best_pos = pos_all[best_global_idx].detach().cpu()
        best_n_match = result.n_matches[best_global_idx].cpu()
        best_frac = result.frac_matches[best_global_idx].cpu()
        best_ia = result.avg_ia[best_global_idx].cpu()
        best_valid = valid[best_global_idx]
        best_n_t = best_valid.sum(dim=-1).cpu()
        best_obs_id = result.matched_obs_id[best_global_idx].cpu()
        best_delta = result.delta_omega[best_global_idx].cpu()
        best_match_mask = result.matched[best_global_idx].cpu()
        if ctx.rings_to_reject.numel() > 0:
            best_theor_rings = theor[best_global_idx, :, 9].long()
            in_reject = (
                best_theor_rings.unsqueeze(-1) == ctx.rings_to_reject
            ).any(dim=-1)
            best_n_t_frac = (best_valid & ~in_reject).sum(dim=-1).cpu()
        else:
            best_n_t_frac = best_n_t
        T = theor.shape[1]
        seeds_out: list[SeedResult] = []
        valid_idx = 0
        for m in seed_meta:
            if not m["valid"]:
                continue
            if not seed_has_match[valid_idx]:
                # In c_compat mode, no tuple passed C's MinMatchesToAccept
                # threshold for this seed — match C's `bestMatchFound == 0`
                # early return.
                valid_idx += 1
                continue
            # Compact (matched-only) layout — see note in `process_seed`.
            mm = best_match_mask[valid_idx]
            pairs = torch.stack(
                [best_obs_id[valid_idx][mm].to(ctx.dtype),
                 best_delta[valid_idx][mm].to(ctx.dtype)],
                dim=-1,
            )
            seeds_out.append(SeedResult(
                spot_id=m["sid"],
                best_or_mat=best_R[valid_idx].clone(),
                best_pos=best_pos[valid_idx].clone(),
                n_matches=int(best_n_match[valid_idx].item()),
                n_t_spots=int(best_n_t[valid_idx].item()),
                n_t_frac_calc=int(best_n_t_frac[valid_idx].item()),
                frac_matches=float(best_frac[valid_idx].item()),
                avg_ia=float(best_ia[valid_idx].item()),
                matched_ids=best_obs_id[valid_idx][mm].clone(),
                matched_pairs=pairs,
            ))
            valid_idx += 1
        return seeds_out

    return finalize


_GC_CALL_COUNTER = [0]
_GC_EVERY_N_SEEDS = 32


class _FusedFallback(Exception):
    """Internal: raised when the CPU fused path can't handle a config and
    we should fall through to the legacy torch pipeline."""
    pass


def _compute_group_cpu_fused(ctx: "IndexerContext", setup: dict, use_c_compat: bool) -> "list[SeedResult]":
    """CPU fast path: one fused numba kernel does simulate + compare_spots +
    avg_ia. Returns SeedResults directly. Falls back via _FusedFallback if
    numba isn't available or the config has soft-attribution / panel coverage.
    """
    try:
        from .compute.fused_numba import (
            _simulate_and_compare_fused_numba, _NUMBA_AVAILABLE,
        )
    except ImportError:
        raise _FusedFallback()
    if not _NUMBA_AVAILABLE:
        raise _FusedFallback()

    import math
    import numpy as np
    from .compute import reduce as reduce_
    from .compute import matching as _matching

    p = ctx.params
    seed_meta = setup["seed_meta"]

    R_all = setup["R_cpu"]
    pos_all = setup["pos_cpu"]
    ref_rad_all = setup["ref_rad_cpu"]
    N = int(R_all.shape[0])
    if N == 0:
        return []

    R_np = np.ascontiguousarray(R_all.detach().cpu().numpy().astype(np.float64, copy=False))
    pos_np = np.ascontiguousarray(pos_all.detach().cpu().numpy().astype(np.float64, copy=False))
    ref_rad_np = np.ascontiguousarray(ref_rad_all.detach().cpu().numpy().astype(np.float64, copy=False))

    # HKL list + cached LUTs from forward adapter
    adapter = ctx.adapter
    hkls_cart_t = adapter.hkls_real[:, :3]
    thetas_t = adapter.hkls_real[:, 5]
    hkls_cart_np = np.ascontiguousarray(hkls_cart_t.detach().cpu().numpy().astype(np.float64, copy=False))
    thetas_np = np.ascontiguousarray(thetas_t.detach().cpu().numpy().astype(np.float64, copy=False))
    len_hkl = np.linalg.norm(hkls_cart_np, axis=-1)
    ring_nr_per_hkl_np = np.ascontiguousarray(
        adapter.ring_nr_per_hkl.detach().cpu().numpy().astype(np.int64, copy=False)
    )
    ring_radius_lut_np = np.ascontiguousarray(
        adapter.ring_radius_lut.detach().cpu().numpy().astype(np.float64, copy=False)
    )
    wedge_rad = float(p.Wedge) * math.pi / 180.0 if hasattr(p, "Wedge") else 0.0
    Lsd = float(p.Distance)
    min_eta_rad = float(p.ExcludePoleAngle) * math.pi / 180.0

    if adapter.omega_ranges is not None and adapter.omega_ranges.numel() > 0:
        omega_ranges_np = np.ascontiguousarray(
            adapter.omega_ranges.detach().cpu().numpy().astype(np.float64, copy=False)
        )
        box_sizes_np = np.ascontiguousarray(
            adapter.box_sizes.detach().cpu().numpy().astype(np.float64, copy=False)
        )
        has_omega_box = True
    else:
        omega_ranges_np = np.zeros((0, 2), dtype=np.float64)
        box_sizes_np = np.zeros((0, 4), dtype=np.float64)
        has_omega_box = False

    # Obs columns (cached via matching helpers)
    obs_views = _matching._cached_obs_numpy(ctx.obs)
    bin_data_np = _matching._cached_int32_view(ctx.bin_data)
    bin_ndata_np = _matching._cached_int32_view(ctx.bin_ndata)
    eta_margins_np = np.ascontiguousarray(
        ctx.eta_margins.detach().cpu().numpy().astype(np.float64, copy=False)
    )
    rings_to_reject_np = (
        np.ascontiguousarray(ctx.rings_to_reject.detach().cpu().numpy().astype(np.int64, copy=False))
        if ctx.rings_to_reject.numel() > 0 else np.zeros(0, dtype=np.int64)
    )

    # Scan-aware
    scan_active = (
        ctx.scan_pos_tol_um > 0
        and ctx.scan_positions is not None
        and ctx.current_voxel_xy is not None
    )
    if scan_active:
        if ctx.obs.shape[-1] < 10:
            raise _FusedFallback()
        v_xy = ctx.current_voxel_xy.detach().cpu().numpy().astype(np.float64)
        voxel_x_np = np.full(N, float(v_xy[0]), dtype=np.float64)
        voxel_y_np = np.full(N, float(v_xy[1]), dtype=np.float64)
        if ctx.obs_scan_nr_int64 is not None:
            obs_scan_idx_np = np.ascontiguousarray(
                ctx.obs_scan_nr_int64.detach().cpu().numpy().astype(np.int64, copy=False)
            )
        else:
            obs_scan_idx_np = np.ascontiguousarray(
                ctx.obs[..., 9].detach().cpu().numpy().astype(np.int64, copy=False)
            )
        scan_pos_arr_np = np.ascontiguousarray(
            ctx.scan_positions.detach().cpu().numpy().astype(np.float64, copy=False)
        )
    else:
        voxel_x_np = np.zeros(N, dtype=np.float64)
        voxel_y_np = np.zeros(N, dtype=np.float64)
        obs_scan_idx_np = np.zeros(1, dtype=np.int64)
        scan_pos_arr_np = np.zeros(1, dtype=np.float64)

    (best_delta_ome, best_matched_id, best_matched_row, has_match,
     theor_omega, theor_eta, theor_yl_disp, theor_zl_disp,
     theor_ring_nr, valid_mask,
     n_matches_out, n_matches_frac_out, n_t_frac_out, avg_ia_out) = (
        _simulate_and_compare_fused_numba(
            R_np, pos_np,
            hkls_cart_np, thetas_np, len_hkl,
            ring_nr_per_hkl_np, ring_radius_lut_np, int(ring_radius_lut_np.shape[0] - 1),
            wedge_rad, Lsd, min_eta_rad, 1e-12,
            omega_ranges_np, box_sizes_np, has_omega_box,
            bin_ndata_np, bin_data_np,
            float(p.EtaBinSize), float(p.OmeBinSize),
            int(ctx.n_eta_bins), int(ctx.n_ome_bins),
            obs_views["y"], obs_views["z"], obs_views["ome"], obs_views["eta"],
            obs_views["ringrad"], obs_views["rad"], obs_views["id"],
            eta_margins_np, int(eta_margins_np.shape[0] - 1),
            float(p.MarginRadial), float(p.MarginRad),
            rings_to_reject_np,
            ref_rad_np,
            bool(scan_active), voxel_x_np, voxel_y_np,
            obs_scan_idx_np, scan_pos_arr_np,
            float(ctx.scan_pos_tol_um),
            bool(ctx.friedel_symmetric_scan_filter),
        )
    )

    # ── Per-seed argmax over the (N, K) match score, then build SeedResults
    seed_meta_valid = [m for m in seed_meta if m["valid"]]
    n_per_valid = [m["n_local"] for m in seed_meta_valid]
    seg_starts: list[int] = []
    cur = 0
    for nl in n_per_valid:
        seg_starts.append(cur)
        cur += nl
    if not seed_meta_valid:
        return []

    # pack_score = (n_matches_frac << ...) | (avg_ia low bits) — same as torch.
    frac_matches = n_matches_frac_out.astype(np.float64) / np.maximum(n_t_frac_out, 1).astype(np.float64)
    # Use torch for pack_score to keep numerics identical.
    import torch as _torch
    frac_t = _torch.from_numpy(frac_matches)
    ia_t = _torch.from_numpy(avg_ia_out)
    keys = reduce_.pack_score(frac_t, ia_t)
    keys_cpu = keys.numpy()

    if use_c_compat:
        mask_cpu = _c_compat_visited_mask(
            frac_matches, n_matches_frac_out, n_t_frac_out,
            seed_meta_valid, seg_starts, p.MinMatchesToAcceptFrac,
        )
        # Mark non-eligible tuples with INT64_MIN so argmax skips them.
        keys_cpu = keys_cpu.copy()
        keys_cpu[~mask_cpu] = np.iinfo(np.int64).min
        best_global_idx_list: list[int] = []
        seed_has_match: list[bool] = []
        for s_start, s_len in zip(seg_starts, n_per_valid):
            seg = keys_cpu[s_start:s_start + s_len]
            seg_mask = mask_cpu[s_start:s_start + s_len]
            if not seg_mask.any():
                seed_has_match.append(False)
                best_global_idx_list.append(s_start)  # placeholder
            else:
                seed_has_match.append(True)
                best_global_idx_list.append(s_start + int(np.argmax(seg)))
    else:
        seed_has_match = [True] * len(seed_meta_valid)
        best_global_idx_list = []
        for s_start, s_len in zip(seg_starts, n_per_valid):
            seg = keys_cpu[s_start:s_start + s_len]
            best_global_idx_list.append(s_start + int(np.argmax(seg)))

    # Gather best per-seed
    best_global_idx = np.asarray(best_global_idx_list, dtype=np.int64)
    best_R = R_np[best_global_idx]
    best_pos = pos_np[best_global_idx]
    best_n_match = n_matches_out[best_global_idx]
    best_frac = frac_matches[best_global_idx]
    best_ia = avg_ia_out[best_global_idx]
    best_valid_mask = valid_mask[best_global_idx]
    best_n_t = best_valid_mask.sum(axis=-1)
    best_obs_id = best_matched_id[best_global_idx]
    best_delta = best_delta_ome[best_global_idx]
    best_match_mask = has_match[best_global_idx]
    if ctx.rings_to_reject.numel() > 0:
        best_theor_rings = theor_ring_nr[best_global_idx].astype(np.int64)
        in_reject = np.isin(best_theor_rings, rings_to_reject_np)
        best_n_t_frac = (best_valid_mask & ~in_reject).sum(axis=-1)
    else:
        best_n_t_frac = best_n_t

    seeds_out: list[SeedResult] = []
    valid_idx = 0
    for m in seed_meta:
        if not m["valid"]:
            continue
        if not seed_has_match[valid_idx]:
            valid_idx += 1
            continue
        mm = best_match_mask[valid_idx]
        ids_at = best_obs_id[valid_idx][mm]
        deltas_at = best_delta[valid_idx][mm]
        pairs = _torch.stack([
            _torch.from_numpy(ids_at.astype(np.float64, copy=False)),
            _torch.from_numpy(deltas_at.astype(np.float64, copy=False)),
        ], dim=-1).to(ctx.dtype)
        seeds_out.append(SeedResult(
            spot_id=m["sid"],
            best_or_mat=_torch.from_numpy(best_R[valid_idx].copy()),
            best_pos=_torch.from_numpy(best_pos[valid_idx].copy()),
            n_matches=int(best_n_match[valid_idx]),
            n_t_spots=int(best_n_t[valid_idx]),
            n_t_frac_calc=int(best_n_t_frac[valid_idx]),
            frac_matches=float(best_frac[valid_idx]),
            avg_ia=float(best_ia[valid_idx]),
            matched_ids=_torch.from_numpy(ids_at.copy()),
            matched_pairs=pairs,
        ))
        valid_idx += 1
    return seeds_out


def _process_seed_group(
    ctx: IndexerContext,
    seeds_block: torch.Tensor,        # (n_seeds,) CPU int64
) -> list[SeedResult]:
    """Sequential compatibility wrapper: run setup_cpu then compute_gpu.

    Releases setup-dict tensors before returning. Periodically (every
    ``_GC_EVERY_N_SEEDS``) forces a gc + malloc_trim so the CPU torch
    allocator doesn't retain >100 GB on PF-scale data. Per-call gc was
    ~50 ms — far too expensive when seeds are now <100 ms total. Batching
    the gc reduces overhead by ~30× without losing the memory bound (we
    just run a slightly bigger working set between trims).
    """
    setup = _setup_group_cpu(ctx, seeds_block)
    if setup is None:
        return []
    out = _compute_group_gpu(ctx, setup)
    setup.clear()
    del setup
    if ctx.device.type == "cpu":
        _GC_CALL_COUNTER[0] += 1
        if _GC_CALL_COUNTER[0] % _GC_EVERY_N_SEEDS == 0:
            import gc as _gc
            _gc.collect()
            try:
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
    return out


