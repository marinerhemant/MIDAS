"""``FitOrientationOMP`` replacement: per-voxel orientation refinement.

Mirrors the C executable's flow:

1. Parse the paramfile and read all binary inputs.
2. For each voxel in the assigned block:
   a. Phase 1 (hard, no autograd): screen all candidate orientations
      against the observation bitmap, keep those above
      ``MinFracAccept``.
   b. Phase 2 (differentiable, L-BFGS): seed L-BFGS at each surviving
      candidate's Euler angles, optimise within ±``OrientTol`` via
      :class:`TanhBox`, run to convergence.
   c. Track the global best Euler + the top-N unique-by-misorientation.
3. Write ``MicFileBinary`` and ``MicFileBinary.AllMatches`` byte-aligned
   with the C output.

The fit kernel is shared with :mod:`.fit_parameters` and
:mod:`.fit_multipoint` via :func:`.fit_kernel.run_lbfgs`.
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from .fit_kernel import LBFGSConfig, run_lbfgs
from .hard_polish import polish_hard_frac
from .io import GridTable, OrientationData, read_grid, read_hkls, read_orientations
from .obs_volume import ObsVolume
from .output import MicRecord, MicWriter
from .params import FitParams, parse_paramfile
from .reparam import (
    TanhBox,
    euler_zxz_to_quat_np,
    misorientation_deg_symmetric,
    normalize_orient_mat,
    pairwise_miso_deg_vec,
)
from .screen import (
    ScreenResult, build_rot_tilts, orientmat_to_euler_zxz,
    screen, write_screen_csv,
)
from .soft_overlap import (
    GeometryOverrides,
    auto_sigma_px,
    build_forward_model,
    forward_batched_grains,
    soft_overlap_loss,
)
from .torch_nm import batched_nelder_mead
from .triton_kernels import HAS_TRITON, fused_hard_frac


# ---------------------------------------------------------------------------
#  Top-N solution tracker (uniqueness by misorientation)
# ---------------------------------------------------------------------------

class TopNTracker:
    """Maintain the top-N unique-by-misorientation solutions for one voxel.

    Mirrors the insertion logic at FitOrientationOMP.c:1581-1644:
    candidates within ``min_miso_deg`` of an existing entry are dropped;
    survivors are inserted in fraction-overlap order, oldest-with-lowest
    pushed off the end when full.

    The misorientation comparison uses
    :func:`midas_stress.orientation.misorientation` via
    :func:`midas_nf_fitorientation.reparam.misorientation_deg_symmetric`,
    so it applies the full crystal symmetry group of ``space_group`` —
    matching the C path at FitOrientationOMP.c:1602-1604 which calls
    ``GetMisOrientationAngle(... NrSymmetries, Sym)``.
    """

    def __init__(self, n_saves: int, min_miso_deg: float, space_group: int):
        self.n_saves = max(1, int(n_saves))
        self.min_miso_deg = float(min_miso_deg)
        self.space_group = int(space_group)
        # Each entry: (eulA, eulB, eulC, frac)
        self.entries: List[Tuple[float, float, float, float]] = []
        # Parallel cache of fundamental-zone quaternions for each entry,
        # so the symmetry-aware uniqueness check can be a single
        # vectorised numpy call against all existing entries instead of
        # a per-pair midas_stress invocation.
        self._entry_quats: List[np.ndarray] = []

    def offer(self, euler, frac: float) -> None:
        # Accept either a torch tensor (serial paths) or a numpy
        # array / 3-tuple (the nm-batched path pre-syncs the entire
        # batch to host memory in one shot).
        if torch.is_tensor(euler):
            eul_arr = euler.detach().cpu().numpy().astype(np.float64, copy=False)
        else:
            eul_arr = np.asarray(euler, dtype=np.float64)
        eul = (float(eul_arr[0]), float(eul_arr[1]), float(eul_arr[2]))

        # Symmetry-aware uniqueness: drop if within min_miso_deg of any
        # existing entry.  The vectorised numpy pairwise call replaces
        # the previous per-pair midas_stress loop (~17 s on the full Au
        # grid).  ``min_miso_deg <= 0`` skips the check entirely.
        #
        # NOTE: min_miso_deg is 1.0 BY DEFAULT (params.py:135) -- an earlier
        # comment here claimed 0 was the default, which was wrong.
        #
        # With n_saves == 1 the check is skipped: it is meaningless (a single
        # slot cannot hold two "distinct" solutions) and it was actively
        # harmful. offer() returns early when a candidate lands within
        # min_miso_deg of the incumbent, so a LATER candidate with a HIGHER
        # frac was discarded and the worse solution survived into `best`,
        # which is what gets written to the mic (see `best = tracker.best`).
        # It was also the dominant cost: one numpy symmetry-expanded
        # misorientation per candidate window per voxel, which is why a voxel
        # with many surviving windows ran ~5/min instead of thousands/min.
        #
        # (For n_saves > 1 the early-return can still drop a better solution
        # in favour of a near-duplicate worse one. Left alone deliberately --
        # it changes multi-solution results and is out of scope here.)
        if self.n_saves == 1:
            q_new_arr = None          # no quat cache needed; nothing to compare against
        elif self.min_miso_deg > 0.0 and self._entry_quats:
            q_new = euler_zxz_to_quat_np(eul_arr)
            existing = np.asarray(self._entry_quats, dtype=np.float64)
            misos = pairwise_miso_deg_vec(q_new, existing, self.space_group)
            if (misos < self.min_miso_deg).any():
                return
            q_new_arr = q_new
        elif self.min_miso_deg > 0.0:
            q_new_arr = euler_zxz_to_quat_np(eul_arr)
        else:
            q_new_arr = None  # uniqueness disabled, no need to cache

        # Sorted insertion (descending by frac).
        new = (eul[0], eul[1], eul[2], frac)
        inserted = False
        for i, (_, _, _, f) in enumerate(self.entries):
            if frac >= f:
                self.entries.insert(i, new)
                if q_new_arr is not None:
                    self._entry_quats.insert(i, q_new_arr)
                inserted = True
                break
        if not inserted:
            self.entries.append(new)
            if q_new_arr is not None:
                self._entry_quats.append(q_new_arr)
        if len(self.entries) > self.n_saves:
            self.entries.pop()
            if self._entry_quats:
                self._entry_quats.pop()

    @property
    def best(self) -> Optional[Tuple[float, float, float, float]]:
        return self.entries[0] if self.entries else None

    def to_array(self) -> np.ndarray:
        if not self.entries:
            return np.zeros((0, 4), dtype=np.float64)
        return np.asarray(self.entries, dtype=np.float64)


# ---------------------------------------------------------------------------
#  Main driver
# ---------------------------------------------------------------------------

def fit_orientation_run(
    paramfile: str,
    block_nr: int = 0,
    n_blocks: int = 1,
    n_cpus: int = 1,
    *,
    device: str = "auto",
    dtype: torch.dtype = torch.float64,
    screen_only: bool = False,
    verbose: bool = False,
    lbfgs_config: Optional[LBFGSConfig] = None,
    voxel_indices: Optional[np.ndarray] = None,
    create_output: bool = True,
    refine: str = "nm-batched",
    nm_max_iter: int = 200,
    nm_batch_size: int = 4096,
) -> str:
    """
    Parameters
    ----------
    refine : {"nm-triton", "nm-batched", "nm-serial", "lbfgs+nm", "lbfgs"}, default ``"nm-batched"``
        Phase-2 refinement strategy:

        - ``"nm-triton"`` (CUDA only, fastest): replaces the eager-
          PyTorch fn closure with one fused Triton kernel that does
          Bragg + projection + tilts + obs lookup in a single launch.
          Eliminates the ~60 small kernels the eager path issues per
          fn call. Auto-selects when ``device='cuda'`` and Triton is
          installed; falls back to ``"nm-batched"`` otherwise.
        - ``"nm-batched"`` (default for non-CUDA): vectorised PyTorch
          Nelder-Mead that runs every ``(voxel, winner)`` fit problem
          in one batched forward call per NM iteration. ~10–20×
          faster than the serial path on GPU; same hard-FracOverlap
          objective. Recommended for production on CPU; superseded by
          ``"nm-triton"`` on CUDA.
        - ``"nm-serial"``: per-winner ``scipy.optimize.minimize`` on
          the hard objective, one Python loop per voxel. Used as a
          parity oracle for the batched path. Same numerical result
          modulo NM convergence noise.
        - ``"lbfgs+nm"``: L-BFGS warmup on the soft Gaussian-splat
          surrogate, then NM polish on the hard objective from the
          warmed point. The L-BFGS step can move the orientation into
          a different basin than C lands in (the soft-vs-hard floor
          differs); kept for ablation only.
        - ``"lbfgs"``: legacy soft-only path, kept for comparison.
          **Not recommended for production.**
    nm_batch_size : int, default 4096
        Maximum number of ``(voxel, winner)`` problems run through one
        batched NM call. The whole block is split into chunks of this
        size to bound peak GPU memory; 4096 fits comfortably on a 16
        GB device with NF-typical M (~250 hkls) and D (1–4) values.
    """
    """Run ``FitOrientationOMP``-equivalent on the given paramfile + block.

    Parameters
    ----------
    paramfile : str
        Path to the C-style parameter file.
    block_nr, n_blocks : int
        Block decomposition; voxel range is computed by
        :meth:`GridTable.slice_block`.
    n_cpus : int
        For backward compat with the C CLI; if a CUDA device is
        available we ignore this (single-process GPU). On CPU we
        currently leave OpenMP-style parallelism to PyTorch's default
        thread-pool (``OMP_NUM_THREADS``).
    device : str
        ``"auto"``, ``"cpu"``, or ``"cuda"``.
    dtype : torch.dtype
        Forward-model dtype (float64 default; pass ``torch.float32``
        for the ``--fp32`` opt-in path).
    screen_only : bool
        If True, write a ``screen_cpu.csv`` diagnostic dump and skip
        Phase 2 fitting. Mirrors the ``MIDAS_SCREEN_ONLY`` env var in
        the C code.

    Returns
    -------
    str
        Path to the ``MicFileBinary`` output.
    """
    p = parse_paramfile(paramfile)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    # Output dir
    out_dir = Path(p.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not p.mic_file_binary:
        raise ValueError("paramfile is missing required key 'MicFileBinary'")
    mic_path = out_dir / Path(p.mic_file_binary).name

    # ---- read inputs ----
    grid = read_grid(out_dir, p.grid_file_name)
    orientations = read_orientations(out_dir)
    hkl_table = read_hkls(out_dir)
    if p.rings_to_use:
        hkl_table = hkl_table.filter_rings(p.rings_to_use)

    obs_path = out_dir / "SpotsInfo.bin"
    # The default v0.4 storage is **packed bits** (1 bit per pixel,
    # 32× smaller than float32 dense — 750 MB vs 24 GB on the
    # bundled Au example). ``hard_fraction`` extracts bits via
    # right-shift + AND on the fly. The L-BFGS soft-overlap path
    # needs a *dense float* volume, so we fall back to ``float32``
    # when one of the L-BFGS-using ``refine`` modes is requested.
    use_soft = refine in ("lbfgs", "lbfgs+nm")
    if use_soft:
        obs = ObsVolume.from_spotsinfo(
            obs_path,
            n_distances=p.n_distances,
            n_frames=p.n_frames_per_distance,
            n_y=p.n_pixels_y, n_z=p.n_pixels_z,
            device=torch_device, dtype=torch.float32, packed=False,
        )
    else:
        obs = ObsVolume.from_spotsinfo(
            obs_path,
            n_distances=p.n_distances,
            n_frames=p.n_frames_per_distance,
            n_y=p.n_pixels_y, n_z=p.n_pixels_z,
            device=torch_device, packed=True,
        )

    # ---- forward model ----
    model = build_forward_model(
        p, hkl_table.hkls_int.astype(np.float64),
        device=torch_device, dtype=dtype,
        hkls_cart=hkl_table.hkls_cart.astype(np.float64),
    )

    # ---- voxel selection ----
    # Explicit ``voxel_indices`` argument wins (used by integration
    # tests and for stratified parity sampling); otherwise fall back
    # to the C-style block decomposition.
    if voxel_indices is None:
        start, end_inclusive = grid.slice_block(block_nr, n_blocks)
        voxel_indices = np.arange(start, end_inclusive + 1)
        if verbose:
            print(f"Block {block_nr}/{n_blocks}: voxels [{start}, {end_inclusive}]"
                  f" ({len(voxel_indices)} voxels)")
    else:
        voxel_indices = np.asarray(voxel_indices, dtype=np.int64)
        if verbose:
            print(f"Custom voxel selection: {len(voxel_indices)} voxels")

    # ---- Phase 1: screen ----
    t_screen = time.perf_counter()
    screen_result = screen(
        grid, orientations, obs, p,
        voxel_indices=voxel_indices, dtype=dtype,
    )
    screen_secs = time.perf_counter() - t_screen
    if verbose:
        print(f"Screen: {len(screen_result.winners)} winners "
              f"in {screen_secs:.2f} s")

    if screen_only:
        write_screen_csv(screen_result, str(out_dir / "screen_cpu.csv"))
        return str(mic_path)

    # Bucketise winners by voxel for the fit phase
    winners_by_voxel: dict[int, list] = {}
    for w in screen_result.winners:
        winners_by_voxel.setdefault(w.voxel_idx, []).append(w)

    # ---- Phase 2 ----
    cfg = lbfgs_config or LBFGSConfig()
    eulers_seed = orientmat_to_euler_zxz(orientations.matrices)
    tol_rad = p.orient_tol * math.pi / 180.0

    # Pre-compute one batched NM result per chunk if we're on the
    # ``"nm-batched"`` path. Map ``(voxel_idx, winner_position) ->
    # (final_eul, hard_frac)`` so the per-voxel writer block below
    # can read them out.
    batched_results: "dict[tuple[int, int], tuple[torch.Tensor, float]]" = {}
    t_batch = t_batch_done = time.perf_counter()
    if refine == "nm-batched":
        t_batch = time.perf_counter()
        problems: list[tuple[int, int]] = []   # (voxel_idx, position-in-wins)
        seeds_l: list[np.ndarray] = []
        positions_l: list[list[float]] = []
        for vi in voxel_indices:
            wins = winners_by_voxel.get(int(vi), [])
            xs = float(grid.xs[vi])
            ys = float(grid.ys[vi])
            for w_idx, w in enumerate(wins):
                problems.append((int(vi), w_idx))
                seeds_l.append(eulers_seed[w.orient_idx])
                positions_l.append([xs, ys, 0.0])

        if problems:
            seeds_t = torch.tensor(
                np.asarray(seeds_l, dtype=np.float64),
                device=torch_device, dtype=dtype,
            )
            positions_t = torch.tensor(
                np.asarray(positions_l, dtype=np.float64),
                device=torch_device, dtype=dtype,
            )
            B_total = seeds_t.shape[0]

            # Per-problem ±OrientTol bounds. Same width for every
            # problem; using the (n_dim, 2) form lets ``torch_nm``
            # broadcast it across the batch.
            uniform_bounds = torch.tensor(
                [[-tol_rad, tol_rad]] * 3,
                device=torch_device, dtype=dtype,
            )
            # We build per-problem bounds by adding seed offsets
            # because the C box is centred on each seed, not the origin.
            bounds_t = torch.empty(B_total, 3, 2, device=torch_device, dtype=dtype)
            bounds_t[..., 0] = seeds_t - tol_rad
            bounds_t[..., 1] = seeds_t + tol_rad

            # Pre-build Triton-friendly fp32 constants if we're
            # going to use the fused kernel. Cheap; the launch sites
            # do not retouch them.
            # The fused Triton kernel implements the paired OmegaRange/BoxSize
            # gate (triton_kernels.py, ``NBOX``), matching
            # midas_diffract.HEDMForwardModel.omega_box_mask and the C
            # CalcDiffrSpots_Furnace gate. The (NBOX, 6) table below is passed
            # straight through; NBOX=0 leaves the kernel bit-identical to the
            # pre-gate version.
            box_gate_active = bool(getattr(model, "_has_omega_box", False))
            ome_box_tbl = None
            if box_gate_active:
                _omr = model._omega_ranges.to(torch.float32).reshape(-1, 2)
                _box = model._box_sizes.to(torch.float32).reshape(-1, 4)
                ome_box_tbl = torch.cat([_omr, _box], dim=1).contiguous()
            use_triton = (
                refine == "nm-triton" or (
                    refine == "nm-batched"
                    and torch_device.type == "cuda"
                    and HAS_TRITON
                    and obs.packed is not None
                )
            )
            if use_triton:
                if obs.packed is None:
                    raise RuntimeError(
                        "Triton fast path requires a packed-bit ObsVolume; "
                        "rerun with the default packed loader."
                    )
                # midas-diffract registers hkls/thetas as float32 buffers
                # internally — read them out and reuse for the kernel.
                hkls_f32 = model.hkls.contiguous().to(torch.float32)
                thetas_f32 = model.thetas.contiguous().to(torch.float32)
                Lsd_f32 = torch.tensor(p.Lsd, device=torch_device, dtype=torch.float32)
                ybc_f32 = torch.tensor(p.ybc, device=torch_device, dtype=torch.float32)
                zbc_f32 = torch.tensor(p.zbc, device=torch_device, dtype=torch.float32)
                has_tilts_run = bool(p.tx != 0 or p.ty != 0 or p.tz != 0)
                if has_tilts_run:
                    R_per_d = build_rot_tilts(
                        p.tx, p.ty, p.tz, torch_device, torch.float32,
                    )
                    R_tilt_flat = (
                        R_per_d.reshape(9).repeat(p.n_distances, 1)
                        if False else
                        R_per_d.reshape(1, 9).expand(p.n_distances, 9).contiguous()
                    )
                else:
                    R_tilt_flat = torch.zeros(
                        p.n_distances, 9, device=torch_device, dtype=torch.float32,
                    )
                has_wedge_run = bool(p.wedge != 0)
                wedge_rad_run = float(p.wedge) * (math.pi / 180.0)
                obs_packed = obs.packed.to(torch.uint8)

            def _batched_neg_hard_frac_factory(positions_chunk):
                if use_triton:
                    pos_chunk_f32 = positions_chunk.to(torch.float32)
                    def _neg_hard_frac(eul_batch, idx_batch):
                        with torch.no_grad():
                            pos_active = pos_chunk_f32[idx_batch].contiguous()
                            eul_f32 = eul_batch.to(torch.float32).contiguous()
                            frac = fused_hard_frac(
                                eul_f32, pos_active,
                                hkls_f32, thetas_f32,
                                Lsd_f32, ybc_f32, zbc_f32,
                                R_tilt_flat, obs_packed,
                                px=p.px,
                                wedge_rad=wedge_rad_run,
                                omega_start_deg=p.omega_start,
                                omega_step_deg=p.omega_step,
                                min_eta_rad=p.exclude_pole_angle * (math.pi / 180.0),
                                n_frames=p.n_frames_per_distance,
                                n_y=p.n_pixels_y,
                                n_z=p.n_pixels_z,
                                has_tilts=has_tilts_run,
                                has_wedge=has_wedge_run,
                                ome_box=ome_box_tbl,
                            )
                        return (1.0 - frac).to(eul_batch.dtype)
                    return _neg_hard_frac

                # Eager-PyTorch fallback.
                def _neg_hard_frac(eul_batch, idx_batch):
                    with torch.no_grad():
                        pos_active = positions_chunk[idx_batch]
                        fn_, val, yp, zp = forward_batched_grains(
                            model, eul_batch, pos_active,
                        )
                        frac = obs.hard_fraction(fn_, yp, zp, val)
                    return 1.0 - frac
                return _neg_hard_frac

            # Chunk to bound peak GPU memory.
            n_chunks = max(1, -(-B_total // nm_batch_size))
            xs_out = torch.empty_like(seeds_t)
            f_out = torch.empty(B_total, device=torch_device, dtype=dtype)
            for ck in range(n_chunks):
                lo = ck * nm_batch_size
                hi = min(lo + nm_batch_size, B_total)
                chunk_seeds = seeds_t[lo:hi]
                chunk_positions = positions_t[lo:hi]
                chunk_bounds = bounds_t[lo:hi]
                fn = _batched_neg_hard_frac_factory(chunk_positions)
                res = batched_nelder_mead(
                    fn, chunk_seeds, chunk_bounds,
                    max_iter=nm_max_iter,
                )
                xs_out[lo:hi] = res.x
                f_out[lo:hi] = res.fun

            # Cache per (voxel, winner-pos).  We pull the whole batch
            # back to host memory once (single sync) instead of doing
            # a per-problem ``.item()`` /  ``.cpu()`` later in the
            # writeback loop — that path was costing ~10 s on the full
            # Au grid because every TopNTracker.offer triggered a
            # device sync.
            xs_out_np = xs_out.detach().cpu().numpy()
            f_out_np = f_out.detach().cpu().numpy()
            for i, key in enumerate(problems):
                batched_results[key] = (
                    xs_out_np[i],
                    float(1.0 - f_out_np[i]),
                )

        t_batch_done = time.perf_counter()
        if verbose:
            print(f"NM batched: {len(problems)} problems in "
                  f"{t_batch_done - t_batch:.2f} s")

    t_writeback = time.perf_counter()
    with MicWriter(
        mic_path, n_voxels=grid.n_voxels,
        n_saves=p.save_n_solutions, block_nr=block_nr,
        create=create_output,
    ) as writer:
        for vi in voxel_indices:
            xs = float(grid.xs[vi])
            ys = float(grid.ys[vi])
            gs = float(grid.gs[vi])
            ud = float(grid.ud[vi])
            grid_size = 2.0 * gs
            sigma_px = auto_sigma_px(gs, p.px, p.gaussian_splat_sigma_px)

            wins = winners_by_voxel.get(int(vi), [])
            n_winners = len(wins)
            if n_winners == 0:
                writer.write_mic(
                    int(vi),
                    MicRecord(0, 0, 0, xs, ys, grid_size, ud, 0, 0, 0, 0),
                )
                writer.write_all_matches(
                    int(vi), 0, xs, ys, grid_size, ud,
                    np.zeros((0, 4), dtype=np.float64),
                )
                continue

            tracker = TopNTracker(
                p.save_n_solutions, p.min_miso_n_saves_deg, p.space_group,
            )
            best_row_nr = 0.0
            best_hard_frac = -1.0

            t_fit = time.perf_counter()
            for w_idx, w in enumerate(wins):
                if refine == "nm-batched":
                    # Result already computed by the pre-pass.
                    final_eul, hard_frac = batched_results[
                        (int(vi), w_idx)
                    ]
                    tracker.offer(final_eul, hard_frac)
                    if hard_frac > best_hard_frac:
                        best_hard_frac = hard_frac
                        best_row_nr = float(w.orient_idx)
                    if (p.save_n_solutions == 1
                            and hard_frac > 1.0 - 1e-4):
                        break
                    continue

                # Serial paths ("nm-serial", "lbfgs+nm", "lbfgs").
                euler_seed = torch.tensor(
                    eulers_seed[w.orient_idx], device=torch_device, dtype=dtype,
                )
                pos_um = torch.tensor(
                    [xs, ys, 0.0], device=torch_device, dtype=dtype,
                )

                if refine == "lbfgs" or refine == "lbfgs+nm":
                    box = TanhBox(euler_seed, tol_rad)

                    def closure():
                        box.u.grad = None
                        eul = box.x
                        loss = soft_overlap_loss(
                            model, obs, eul, pos_um, sigma_px,
                        )
                        loss.backward()
                        return loss

                    run_lbfgs(closure, [box.u], cfg)
                    warmed_eul = box.x.detach()
                else:
                    warmed_eul = euler_seed

                if refine in ("nm-serial", "lbfgs+nm"):
                    polish = polish_hard_frac(
                        model, obs, warmed_eul, pos_um, tol_rad,
                        max_iter=nm_max_iter,
                    )
                    final_eul = polish.eul
                    hard_frac = polish.hard_frac
                else:
                    final_eul = warmed_eul
                    with torch.no_grad():
                        spots = model(final_eul.unsqueeze(0), pos_um.unsqueeze(0))
                        hard_frac = float(obs.hard_fraction(
                            spots.frame_nr, spots.y_pixel, spots.z_pixel, spots.valid,
                        ))

                tracker.offer(final_eul, hard_frac)
                if hard_frac > best_hard_frac:
                    best_hard_frac = hard_frac
                    best_row_nr = float(w.orient_idx)

                if (p.save_n_solutions == 1
                        and hard_frac > 1.0 - 1e-4):
                    break

            fit_secs = time.perf_counter() - t_fit

            best = tracker.best
            if best is None:
                rec = MicRecord(
                    best_row_nr, n_winners, fit_secs,
                    xs, ys, grid_size, ud, 0, 0, 0, 0,
                )
            else:
                rec = MicRecord(
                    best_row_nr, n_winners, fit_secs,
                    xs, ys, grid_size, ud,
                    best[0], best[1], best[2], best[3],
                )
            writer.write_mic(int(vi), rec)
            writer.write_all_matches(
                int(vi), n_winners, xs, ys, grid_size, ud, tracker.to_array(),
            )

            if verbose:
                print(f"Voxel {vi}: {n_winners} winners → frac {best[3]:.4f}"
                      if best else f"Voxel {vi}: {n_winners} winners, no fit")

    writeback_secs = time.perf_counter() - t_writeback
    print(f"PERF: screen={screen_secs:.2f}s  nm_batched={t_batch_done - t_batch:.2f}s  writeback={writeback_secs:.2f}s")

    return str(mic_path)
