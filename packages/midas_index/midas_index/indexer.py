"""Public library API: the `Indexer` class.

Two construction paths:
  - `Indexer.from_param_file("paramstest.txt", device=...)` — file-driven.
  - `Indexer(params, device=..., dtype=...)` — programmatic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, List

import numpy as np
import torch

from .device import apply_cpu_threads, resolve_device, resolve_dtype
from .params import IndexerParams

if TYPE_CHECKING:
    from .result import IndexerResult


class Indexer:
    """Top-level entry point. Wraps the full pipeline."""

    def __init__(
        self,
        params: IndexerParams,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
    ) -> None:
        self.params = params
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(self.device, dtype)
        self._observations: dict | None = None
        # Source paramstest path — set by from_param_file(), used as the
        # default for backend="c-omp" so the user doesn't have to repeat it.
        self._param_path: Path | None = None

    @classmethod
    def from_param_file(
        cls,
        path: str | os.PathLike,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
    ) -> "Indexer":
        from .io.params import read_params

        inst = cls(read_params(path), device=device, dtype=dtype)
        inst._param_path = Path(path).resolve()
        return inst

    # ------------------------------------------------------------------
    # Loading observations (file-driven or programmatic)
    # ------------------------------------------------------------------

    def load_observations(
        self,
        cwd: str | Path | None = None,
        *,
        spots: np.ndarray | torch.Tensor | None = None,
        bins: tuple[np.ndarray, np.ndarray] | None = None,
        hkls: tuple[np.ndarray, np.ndarray] | None = None,
        spot_ids: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        """Load Spots.bin, Data.bin, nData.bin, hkls.csv, SpotsToIndex.csv.

        File-driven: pass `cwd` (the directory containing the binaries; this
        defaults to `dirname(OutputFolder)` per IndexerOMP.c:2230). All other
        kwargs override the on-disk file with explicit data (useful for
        synthetic / unit-test cases).
        """
        from .io import (
            read_bins,
            read_grains_csv,
            read_hkls_csv,
            read_spots,
            read_spots_to_index_csv,
            write_spots_to_index_csv,
        )
        from .io.binary import read_bins_scanning

        if cwd is None:
            cwd = os.path.dirname(self.params.OutputFolder.rstrip("/")) or "."
        cwd = Path(cwd)

        if spots is None:
            _, spots = read_spots(cwd)
        if bins is None:
            # PF / scanning fixtures emit Data.bin + nData.bin as int64 with
            # (spot_id, scan_nr) pairs / (count, offset) pairs respectively
            # (SaveBinDataScanning.c:672-700). FF fixtures use int32.
            # Disambiguate via the Spots.bin column count: 10-col = PF.
            spots_arr = np.asarray(spots)
            n_cols = spots_arr.shape[1] if spots_arr.ndim == 2 else 0
            if n_cols >= 10:
                bins = read_bins_scanning(cwd)
            else:
                bins = read_bins(cwd)
        if hkls is None:
            hkls = read_hkls_csv("hkls.csv", ring_numbers=self.params.RingNumbers)

        if spot_ids is None:
            sti = cwd / "SpotsToIndex.csv"
            if not sti.exists() and self.params.isGrainsInput:
                # Mode A: derive SpotsToIndex.csv from Grains.csv
                grains_path = self.params.GrainsFileName
                if not Path(grains_path).is_absolute():
                    grains_path = str(cwd / grains_path)
                grains = read_grains_csv(grains_path)
                # Default mode-A row layout: (newID=grainID, origID=grainID)
                pairs = [(int(g), int(g)) for g in grains["ids"]]
                write_spots_to_index_csv(sti, pairs)
            spot_ids = read_spots_to_index_csv(sti)

        self._observations = {
            "spots": np.asarray(spots),
            "bin_data": np.asarray(bins[0]),
            "bin_ndata": np.asarray(bins[1]),
            "hkls_real": np.asarray(hkls[0]),
            "hkls_int": np.asarray(hkls[1]),
            "spot_ids": np.asarray(spot_ids).astype(np.int64),
        }

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        block_nr: int = 0,
        n_blocks: int = 1,
        n_spots_to_index: int | None = None,
        num_procs: int = 1,
        seed_group_size: int | None = None,
        backend: str = "python",
        paramstest_path: str | os.PathLike | None = None,
    ) -> "IndexerResult":
        """Run the indexer on `[block_nr/n_blocks]` of the seed list.

        Parameters
        ----------
        backend
            ``"python"`` (default) — the in-process numba/torch indexer.
            ``"c-omp"`` — shell out to the bundled unified C binary
            (``midas_indexer``, built by scikit-build-core at pip-install
            time). The C path writes consolidated output files
            (``IndexBest_all.bin``, etc.) under ``Params.OutputFolder`` and
            returns a minimal :class:`IndexerResult` with ``seeds=[]`` —
            downstream stages (find_grains, refinement) read those files
            directly from disk per the same contract as the legacy C path.
        paramstest_path
            Required when ``backend="c-omp"`` unless this indexer was
            built via :meth:`from_param_file` (in which case the source
            path is reused). The C binary reads this file.
        """
        if backend == "c-omp":
            return self._run_c_omp(
                block_nr=block_nr, n_blocks=n_blocks,
                n_spots_to_index=n_spots_to_index, num_procs=num_procs,
                paramstest_path=paramstest_path,
            )
        if backend != "python":
            raise ValueError(
                f"unknown backend {backend!r}; expected 'python' or 'c-omp'"
            )
        from .pipeline import IndexerContext, run_block

        if self._observations is None:
            self.load_observations()
        obs = self._observations
        assert obs is not None

        apply_cpu_threads(num_procs, self.device)

        ctx = IndexerContext(
            params=self.params,
            hkls_real=obs["hkls_real"],
            hkls_int=obs["hkls_int"],
            obs=obs["spots"],
            bin_data=obs["bin_data"],
            bin_ndata=obs["bin_ndata"],
            device=self.device,
            dtype=self.dtype,
        )

        spot_ids = torch.as_tensor(obs["spot_ids"], dtype=torch.int64)
        if n_spots_to_index is not None:
            spot_ids = spot_ids[:n_spots_to_index]
        return run_block(ctx, spot_ids, block_nr=block_nr, n_blocks=n_blocks,
                         seed_group_size=seed_group_size)

    # ------------------------------------------------------------------
    # Scan-aware run (pf-HEDM)
    # ------------------------------------------------------------------

    def run_scanning(
        self,
        scan_positions: np.ndarray | torch.Tensor,
        *,
        out_path: str | Path,
        n_spots_to_index: int | None = None,
        num_procs: int = 1,
        seed_group_size: int | None = None,
        voxel_block_nr: int = 0,
        voxel_n_blocks: int = 1,
        backend: str = "python",
        paramstest_path: str | os.PathLike | None = None,
    ) -> int:
        """Run the per-voxel scanning indexer (pf-HEDM mode).

        Iterates over the (n_scans × n_scans) voxel grid built as the
        Cartesian product of ``scan_positions`` (1-D Y values, µm). For
        each voxel, sets the scan-aware kwargs on ``IndexerContext`` and
        runs the full seed pipeline; collects each voxel's solutions
        into a list. After the loop, writes the consolidated
        ``IndexBest_all.bin`` per the C
        ``IndexerScanningOMP``/``IndexerConsolidatedIO.h`` byte layout.

        Notes
        -----
        - The voxel grid layout matches
          ``IndexerScanningOMP.c:1667-1683``: ``grid[i*nScans + j] =
          (scan_positions[j], scan_positions[i])`` — Cartesian product
          of sorted 1-D Y positions (scan-axis is Y only per P0 audit).
        - ``params.scan_pos_tol_um`` and
          ``params.friedel_symmetric_scan_filter`` drive the per-voxel
          filter inside ``compare_spots``. ``scan_pos_tol_um == 0`` ⇒
          filter inactive (degenerates to FF behavior per voxel, useful
          for sanity).
        - For very large grids the per-voxel cost is significant — call
          with ``voxel_block_nr/voxel_n_blocks > 1`` to shard.

        Returns
        -------
        int
            Number of voxels processed (== ``end - start`` over the
            sharded range).
        """
        if backend == "c-omp":
            return self._run_scanning_c_omp(
                scan_positions=scan_positions, num_procs=num_procs,
                voxel_block_nr=voxel_block_nr, voxel_n_blocks=voxel_n_blocks,
                paramstest_path=paramstest_path,
            )
        if backend != "python":
            raise ValueError(
                f"unknown backend {backend!r}; expected 'python' or 'c-omp'"
            )
        from .pipeline import IndexerContext, run_block
        from .io.consolidated import write_index_best_all

        # 1. Validate scan positions BEFORE touching the context — we'd
        # rather fail with a clear ValueError than dive into the
        # pipeline's IndexerContext constructor which may need
        # configured params (EtaBinSize etc.).
        scan_positions_t = torch.as_tensor(
            np.asarray(scan_positions), dtype=self.dtype, device=self.device,
        ).view(-1)
        # MUST sort ascending — C IndexerScanningOMP.c:1676 does
        # ``qsort(ypos_sorted, numScans, sizeof(double), cmp_double_asc)``
        # before building the voxel grid. Some PF runs (e.g. Wenxi
        # CP-Ti) ship positions.csv in DESCENDING order; without the
        # sort, voxel positions get sign-flipped vs C.
        scan_positions_t, _ = torch.sort(scan_positions_t)
        n_scans = int(scan_positions_t.numel())
        if n_scans < 2:
            raise ValueError(
                f"run_scanning requires n_scans >= 2; got {n_scans}. "
                "Use run() for the FF (single-scan) case."
            )

        if self._observations is None:
            self.load_observations()
        obs = self._observations
        assert obs is not None
        apply_cpu_threads(num_procs, self.device)

        # 2. Build context.
        ctx = IndexerContext(
            params=self.params,
            hkls_real=obs["hkls_real"],
            hkls_int=obs["hkls_int"],
            obs=obs["spots"],
            bin_data=obs["bin_data"],
            bin_ndata=obs["bin_ndata"],
            device=self.device,
            dtype=self.dtype,
        )
        ctx.scan_positions = scan_positions_t
        # P6/P8: forward the optional soft-attribution weight fn if the caller
        # set it on this Indexer instance.  None ⇒ legacy binary scan filter.
        ctx.soft_beam_weight_fn = getattr(self, "soft_beam_weight_fn", None)

        # 3. Build voxel grid: nVox = nScans * nScans, with the per-voxel
        # (x, y) sample-frame coordinates matching
        # IndexerScanningOMP.c:1676-1684 + :1731-1732:
        #     grid[(i*nScans + j) * 2 + 0] = ypos_sorted[i]   → xThis
        #     grid[(i*nScans + j) * 2 + 1] = ypos_sorted[j]   → yThis
        # For voxel v = i*nScans + j: voxel_xy[v] = (ypos[i], ypos[j]).
        # The C code's ga/gb (sample-frame x/y the forward sim sees) are
        # (xThis, yThis), and the scan filter uses
        #     yRot = xThis*sin(ω) + yThis*cos(ω)
        # so this ordering pins both the position and the filter.
        idx = torch.arange(n_scans * n_scans, device=self.device)
        i_idx = idx // n_scans
        j_idx = idx % n_scans
        voxel_xy_table = torch.stack(
            [scan_positions_t[i_idx], scan_positions_t[j_idx]], dim=-1,
        )                                     # (nVox, 2)
        n_vox = int(voxel_xy_table.shape[0])

        # 4. Voxel sharding (used for cluster runs).
        # Env-var overrides for ad-hoc subsetting without changing the call site:
        #   MIDAS_INDEX_VOXEL_BLOCK_NR / MIDAS_INDEX_VOXEL_N_BLOCKS — pick a block.
        #   MIDAS_INDEX_MAX_VOXELS — cap voxel count from v_start (overrides block).
        env_block = os.environ.get("MIDAS_INDEX_VOXEL_BLOCK_NR")
        env_n = os.environ.get("MIDAS_INDEX_VOXEL_N_BLOCKS")
        if env_block is not None:
            voxel_block_nr = int(env_block)
        if env_n is not None:
            voxel_n_blocks = int(env_n)
        if voxel_n_blocks < 1 or voxel_block_nr < 0 or voxel_block_nr >= voxel_n_blocks:
            raise ValueError(
                f"invalid voxel sharding: block={voxel_block_nr}, n={voxel_n_blocks}"
            )
        block_size = (n_vox + voxel_n_blocks - 1) // voxel_n_blocks
        v_start = voxel_block_nr * block_size
        v_end = min(v_start + block_size, n_vox)
        env_max = os.environ.get("MIDAS_INDEX_MAX_VOXELS")
        if env_max is not None:
            v_end = min(v_start + int(env_max), v_end)

        # 5. Build the per-voxel seed pool. Scanning mode mirrors
        # IndexerScanningOMP.c:1687-1693 + 1786-1793: the seed pool is ALL
        # obs spots with ObsSpotsLab[5] == RingToIndex. SpotsToIndex.csv is
        # NOT consulted in scanning mode (the C scanning indexer doesn't
        # read it either). Falls back to obs["spot_ids"] only when
        # RingToIndex==0 (older configs that didn't set it).
        obs_np = np.asarray(obs["spots"])
        if obs_np.shape[1] < 10:
            raise ValueError(
                "scanning indexer needs 10-col Spots.bin (PF layout); got "
                f"{obs_np.shape[1]} cols. Check Spots.bin emitter."
            )
        ring_to_index = int(getattr(self.params, "RingToIndex", 0))
        if ring_to_index > 0:
            ring_mask = obs_np[:, 5].astype(np.int64) == ring_to_index
            seed_obs_rows_np = np.flatnonzero(ring_mask).astype(np.int64)
            seed_ids_np = obs_np[seed_obs_rows_np, 4].astype(np.int64)
        else:
            seed_ids_np = np.asarray(obs["spot_ids"]).astype(np.int64)
            if n_spots_to_index is not None:
                seed_ids_np = seed_ids_np[:n_spots_to_index]
            obs_id_to_row = {
                int(v): i for i, v in enumerate(obs_np[:, 4].astype(np.int64))
            }
            seed_obs_rows_np = np.array(
                [obs_id_to_row.get(int(sid), -1) for sid in seed_ids_np],
                dtype=np.int64,
            )

        # 5b. Per-seed scan-aware pre-filter (mirrors IndexerScanningOMP.c:1786-1793).
        # Builds (omega_rad, scan_y_obs) for every seed once, before the voxel
        # loop. Per voxel we then compute s_proj = x*sin(omega) + y*cos(omega)
        # vectorised over seeds, keep seeds with |s_proj - scan_y_obs| <= tol.
        # Without this pre-filter, every voxel re-runs ALL seeds through the
        # full orientation grid + compare_spots, even those that the C
        # reference would have rejected in O(1) at the outer loop — dominant
        # perf hotspot in the per-voxel solve.
        seed_has_obs = seed_obs_rows_np >= 0
        seed_obs_rows_safe = np.where(seed_has_obs, seed_obs_rows_np, 0)
        seed_omega_deg = obs_np[seed_obs_rows_safe, 2]
        seed_scan_nr = obs_np[seed_obs_rows_safe, 9].astype(np.int64)
        seed_omega_rad_np = np.deg2rad(seed_omega_deg)
        seed_sin_ome = np.sin(seed_omega_rad_np)
        seed_cos_ome = np.cos(seed_omega_rad_np)
        scan_positions_np = scan_positions_t.cpu().numpy().astype(np.float64)
        n_scans_pos = scan_positions_np.size
        seed_scan_nr_clamped = np.clip(seed_scan_nr, 0, n_scans_pos - 1)
        seed_scan_y_obs = scan_positions_np[seed_scan_nr_clamped]
        scan_pos_tol = float(ctx.scan_pos_tol_um)
        friedel_sym = bool(ctx.friedel_symmetric_scan_filter)
        pre_filter_enabled = scan_pos_tol > 0
        voxel_xy_np = voxel_xy_table.cpu().numpy().astype(np.float64)

        # 6. Per-voxel loop. Collect each voxel's seed results into a
        # (n_solutions, 16) float64 record block matching the C
        # IndexerScanningOMP consolidated layout. Cols 11-13 (posX/Y/Z) are
        # written as the voxel center per IndexerScanningOMP.c — NOT the
        # refined per-seed position. PF refinement (midas-fit-grain) takes
        # voxel center as a starting point and refines downstream.
        per_voxel_records: list[np.ndarray] = [
            np.zeros((0, 16), dtype=np.float64) for _ in range(n_vox)
        ]
        per_voxel_keys: list[np.ndarray] = [
            np.zeros((0, 4), dtype=np.uint64) for _ in range(n_vox)
        ]
        per_voxel_ids: list[np.ndarray] = [
            np.zeros((0,), dtype=np.int32) for _ in range(n_vox)
        ]
        import time as _time
        progress_env = os.environ.get("MIDAS_INDEX_PROGRESS", "1")
        progress_on = progress_env not in ("0", "false", "False", "")
        report_every = max(1, int(os.environ.get("MIDAS_INDEX_PROGRESS_EVERY", "20")))
        # Inter-voxel parallelism: when MIDAS_INDEX_INTER_VOXEL_WORKERS > 1,
        # voxels are dispatched to a fork-based mp.Pool. Each worker inherits
        # the parent's `ctx` (via fork's COW) and mutates only its own copy of
        # `current_voxel_xy`. Numba @njit kernels load from the on-disk cache
        # (cache=True) so worker JIT cost is small. NUMBA_NUM_THREADS in each
        # worker should be tuned so workers don't oversubscribe the CPU.
        n_workers = max(1, int(os.environ.get("MIDAS_INDEX_INTER_VOXEL_WORKERS", "1")))
        # Build (v, vx, vy, seeds_np) task list up-front so we don't pickle ctx.
        tasks: list[tuple[int, int, float, float, np.ndarray]] = []
        for vi, v in enumerate(range(v_start, v_end)):
            vx, vy = voxel_xy_np[v]
            if pre_filter_enabled:
                s_proj = vx * seed_sin_ome + vy * seed_cos_ome
                diff = np.abs(s_proj - seed_scan_y_obs)
                ok = diff <= scan_pos_tol
                if friedel_sym:
                    diff_friedel = np.abs(s_proj + seed_scan_y_obs)
                    ok = ok | (diff_friedel <= scan_pos_tol)
                ok = ok & seed_has_obs
                if not ok.any():
                    continue
                voxel_seeds_ids = seed_ids_np[ok]
            else:
                voxel_seeds_ids = seed_ids_np
            tasks.append((vi, v, float(vx), float(vy), voxel_seeds_ids))
        n_target_active = len(tasks)
        n_target = v_end - v_start
        t_start = _time.monotonic()
        if n_workers <= 1:
            # Serial path (original behavior).
            for ti, (vi, v, vx, vy, voxel_seeds_ids) in enumerate(tasks):
                ctx.current_voxel_xy = voxel_xy_table[v]
                t_v0 = _time.monotonic()
                voxel_seeds = torch.as_tensor(voxel_seeds_ids, dtype=torch.int64)
                voxel_result = run_block(
                    ctx, voxel_seeds,
                    block_nr=0, n_blocks=1,
                    seed_group_size=seed_group_size,
                )
                vox_rec, vox_keys, vox_ids = _seeds_to_record_block(
                    voxel_result, voxel_xyz=(vx, vy, 0.0),
                )
                per_voxel_records[v] = vox_rec
                per_voxel_keys[v] = vox_keys
                per_voxel_ids[v] = vox_ids
                if progress_on and ((ti + 1) % report_every == 0 or ti + 1 == n_target_active):
                    now = _time.monotonic()
                    rate = (ti + 1) / max(now - t_start, 1e-9)
                    eta_s = (n_target_active - (ti + 1)) / max(rate, 1e-9)
                    print(
                        f"[indexer] voxel {ti+1}/{n_target_active} active "
                        f"({(ti+1)/n_target_active*100:.1f}%) "
                        f"rate={rate:.2f}/s "
                        f"elapsed={now-t_start:.0f}s eta={eta_s:.0f}s "
                        f"[v={v} seeds={voxel_seeds_ids.size} "
                        f"sols={vox_rec.shape[0]} dt={now-t_v0:.2f}s]",
                        flush=True,
                    )
        else:
            # Inter-voxel parallel path via spawn-based mp.Pool.
            # Why spawn (not fork): the parent already initialized numba's TBB
            # thread pool (creating 100+ helper threads). Fork inherits those
            # zombie threads and numba kernels in children deadlock on the
            # broken TBB state. Spawn boots each worker from a clean
            # interpreter — ctx is serialized via initargs once per worker.
            import multiprocessing as _mp
            intra_threads = max(1, int(
                os.environ.get(
                    "MIDAS_INDEX_INTRA_VOXEL_THREADS",
                    str(max(1, int(num_procs) // n_workers)),
                )
            ))
            print(
                f"[indexer] inter-voxel parallel: workers={n_workers} "
                f"intra_threads={intra_threads} tasks={n_target_active} "
                f"(spawn ctx)",
                flush=True,
            )
            mp_ctx = _mp.get_context("spawn")
            # Pickle ctx + voxel_xy_table into initargs. ctx contains the
            # observation tensor (~tens of MB) so this is a one-time cost
            # per worker boot, not per task.
            init_args = (
                seed_group_size, intra_threads, ctx, voxel_xy_table,
            )
            done = 0
            with mp_ctx.Pool(
                processes=n_workers,
                initializer=_voxel_worker_init_spawn,
                initargs=init_args,
            ) as pool:
                for v, vx_done, vy_done, vox_rec, vox_keys, vox_ids, n_seeds_done, dt_done in pool.imap_unordered(
                    _voxel_worker_task, tasks, chunksize=1,
                ):
                    per_voxel_records[v] = vox_rec
                    per_voxel_keys[v] = vox_keys
                    per_voxel_ids[v] = vox_ids
                    done += 1
                    if progress_on and (done % report_every == 0 or done == n_target_active):
                        now = _time.monotonic()
                        rate = done / max(now - t_start, 1e-9)
                        eta_s = (n_target_active - done) / max(rate, 1e-9)
                        print(
                            f"[indexer] voxel {done}/{n_target_active} active "
                            f"({done/n_target_active*100:.1f}%) "
                            f"rate={rate:.2f}/s "
                            f"elapsed={now-t_start:.0f}s eta={eta_s:.0f}s "
                            f"[v={v} seeds={n_seeds_done} "
                            f"sols={vox_rec.shape[0]} dt={dt_done:.2f}s]",
                            flush=True,
                        )

        # 7. Write all three consolidated files: IndexBest_all.bin (vals)
        # + IndexKey_all.bin (keys) + IndexBest_IDs_all.bin (IDs).
        # find_grains downstream needs all three siblings.
        from .io.consolidated import (
            write_index_key_all, write_index_best_ids_all,
        )
        write_index_best_all(out_path, per_voxel_records)
        out_path_p = Path(out_path)
        keys_path = out_path_p.with_name("IndexKey_all.bin")
        ids_path = out_path_p.with_name("IndexBest_IDs_all.bin")
        write_index_key_all(keys_path, per_voxel_keys)
        write_index_best_ids_all(ids_path, per_voxel_ids)
        return v_end - v_start

    # ------------------------------------------------------------------
    # C backend dispatchers (Phase 7).
    # ------------------------------------------------------------------

    def _resolve_paramstest_path(
        self, paramstest_path: str | os.PathLike | None,
    ) -> Path:
        if paramstest_path is not None:
            return Path(paramstest_path).resolve()
        if self._param_path is not None:
            return self._param_path
        raise ValueError(
            "backend='c-omp' requires paramstest_path (or construct the "
            "Indexer via Indexer.from_param_file(...) which captures it "
            "automatically)."
        )

    def _run_c_omp(
        self,
        *,
        block_nr: int,
        n_blocks: int,
        n_spots_to_index: int | None,
        num_procs: int,
        paramstest_path: str | os.PathLike | None,
    ) -> "IndexerResult":
        """C-backend FF dispatch. Returns a minimal IndexerResult — downstream
        stages read IndexBest_all.bin from OutputFolder directly."""
        from . import backend_c
        from .result import IndexerResult

        pp = self._resolve_paramstest_path(paramstest_path)
        # nWork (FF) = nSpotsToIndex. Default to the loaded spot count when
        # not supplied; SpotsToIndex.csv on disk is the source of truth for
        # the binary in any case.
        if n_spots_to_index is None:
            if self._observations is not None and "spot_ids" in self._observations:
                n_work = int(len(self._observations["spot_ids"]))
            else:
                # Best-effort fallback: count lines of SpotsToIndex.csv.
                ids_csv = pp.parent / "SpotsToIndex.csv"
                n_work = sum(1 for _ in ids_csv.open()) if ids_csv.exists() else 1
        else:
            n_work = int(n_spots_to_index)

        proc = backend_c.run_indexer(
            pp, block_nr=block_nr, n_blocks=n_blocks,
            n_work=n_work, num_procs=num_procs,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"midas_indexer (c-omp) exited {proc.returncode}.\n"
                f"stderr:\n{proc.stderr.decode('utf-8', errors='replace')[-2000:]}"
            )
        # Caller reads IndexBest_all.bin from OutputFolder/. We return an
        # empty in-memory result so the API shape stays uniform.
        return IndexerResult(block_nr=block_nr, n_blocks=n_blocks, seeds=[])

    def _run_scanning_c_omp(
        self,
        *,
        scan_positions: np.ndarray | torch.Tensor,
        num_procs: int,
        voxel_block_nr: int,
        voxel_n_blocks: int,
        paramstest_path: str | os.PathLike | None,
    ) -> int:
        """C-backend PF dispatch. Verifies positions.csv matches the supplied
        scan_positions (sanity), then shells out to the binary."""
        from . import backend_c

        pp = self._resolve_paramstest_path(paramstest_path)

        # Sanity: positions.csv should match the supplied scan_positions
        # length. We don't enforce value equality (sort order, formatting),
        # only count.
        scan_positions_np = np.asarray(scan_positions).reshape(-1)
        n_scans = int(scan_positions_np.size)
        positions_csv = pp.parent / "positions.csv"
        if positions_csv.exists():
            n_lines = sum(
                1 for line in positions_csv.read_text().splitlines()
                if line.strip()
            )
            if n_lines != n_scans:
                raise ValueError(
                    f"positions.csv at {positions_csv} has {n_lines} rows "
                    f"but run_scanning was called with {n_scans} scan_positions. "
                    "Re-emit positions.csv (or pass matching positions)."
                )

        proc = backend_c.run_indexer(
            pp, block_nr=voxel_block_nr, n_blocks=voxel_n_blocks,
            n_work=n_scans, num_procs=num_procs,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"midas_indexer (c-omp) exited {proc.returncode}.\n"
                f"stderr:\n{proc.stderr.decode('utf-8', errors='replace')[-2000:]}"
            )
        # Return processed voxel count for sharded runs (consistent with the
        # python path's return value).
        n_vox_total = n_scans * n_scans
        block_size = (n_vox_total + voxel_n_blocks - 1) // voxel_n_blocks
        v_start = voxel_block_nr * block_size
        v_end = min(v_start + block_size, n_vox_total)
        return v_end - v_start


# ---------------------------------------------------------------------------
# Inter-voxel parallel worker.
# Module-level state set by the parent before forking the mp.Pool; children
# inherit via fork's COW. Pickled across the boundary is only the per-voxel
# task tuple, never `ctx` itself (huge tensors stay shared on Linux).
# ---------------------------------------------------------------------------
_WORKER_CTX = None
_WORKER_VOXEL_XY_TABLE = None
_WORKER_SEED_GROUP_SIZE: int | None = None


def _voxel_worker_init(seed_group_size: int | None) -> None:
    """Pool initializer for fork-based pools (legacy / unused).

    Children inherit _WORKER_CTX + _WORKER_VOXEL_XY_TABLE via fork. Numba's
    TBB pool inherited from a multi-threaded parent often deadlocks though,
    so we recommend ``_voxel_worker_init_spawn`` with spawn context instead.
    """
    global _WORKER_SEED_GROUP_SIZE
    _WORKER_SEED_GROUP_SIZE = seed_group_size
    try:
        import numba as _nb
        n = int(os.environ.get("NUMBA_NUM_THREADS", "1"))
        _nb.set_num_threads(max(1, n))
    except Exception:
        pass


def _voxel_worker_init_spawn(
    seed_group_size: int | None,
    intra_threads: int,
    ctx,
    voxel_xy_table,
) -> None:
    """Pool initializer for spawn-based pools.

    Sets numba thread count BEFORE the first kernel call so the child's
    fresh interpreter doesn't oversubscribe. ctx is pickled across the
    boundary once per worker boot — significant but amortized over the
    worker's lifetime (many voxels).
    """
    global _WORKER_CTX, _WORKER_VOXEL_XY_TABLE, _WORKER_SEED_GROUP_SIZE
    _WORKER_SEED_GROUP_SIZE = seed_group_size
    _WORKER_CTX = ctx
    _WORKER_VOXEL_XY_TABLE = voxel_xy_table
    os.environ["NUMBA_NUM_THREADS"] = str(max(1, intra_threads))
    try:
        import numba as _nb
        _nb.set_num_threads(max(1, intra_threads))
    except Exception:
        pass
    # Cap torch's internal thread pool similarly so its ops don't fight
    # numba inside one worker.
    try:
        torch.set_num_threads(max(1, intra_threads))
    except Exception:
        pass


def _voxel_worker_task(
    task: tuple[int, int, float, float, np.ndarray],
) -> tuple[int, float, float, np.ndarray, np.ndarray, np.ndarray, int, float]:
    import time as _time
    vi, v, vx, vy, voxel_seeds_ids = task
    ctx = _WORKER_CTX
    assert ctx is not None, "_WORKER_CTX not set; forking before parent ctx ready?"
    assert _WORKER_VOXEL_XY_TABLE is not None
    ctx.current_voxel_xy = _WORKER_VOXEL_XY_TABLE[v]
    from .pipeline import run_block as _rb
    t0 = _time.monotonic()
    voxel_seeds = torch.as_tensor(voxel_seeds_ids, dtype=torch.int64)
    voxel_result = _rb(
        ctx, voxel_seeds,
        block_nr=0, n_blocks=1,
        seed_group_size=_WORKER_SEED_GROUP_SIZE,
    )
    vox_rec, vox_keys, vox_ids = _seeds_to_record_block(
        voxel_result, voxel_xyz=(vx, vy, 0.0),
    )
    dt = _time.monotonic() - t0
    return v, vx, vy, vox_rec, vox_keys, vox_ids, int(voxel_seeds_ids.size), dt


def _seeds_to_record_block(
    result, *, voxel_xyz: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert an IndexerResult into the consolidated trio of arrays.

    Returns ``(vals, keys, ids)`` where:
      - ``vals`` : (n_sol, 16) float64 — the IndexBest_all.bin record block.
      - ``keys`` : (n_sol, 4)  uint64  — the IndexKey_all.bin keys per
                   solution: ``[SpotID, nMatches, nIDs, reserved=0]``.
      - ``ids``  : (n_ids_total,) int32 — concatenated matched obs IDs
                   across all solutions for IndexBest_IDs_all.bin.

    The ``vals`` column map (matches IndexerScanningOMP.c):
        col 0  : seed spot id
        col 1  : avg internal-angle score (radians)
        col 2-10: 9-element orientation matrix (row-major 3×3)
        col 11 : posX — voxel center x (µm) in scanning mode
        col 12 : posY — voxel center y (µm) in scanning mode
        col 13 : posZ — voxel center z (µm, =0) in scanning mode
        col 14 : nExpected (total predicted spots, denominator)
        col 15 : nMatches (matched predicted spots, numerator)

    When ``voxel_xyz`` is provided (scanning mode), cols 11-13 are written
    as the voxel center to match IndexerScanningOMP.c's convention. PF
    refinement (midas-fit-grain) consumes the voxel center as the
    refinement starting point. Empty seeds (no matches) are dropped.
    """
    rows: list[list[float]] = []
    key_rows: list[list[int]] = []
    id_chunks: list[np.ndarray] = []
    for s in result.seeds:
        if s.n_matches <= 0:
            continue
        om = s.best_or_mat.detach().cpu().numpy().reshape(9)
        if voxel_xyz is None:
            pos_xyz = s.best_pos.detach().cpu().numpy().reshape(3)
            px, py, pz = float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2])
        else:
            px, py, pz = voxel_xyz
        matched_ids_np = s.matched_ids.detach().cpu().numpy().astype(np.int32).ravel()
        n_ids = int(matched_ids_np.size)
        rows.append([
            float(s.spot_id),                          # 0
            float(s.avg_ia),                           # 1
            *[float(x) for x in om],                   # 2-10
            px, py, pz,                                 # 11-13
            float(s.n_t_spots),                        # 14: nExpected
            float(s.n_matches),                        # 15: nMatches
        ])
        key_rows.append([
            int(s.spot_id),                            # 0: SpotID
            int(s.n_matches),                          # 1: nMatches
            n_ids,                                     # 2: nIDs
            0,                                         # 3: reserved
        ])
        if n_ids > 0:
            id_chunks.append(matched_ids_np)
    if not rows:
        return (
            np.zeros((0, 16), dtype=np.float64),
            np.zeros((0, 4), dtype=np.uint64),
            np.zeros((0,), dtype=np.int32),
        )
    vals = np.asarray(rows, dtype=np.float64)
    keys = np.asarray(key_rows, dtype=np.uint64)
    ids = (
        np.concatenate(id_chunks).astype(np.int32, copy=False)
        if id_chunks else np.zeros((0,), dtype=np.int32)
    )
    return vals, keys, ids
