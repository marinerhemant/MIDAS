"""Top-level orchestrator: load → cluster → resolve → strain → stress → write.

Mirrors the structure of ``midas_index.indexer.Indexer`` and
``midas_transforms.pipeline.Pipeline``: a class with a ``from_param_file``
constructor and a ``run()`` method that returns a fully-populated
:class:`ProcessGrainsResult`.

The class is intentionally device-/dtype-aware throughout. Tensors stay on
the configured device until ``result.write()`` is called.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from midas_stress.orientation import (
    orient_mat_to_quat,
    misorientation_quat_batch,
    fundamental_zone,
)

from .compute.canonicalize import _quat_mul
from .compute.cluster import (
    ClusterResult,
    cluster_by_misorientation,
    cluster_by_misorientation_from_orient_mats,
)
from .compute.conflict import ResolvedClaim, resolve_conflicts
from .compute.geometry import lab_obs_to_g_and_d
from .compute.refine_cluster import GrainCandidate, refine_cluster_spot_aware
from .compute.strain import (
    PerSpotStrainResult,
    solve_strain_fable_beaudoin,
    solve_strain_kenesei_bounded,
    solve_strain_kenesei_prior_anchored,
    solve_strain_kenesei_unbounded,
    solve_strain_lattice,
)
from .compute.stress import cauchy_stress, resolve_stiffness
from .compute.symmetry import SymmetryTable, build_symmetry_table
from .device import resolve_device, resolve_dtype
from .io.binary import (
    BinaryInputs,
    ORIENT_POS_FIT_LAYOUT,
    read_all,
)
from .io.hkls import HklTable, load_hkl_table
from .io.ids_hash import IDsHash, load_ids_hash
from .modes import (
    VALID_MODES, apply_mode_defaults, misori_tol_rad, needs_adaptive_misori,
)
from .params import ProcessGrainsParams, read_paramstest_pg
from .result import ProcessGrainsResult


__all__ = ["ProcessGrains"]


_DEG2RAD = math.pi / 180.0


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


@dataclass
class ProcessGrains:
    """Run the ProcessGrains pipeline on a directory of pipeline outputs."""

    params: ProcessGrainsParams
    run_dir: Path
    device: torch.device
    dtype: torch.dtype

    # Lazily populated by ``run()``.
    binaries: Optional[BinaryInputs] = None
    hkl_table: Optional[HklTable] = None
    sym_table: Optional[SymmetryTable] = None
    ids_hash: Optional[IDsHash] = None
    # Per-spot grain radius (SpotID → µm) from Radius_*.csv; None if absent.
    spot_radius_by_id: Optional[np.ndarray] = None

    # ---- Constructors ------------------------------------------------------

    @classmethod
    def from_param_file(
        cls,
        param_file: Union[str, Path],
        *,
        run_dir: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ) -> "ProcessGrains":
        """File-driven constructor.

        ``run_dir`` defaults to the directory containing ``param_file``.
        """
        pf = Path(param_file)
        if run_dir is None:
            run_dir = pf.parent
        params = read_paramstest_pg(pf)
        dev = resolve_device(device)
        dt = resolve_dtype(dev, dtype)
        return cls(params=params, run_dir=Path(run_dir), device=dev, dtype=dt)

    # ---- Loading -----------------------------------------------------------

    def load(
        self,
        *,
        require_fit_best: bool = True,
        require_index_best_full: bool = True,
    ) -> None:
        """Read every input the pipeline depends on (mmap-backed)."""
        if self.binaries is None:
            self.binaries = read_all(
                self.run_dir,
                require_fit_best=require_fit_best,
                require_index_best_full=require_index_best_full,
            )
        if self.hkl_table is None:
            hkls_path = self.run_dir / "hkls.csv"
            if not hkls_path.exists():
                raise FileNotFoundError(
                    f"hkls.csv not found at {hkls_path}; cannot build the "
                    "symmetry-permutation table."
                )
            self.hkl_table = load_hkl_table(
                hkls_path, ring_numbers=self.params.RingNumbers,
            )
        if self.sym_table is None:
            self.sym_table = build_symmetry_table(
                self.params.SGNr,
                self.hkl_table,
                device=self.device,
                dtype=self.dtype,
                warn_missing=False,
            )
        if self.ids_hash is None:
            ih_path = self.run_dir / "IDsHash.csv"
            if ih_path.exists():
                self.ids_hash = load_ids_hash(ih_path)
            else:
                # c-omp pipeline doesn't emit IDsHash.csv (only the legacy C
                # FitSetup did); synthesize it from the per-spot InputAll table
                # so spot-aware per-spot strain has reference d-spacings.
                from .io.ids_hash import build_ids_hash_from_inputall
                self.ids_hash = build_ids_hash_from_inputall(
                    self.run_dir, float(self.params.Wavelength),
                    list(self.params.RingNumbers))

        # Per-spot grain radius (Radius_*.csv, col 0=SpotID, col 15=GrainRadius
        # from midas-calc-radius). The c-omp refiner hardcodes the OrientPosFit
        # meanRadius to 1, so the physical per-grain GrainRadius is recovered
        # here by averaging the per-spot radii over each grain's matched spots
        # — exactly what legacy FitPosOrStrainsOMP.c does (meanRadius += per-spot
        # radius; /= nSpotsRad). spot_radius_by_id[SpotID] → radius (µm), 0 when
        # absent. None when no Radius_*.csv exists (keeps the placeholder).
        if self.spot_radius_by_id is None:
            self.spot_radius_by_id = self._load_spot_radius_by_id()

    def _load_spot_radius_by_id(self) -> Optional[np.ndarray]:
        """Build a SpotID → grain-radius (µm) lookup from ``Radius_*.csv``.

        ``midas-calc-radius`` (midas_transforms) writes ``Radius_StartNr_*.csv``
        with col 0 = SpotID and col 15 = per-spot GrainRadius. Returns a dense
        array indexed by SpotID (0 where unknown), or ``None`` when no Radius
        file is present (callers then keep the refiner's meanRadius).
        """
        cands = sorted(self.run_dir.glob("Radius_StartNr_*.csv"))
        if not cands:
            return None
        try:
            arr = np.loadtxt(cands[0], skiprows=1, usecols=(0, 15))
        except Exception:
            return None
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.size == 0:
            return None
        sids = arr[:, 0].astype(np.int64)
        rad = arr[:, 1].astype(np.float64)
        max_sid = int(sids.max())
        out = np.zeros(max_sid + 1, dtype=np.float64)
        ok = (sids >= 0) & (sids <= max_sid)
        out[sids[ok]] = rad[ok]
        return out

    # ---- Run ---------------------------------------------------------------

    def run(self, mode: str = "spot_aware") -> ProcessGrainsResult:
        """End-to-end: cluster + resolve + strain + (optional) stress.

        Parameters
        ----------
        mode : str
            One of ``legacy``, ``paper_claim``, ``spot_aware``.

        Returns
        -------
        ProcessGrainsResult
        """
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}; got {mode!r}")
        params = apply_mode_defaults(self.params, mode)
        if self.binaries is None:
            # The per-spot residual table (FitBest.bin) is only needed by the
            # spot-aware conflict resolution; ``legacy`` (single grain per
            # cluster) dedups from OrientPosFit + ProcessKey alone. Skipping it
            # lets the c-omp FF refiner (which emits ProcessKey.bin, not the
            # consolidated FitBest.bin) run through legacy process-grains.
            self.load(require_fit_best=(mode != "legacy"))

        # ---- Stage 1: pull per-seed orientations / positions / lattice -----
        opf = np.asarray(self.binaries.orient_pos_fit)            # (N_seeds, 27)
        n_seeds = opf.shape[0]
        keys = np.asarray(self.binaries.key)                       # (N_seeds, 2)
        alive_mask = keys[:, 0] != 0
        # Optional subset for smoke / dev runs (CLI --max-seeds).
        max_seeds_raw = self.params.raw.get("__max_seeds__")
        if max_seeds_raw:
            cap = int(max_seeds_raw[0])
            alive_idx = np.flatnonzero(alive_mask)
            if alive_idx.size > cap:
                drop = alive_idx[cap:]
                alive_mask[drop] = False
                print(
                    f"midas-process-grains: capping at {cap} alive seeds "
                    f"(was {alive_idx.size})", flush=True,
                )

        # OrientMat row-major (cols 1..9), positions (cols 11..13),
        # lattice (cols 15..20), meanRadius (col 25), completeness (col 26),
        # internal_angle (col 24).
        orient_mats = opf[:, ORIENT_POS_FIT_LAYOUT["orient_mat"]].reshape(
            n_seeds, 3, 3
        )
        positions = opf[:, ORIENT_POS_FIT_LAYOUT["position"]]
        lattices = opf[:, ORIENT_POS_FIT_LAYOUT["lattice"]]
        radii = opf[:, ORIENT_POS_FIT_LAYOUT["mean_radius"]]
        confidences = opf[:, ORIENT_POS_FIT_LAYOUT["completeness"]]
        ias = opf[:, ORIENT_POS_FIT_LAYOUT["internal_ang"]]

        # ---- Stage 1b: Resolve MisoriTol for ``adaptive`` mode ----
        # In adaptive mode (without user override), derive the misori threshold
        # from the antimode of the pairwise-misorientation histogram. The
        # antimode separates "same-grain duplicate" pairs (near zero) from
        # "different grain" pairs (above ~degrees); placing the cutoff at the
        # antimode is the data-driven equivalent of the §3.6 paper's 0.01°
        # specification (which is itself near the antimode on typical Ni FF
        # data). See ``compute/adaptive.derive_misori_tol`` for details.
        if needs_adaptive_misori(self.params, mode):
            from .compute.adaptive import derive_misori_tol
            theta_star_deg, adaptive_diag = derive_misori_tol(
                orient_mats, self.params.SGNr, alive_mask=alive_mask,
            )
            print(
                f"[pg adaptive] derived misori threshold from antimode: "
                f"θ* = {theta_star_deg:.4f}°  "
                f"(raw = {adaptive_diag['raw_antimode_deg']:.4f}°, "
                f"n_pairs = {adaptive_diag['n_pairs']:,})",
                flush=True,
            )
            params.MisoriTol = theta_star_deg
            params = params.validated()

        # ---- Stage 2: Phase 1 cluster gather (vectorised + sym-extended) ---
        # Symmetry-extended bucket prefilter handles the FZ-boundary
        # discontinuity that broke the v0.1 path; vectorised OM→quat→FZ
        # uses midas-stress's torch ops in one bulk call.
        tol_rad = misori_tol_rad(params)
        cluster = cluster_by_misorientation_from_orient_mats(
            orient_mats,
            self.params.SGNr,
            misori_tol_rad=tol_rad,
            alive_mask=alive_mask,
        )

        # Compute quats once for downstream Phase 2 (sym-op alignment).
        # Reuse the same vectorised path as the cluster prefilter.
        alive_idx_local = np.flatnonzero(alive_mask)
        if alive_idx_local.size > 0:
            om_t = torch.from_numpy(
                np.ascontiguousarray(
                    orient_mats[alive_idx_local], dtype=np.float64,
                )
            )
            quats_alive_t = orient_mat_to_quat(om_t)
            quats_alive = quats_alive_t.detach().cpu().numpy()
        else:
            quats_alive = np.empty((0, 4), dtype=np.float64)
        quats = np.empty((n_seeds, 4), dtype=np.float64)
        quats[:, 0] = 1.0
        quats[alive_idx_local] = quats_alive

        # ---- Stage 3: per-cluster Phase 2 + Phase 3 ------------------------
        # IndexBestFull is huge (~28 GB). Only load the rows we need: the
        # alive-seed positions we'll actually visit in the cluster loop.
        # Eagerly reading the full ``[..., 0]`` strided view across the
        # whole mmap was costing ~10 min of NFS time even for tiny smoke
        # runs.
        ibf_full = self.binaries.index_best_full
        alive_pos = np.flatnonzero(alive_mask)
        ibf_alive_col0 = np.empty(
            (alive_pos.size, ibf_full.shape[1]), dtype=np.int64,
        )
        ibf_alive_col1 = np.empty(
            (alive_pos.size, ibf_full.shape[1]), dtype=np.float64,
        )
        # Sorted sequential reads → one big stripe per seed → much faster
        # than strided whole-file scans.
        for li, ap in enumerate(alive_pos):
            block = np.array(ibf_full[ap], copy=True)        # (5000, 2) in RAM
            ibf_alive_col0[li] = block[:, 0].astype(np.int64)
            ibf_alive_col1[li] = block[:, 1]
        # Map global seed index → row in the compact ``alive`` arrays.
        ibf_global_to_local = np.full(n_seeds, -1, dtype=np.int64)
        ibf_global_to_local[alive_pos] = np.arange(alive_pos.size, dtype=np.int64)

        # Lookup helper: global seed pos → ``(n_hkls,)`` int64 of matched SpotIDs.
        def _ibf_col0(pos: int) -> np.ndarray:
            local = ibf_global_to_local[pos]
            return ibf_alive_col0[local]

        def _ibf_col1(pos: int) -> np.ndarray:
            local = ibf_global_to_local[pos]
            return ibf_alive_col1[local]

        # Backwards-compat names used by the loop body below — view of the
        # compact buffers, indexed by ``ibf_global_to_local``.
        # We simulate ``ibf_col0[m_pos]`` calls via the helpers above.
        ibf_col0 = None  # sentinel — should not be indexed directly anymore
        ibf_col1 = None

        ring_radii_per_hkl = np.asarray(
            self.hkl_table.real[:, 6], dtype=np.float64,
        )

        out_grains: List[Dict] = []
        cluster_sizes_diag: List[int] = []
        n_resolved_hkls_diag: List[int] = []
        n_majority_diag: List[int] = []
        n_tie_diag: List[int] = []
        n_fs_diag: List[int] = []
        grain_ids_key: List[Tuple[int, int, List[Tuple[int, int]]]] = []

        # Pre-compute cluster_id → member-positions in one pass.
        # Replaces O(n_clusters × n_seeds) np.flatnonzero scan that was the
        # dominant cost when n_clusters ~ 20 k.
        members_by_label: Dict[int, np.ndarray] = {}
        sort_order = np.argsort(cluster.labels, kind="stable")
        sorted_labels = cluster.labels[sort_order]
        # Skip the leading -1 (non-alive) block.
        first_alive = int(np.searchsorted(sorted_labels, 0, side="left"))
        sorted_labels = sorted_labels[first_alive:]
        sort_order = sort_order[first_alive:]
        if sorted_labels.size > 0:
            label_changes = np.flatnonzero(np.diff(sorted_labels) != 0) + 1
            label_breaks = np.concatenate([[0], label_changes, [sorted_labels.size]])
            for k in range(label_breaks.size - 1):
                lo, hi = int(label_breaks[k]), int(label_breaks[k + 1])
                lab = int(sorted_labels[lo])
                members_by_label[lab] = sort_order[lo:hi]

        # ---- Stage 2b: Pass A spot-overlap merge ---------------------------
        # Replaces C ProcessGrains' position-based Pass A with a spot-overlap
        # merge. Two Phase-1 cluster reps merge if misori < PassAMisoriTol AND
        # Jaccard(rep_spotIDs) ≥ PassAJaccardTol. See compute/pass_a.py.
        if params.EnablePassA and cluster.n_clusters > 1:
            from .compute.pass_a import merge_clusters_by_spot_overlap
            new_labels, new_n, new_members = merge_clusters_by_spot_overlap(
                cluster_labels=cluster.labels,
                n_phase1_clusters=cluster.n_clusters,
                members_by_label=members_by_label,
                quats_per_seed=quats,
                ias_per_seed=ias,
                ibf_alive_col0=ibf_alive_col0,
                ibf_global_to_local=ibf_global_to_local,
                space_group=self.params.SGNr,
                misori_tol_rad=math.radians(params.PassAMisoriTol),
                jaccard_tol=params.PassAJaccardTol,
                progress=True,
            )
            # Replace cluster.labels and the members map in-place. The
            # ClusterResult is a frozen dataclass; mutate fields directly.
            cluster.labels = new_labels
            cluster.n_clusters = new_n
            members_by_label = new_members

        n_clusters = cluster.n_clusters
        progress_every = max(1, n_clusters // 50)
        import time as _time
        _t_loop_start = _time.time()

        for cluster_id in range(n_clusters):
            if cluster_id % progress_every == 0:
                elapsed = _time.time() - _t_loop_start
                rate = (cluster_id / elapsed) if elapsed > 0 else 0.0
                eta = (n_clusters - cluster_id) / rate if rate > 0 else 0.0
                print(
                    f"[pg] cluster {cluster_id}/{n_clusters}  "
                    f"elapsed={elapsed:.0f}s  rate={rate:.1f}/s  eta={eta:.0f}s",
                    flush=True,
                )
            members = members_by_label.get(cluster_id)
            if members is None or members.size == 0:
                continue
            # Min-IA rep within the cluster.
            rep_local = int(np.argmin(ias[members]))
            rep_pos = int(members[rep_local])

            # When Pass A is enabled it already performs the spot-overlap merge
            # at the cluster-rep level; running Phase 2 spot-aware sub-clustering
            # afterwards just re-splits each super-cluster back into per-seed
            # singletons (because IBF col-0 Jaccard between same-grain seeds is
            # only ~0.16, well below Phase 2's edge_weight_threshold=0.7).
            # Treat super-clusters as single grains in that case.
            single_grain_per_cluster = (mode == "legacy") or params.EnablePassA

            if single_grain_per_cluster:
                # Legacy: emit only rep's matched-spot list (matches C behaviour
                # before adding spot-aware merge).
                aligned_col0 = _ibf_col0(rep_pos)[None, :]      # (1, n_hkls)
                aligned_col1 = _ibf_col1(rep_pos)[None, :]
                resolved = resolve_conflicts(
                    aligned_col0, aligned_col1,
                    policy="vote_then_residual",
                )
                cluster_size = int(members.size)
                others = []
                for m in members:
                    if int(m) != rep_pos:
                        # Other (id, pos) pairs — id reuses the matched SpotID
                        # convention (which the C code uses as a label).
                        others.append((int(m + 1), int(m)))
                grain_ids_key.append((rep_pos + 1, rep_pos, others))
                _maj = sum(1 for r in resolved if r.policy_used == "majority")
                _tie = sum(1 for r in resolved if r.policy_used == "residual_tie")
                _fs = sum(1 for r in resolved if r.policy_used == "forward_sim")

                out_grains.append({
                    "rep_pos": rep_pos,
                    "members": members.tolist(),
                    "resolved": resolved,
                    "orient_mat": orient_mats[rep_pos],
                    "position": positions[rep_pos],
                    "lattice": lattices[rep_pos],
                    "grain_radius": radii[rep_pos],
                    "confidence": confidences[rep_pos],
                })
                cluster_sizes_diag.append(cluster_size)
                n_resolved_hkls_diag.append(len(resolved))
                n_majority_diag.append(_maj)
                n_tie_diag.append(_tie)
                n_fs_diag.append(_fs)
                continue

            # spot_aware / paper_claim: full Phase 2 + Phase 3
            member_quats = quats[members]
            rep_quat = quats[rep_pos]
            # Vectorised member lookup (was Python list comprehension).
            member_local = ibf_global_to_local[members]
            member_col0 = ibf_alive_col0[member_local]

            sub_grains = refine_cluster_spot_aware(
                member_positions=members.tolist(),
                rep_pos=rep_pos,
                member_quats=member_quats,
                rep_quat=rep_quat,
                member_index_best_full_col0=member_col0,
                sym_table=self.sym_table,
                ring_radii_per_hkl=ring_radii_per_hkl,
                pixel_size_um=self.params.px,
                pixel_tol=params.PixelTol,
                jaccard_tol=params.JaccardTol,
                agreement_tol=params.AgreementTol,
                merge_alpha=params.MergeAlpha,
                edge_weight_threshold=max(
                    params.JaccardTol, params.AgreementTol,
                ),
                min_nr_spots=params.MinNrSpots,
            )
            for sub in sub_grains:
                # Build aligned (col0, col1) tables using the same op indices.
                aligned_c0 = []
                aligned_c1 = []
                for m_pos, s_idx in zip(sub.member_positions, sub.member_sym_ops):
                    perm = self.sym_table.hkl_perm[s_idx].cpu().numpy()
                    valid = perm >= 0
                    safe = np.where(valid, perm, 0)
                    a0 = _ibf_col0(int(m_pos))[safe]
                    a1 = _ibf_col1(int(m_pos))[safe]
                    a0[~valid] = 0
                    a1[~valid] = 0.0
                    aligned_c0.append(a0)
                    aligned_c1.append(a1)
                aligned_c0_arr = np.stack(aligned_c0, axis=0)
                aligned_c1_arr = np.stack(aligned_c1, axis=0)
                resolved = resolve_conflicts(
                    aligned_c0_arr, aligned_c1_arr,
                    policy=params.ConflictPolicy,
                )
                others = [
                    (int(m + 1), int(m))
                    for m in sub.member_positions
                    if int(m) != sub.rep_pos
                ]
                grain_ids_key.append((sub.rep_pos + 1, sub.rep_pos, others))

                _maj = sum(1 for r in resolved if r.policy_used == "majority")
                _tie = sum(1 for r in resolved if r.policy_used == "residual_tie")
                _fs = sum(1 for r in resolved if r.policy_used == "forward_sim")
                out_grains.append({
                    "rep_pos": sub.rep_pos,
                    "members": sub.member_positions,
                    "resolved": resolved,
                    "orient_mat": orient_mats[sub.rep_pos],
                    "position": positions[sub.rep_pos],
                    "lattice": lattices[sub.rep_pos],
                    "grain_radius": radii[sub.rep_pos],
                    "confidence": confidences[sub.rep_pos],
                })
                cluster_sizes_diag.append(len(sub.member_positions))
                n_resolved_hkls_diag.append(len(resolved))
                n_majority_diag.append(_maj)
                n_tie_diag.append(_tie)
                n_fs_diag.append(_fs)

        if not out_grains:
            return _empty_result(self, params, mode)

        # ---- Stage 4: per-grain strain via configured method ---------------
        n_g = len(out_grains)
        strain_lab = torch.zeros((n_g, 3, 3), dtype=self.dtype, device=self.device)
        strain_grain = torch.zeros_like(strain_lab)

        # FitBest column 0 = SpotID, columns 1..6 are observed (y, z, ome, ?, ?, ?)
        # Column layout per FitPosOrStrainsOMP.c:689-702:
        #   0=SpotID, 1..3 obs y/z/ome, 4..6 theor, 7..12 same in detector px,
        #   13..15 raw obs y/z/ome, 16..18 raw obs eta etc., 19 minIA,
        #   20 diffLen, 21 diffOme.
        fb = self.binaries.fit_best
        for gi, g in enumerate(out_grains):
            rep = g["rep_pos"]
            if fb is None:
                # Without FitBest we can only use the lattice-params method.
                lat_strained = torch.from_numpy(g["lattice"]).to(self.device, self.dtype)
                lat_ref = torch.tensor(
                    self.params.LatticeConstant, device=self.device, dtype=self.dtype,
                )
                eps_gr = solve_strain_lattice(lat_strained, lat_ref)
                eps_gr = torch.as_tensor(eps_gr, device=self.device, dtype=self.dtype)
                strain_grain[gi] = eps_gr
                # Lab frame = U eps_gr U^T
                U = torch.from_numpy(g["orient_mat"]).to(self.device, self.dtype)
                strain_lab[gi] = U @ eps_gr @ U.T
                continue

            # Build per-spot (g_obs, ds_obs, ds_0) from FitBest rows for this grain.
            # We use the resolved hkls (Phase 3 outputs).
            resolved_rows = g["resolved"]
            if not resolved_rows:
                continue
            n_spots = len(resolved_rows)
            wavelength_a = float(self.params.Wavelength)
            lsd_um = float(self.params.Lsd)

            # Vectorise the FitBest lookups: build a SpotID → row index map
            # for this seed once, then bulk-fetch (y, z) and reference d.
            #
            # Performance NOTE: ``fb[rep]`` on the (n_seeds, 5000, 22) memmap
            # returns another mmap view, NOT a RAM copy. Subsequent point
            # accesses ``fb_seed[k, 1]`` cause one page-fault per random row
            # across a 314 GB file, making per-grain access take ~2 s on
            # NFS-backed scratch. ``.copy()`` forces a contiguous 880 KB
            # sequential read into RAM, dropping per-grain time to ~10 ms.
            # FitBest.bin can be one seed short of OrientPosFit.bin's row
            # count when the C pwrite path leaves the last slot
            # uninitialized (observed on the peakfit hard dataset). Skip
            # any rep beyond FitBest's actual end.
            if rep >= fb.shape[0]:
                continue
            fb_seed = np.array(fb[rep], copy=True)        # (5000, 22), in RAM
            sid_col = fb_seed[:, 0].astype(np.int64)
            sid_to_row: dict = {int(s): k for k, s in enumerate(sid_col) if s != 0}

            # Physical grain radius: mean of the per-spot radii over this
            # grain's matched spots (legacy FitPosOrStrainsOMP meanRadius). The
            # c-omp refiner wrote a meanRadius=1 placeholder into OrientPosFit;
            # override it here from Radius_*.csv when available.
            if self.spot_radius_by_id is not None:
                sids = sid_col[sid_col > 0]
                if sids.size:
                    in_range = sids < self.spot_radius_by_id.shape[0]
                    rad_vals = self.spot_radius_by_id[sids[in_range]]
                    rad_vals = rad_vals[rad_vals > 0]
                    if rad_vals.size:
                        g["grain_radius"] = float(rad_vals.mean())

            y_list = []
            z_list = []
            d0_list = []
            gsmp_list = []
            for r in resolved_rows:
                k = sid_to_row.get(int(r.spot_id))
                if k is None:
                    continue
                # Reference d-spacing comes from IDsHash (SpotID -> ring).
                # The C strain solver uses the same path; the hkl_row index
                # in IndexBestFull is a theoretical-spot index (per-seed,
                # ~240 entries for FCC at 7 rings), not a canonical hkl row,
                # so it cannot be used to index the filtered hkl_table.
                if self.ids_hash is not None:
                    d_ref = self.ids_hash.d_for_spot_id(int(r.spot_id))
                else:
                    # Fall back to hkl_row indexing for synthetic / smoke
                    # data without IDsHash.
                    d_ref = float(self.hkl_table.real[r.hkl_row, 4])
                if d_ref <= 0:
                    continue
                y_list.append(float(fb_seed[k, 1]))      # observed y_lab (µm)
                z_list.append(float(fb_seed[k, 2]))      # observed z_lab (µm)
                # Sample-frame unit g-vector (FitBest cols 4,5,6), already
                # ω-rotated by the C refiner. The Kenesei strain tensor lives
                # in the sample frame, so the design matrix must use this, NOT
                # a lab-frame ĝ recomputed from (y, z) — which would mix spots
                # taken at different ω into one tensor fit and dump the per-ω
                # variation into spurious shear (E12/E13). Mirrors
                # FitUnified.c::StrainTensorKenesei (gobs = SpotsInfo[i][4..6]).
                gsmp_list.append(fb_seed[k, 4:7].astype(np.float64))
                d0_list.append(d_ref)

            if not y_list:
                continue

            y_arr = np.asarray(y_list, dtype=np.float64)
            z_arr = np.asarray(z_list, dtype=np.float64)
            ds_0_arr = np.asarray(d0_list, dtype=np.float64)
            gsmp_arr = np.asarray(gsmp_list, dtype=np.float64)   # (n, 3)

            # d_obs comes from the lab (y, z) radius (2θ) exactly as the C
            # refiner does; the lab-frame ĝ it returns is discarded in favour
            # of the sample-frame ĝ from FitBest (see comment above).
            _g_lab_t, ds_t = lab_obs_to_g_and_d(
                torch.from_numpy(y_arr),
                torch.from_numpy(z_arr),
                lsd=lsd_um,
                wavelength_a=wavelength_a,
            )
            ds_obs = ds_t.numpy()
            ds_0 = ds_0_arr

            # Use the sample-frame ĝ from FitBest. Fall back to the lab-frame
            # ĝ per-spot only when FitBest did not populate cols 4,5,6 (e.g.
            # synthetic/smoke data without a c-omp refiner pass).
            g_smp_norm = np.linalg.norm(gsmp_arr, axis=1)
            g_obs = gsmp_arr.copy()
            unpop = g_smp_norm <= 0
            if unpop.any():
                g_obs[unpop] = _g_lab_t.numpy()[unpop]

            mask = (np.linalg.norm(g_obs, axis=1) > 0) & (ds_0 > 0)
            n_good = int(mask.sum())
            U = torch.from_numpy(np.ascontiguousarray(g["orient_mat"])).to(
                self.device, self.dtype,
            )

            wants_kenesei = params.StrainMethod in {
                "kenesei", "kenesei_unbounded", "both",
            }
            wants_fable = params.StrainMethod in {"fable_beaudoin", "both"}

            kenesei_lab: Optional[torch.Tensor] = None
            fable_gr: Optional[torch.Tensor] = None

            # Always compute Fable-Beaudoin first when we'll need it as the
            # Kenesei prior (or as a primary or diagnostic output).
            need_fable = (
                wants_fable
                or params.StrainMethod == "kenesei"
            )
            if need_fable:
                lat_strained = torch.from_numpy(np.ascontiguousarray(g["lattice"])).to(
                    self.device, self.dtype,
                )
                lat_ref = torch.tensor(
                    self.params.LatticeConstant,
                    device=self.device, dtype=self.dtype,
                )
                fable_gr = solve_strain_fable_beaudoin(lat_strained, lat_ref)
                fable_gr = torch.as_tensor(
                    fable_gr, device=self.device, dtype=self.dtype,
                )

            if wants_kenesei and n_good >= 6:
                g_t = torch.from_numpy(g_obs[mask]).to(self.device, self.dtype)
                ds_obs_t = torch.from_numpy(ds_obs[mask]).to(self.device, self.dtype)
                ds_0_t = torch.from_numpy(ds_0[mask]).to(self.device, self.dtype)

                if params.StrainMethod == "kenesei_unbounded":
                    res = solve_strain_kenesei_unbounded(
                        g_t, ds_obs_t, ds_0_t, regularization=1e-3,
                    )
                    kenesei_lab = res.epsilon_tensor
                else:
                    # Default "kenesei" path: prior-anchored at Fable-Beaudoin
                    # in lab frame. Mirrors the C Nelder-Mead initialised at
                    # Fable-Beaudoin → tiny moves on poorly-constrained
                    # components, full data fit on well-constrained ones.
                    fable_lab = U @ fable_gr @ U.T
                    fable_voigt = torch.tensor([
                        fable_lab[0, 0], fable_lab[1, 1], fable_lab[2, 2],
                        fable_lab[0, 1], fable_lab[0, 2], fable_lab[1, 2],
                    ], device=self.device, dtype=self.dtype)
                    res = solve_strain_kenesei_prior_anchored(
                        g_t, ds_obs_t, ds_0_t,
                        eps_prior_voigt=fable_voigt,
                        anchor_strength=0.1,
                    )
                    kenesei_lab = res.epsilon_tensor

            # Choose which strain populates Grains.csv (the "primary").
            # Order of preference: explicit method > Kenesei > Fable-Beaudoin.
            if kenesei_lab is not None:
                strain_lab[gi] = kenesei_lab
                strain_grain[gi] = U.T @ kenesei_lab @ U
            elif fable_gr is not None:
                strain_grain[gi] = fable_gr
                strain_lab[gi] = U @ fable_gr @ U.T
            else:
                # No method produced a result (n_good < 6, no fable, etc.).
                # Fall through with zeros.
                pass

            # When "both" is requested, also stash the Fable-Beaudoin pair
            # for diagnostic emission alongside the primary.
            if params.StrainMethod == "both" and fable_gr is not None:
                g["strain_fable_beaudoin_grain"] = fable_gr
                g["strain_fable_beaudoin_lab"] = U @ fable_gr @ U.T

        # ---- Stage 5: optional stress ---------------------------------------
        stiffness = resolve_stiffness(
            material_name=params.MaterialName,
            stiffness_file=params.StiffnessFile,
            device=self.device,
            dtype=self.dtype,
        )
        stress_lab = None
        stress_grain = None
        if stiffness is not None:
            stress_lab = torch.zeros_like(strain_lab)
            stress_grain = torch.zeros_like(strain_grain)
            for gi, g in enumerate(out_grains):
                U = torch.from_numpy(g["orient_mat"]).to(self.device, self.dtype)
                stress_lab[gi] = cauchy_stress(
                    strain_lab[gi], stiffness, orient=U, frame="lab",
                )
                stress_grain[gi] = cauchy_stress(
                    strain_grain[gi], stiffness, orient=U, frame="grain",
                )

        # ---- Stage 6: pack the result --------------------------------------
        ids = torch.tensor([g["rep_pos"] + 1 for g in out_grains], dtype=torch.int64)
        rep_pos = torch.tensor([g["rep_pos"] for g in out_grains], dtype=torch.int64)
        om = torch.tensor(
            np.stack([g["orient_mat"] for g in out_grains], axis=0),
            dtype=self.dtype, device=self.device,
        )
        pos = torch.tensor(
            np.stack([g["position"] for g in out_grains], axis=0),
            dtype=self.dtype, device=self.device,
        )
        lat = torch.tensor(
            np.stack([g["lattice"] for g in out_grains], axis=0),
            dtype=self.dtype, device=self.device,
        )
        rad = torch.tensor(
            np.array([g["grain_radius"] for g in out_grains]),
            dtype=self.dtype, device=self.device,
        )
        conf = torch.tensor(
            np.array([g["confidence"] for g in out_grains]),
            dtype=self.dtype, device=self.device,
        )

        # Per-grain refiner residuals (legacy Grains.csv DiffPos/Ome/Angle).
        # ErrorsFin lives at OrientPosFit cols 22-24 (pos_err_µm, ome_err_deg,
        # internal_angle_deg). The c-omp refiner FitUnified.c writes the same
        # layout as legacy FitPosOrStrainsOMP.
        rep_idx_np = np.array([g["rep_pos"] for g in out_grains], dtype=np.int64)
        if opf.ndim == 2 and opf.shape[1] >= 25:
            diff_pos = torch.as_tensor(opf[rep_idx_np, 22], dtype=self.dtype, device=self.device)
            diff_ome = torch.as_tensor(opf[rep_idx_np, 23], dtype=self.dtype, device=self.device)
            diff_ang = torch.as_tensor(opf[rep_idx_np, 24], dtype=self.dtype, device=self.device)
        else:
            diff_pos = torch.zeros(len(out_grains), dtype=self.dtype, device=self.device)
            diff_ome = diff_pos.clone(); diff_ang = diff_pos.clone()
        # Per-grain strain L2 residual (RMSErrorStrain, µε).  We don't yet
        # capture the per-grain solver residual in the strain loop; emit zeros
        # for now (legacy schema requires the column to be present).
        rms_strain = torch.zeros(len(out_grains), dtype=self.dtype, device=self.device)
        # Phase index — single-phase for now.
        phase_nr = torch.ones(len(out_grains), dtype=torch.int32, device=self.device)

        # Build SpotMatrix rows.
        sm_rows = _build_spot_matrix_rows(
            out_grains,
            fb=self.binaries.fit_best,
            hkl_table=self.hkl_table,
            ids_hash=self.ids_hash,
        )

        # ---- Optional: joint-NNLS grain-volume correction ------------------
        # Replaces R_grain = mean(R_per_spot) with sparse non-negative least
        # squares that attributes shared-spot intensity correctly between
        # twin partners and overlapping grains. Off by default for byte-level
        # compatibility with the C reference; opt in via
        # ``ProcessGrainsParams.NnlsVolume = True`` or ``--nnls-volume``.
        if bool(getattr(self.params, "NnlsVolume", False)) and len(out_grains) > 0:
            from .compute.volume_nnls import compute_nnls_volumes
            # SpotMatrix layout: col 0 = GrainID, col 1 = SpotID, col 7 = RingNr
            sm_g = sm_rows[:, 0].astype(np.int64)
            sm_s = sm_rows[:, 1].astype(np.int64)
            sm_r = sm_rows[:, 7].astype(np.int64) if sm_rows.shape[1] > 7 else (
                np.zeros(sm_rows.shape[0], dtype=np.int64))
            # Spot intensity + ring lookup from InputAllExtraInfo
            # (loaded once per pipeline run; available via self.binaries.input_all_extra_info or similar)
            try:
                ia = self.binaries.input_all_extra
                spot_intensity = {int(r[4]): float(r[14]) for r in ia}
                spot_ring = {int(r[4]): int(r[5]) for r in ia}
            except (AttributeError, KeyError):
                # Fallback: re-read from file
                ia_path = self.run_dir / "InputAllExtraInfoFittingAll.csv"
                if ia_path.exists():
                    import pandas as _pd
                    ia_df = _pd.read_csv(
                        ia_path, sep=r"\s+", comment="%", header=None,
                        usecols=[4, 5, 14],
                        names=["SpotID", "RingNr", "Intensity"], low_memory=False,
                    )
                    spot_intensity = dict(zip(ia_df["SpotID"].astype(int), ia_df["Intensity"]))
                    spot_ring = dict(zip(ia_df["SpotID"].astype(int), ia_df["RingNr"].astype(int)))
                else:
                    print("[pg nnls] InputAllExtraInfo.csv not found; skipping NNLS volume",
                          flush=True)
                    spot_intensity, spot_ring = {}, {}
            if spot_intensity:
                R_naive_np = rad.detach().cpu().numpy().astype(np.float64)
                gid_np = ids.detach().cpu().numpy().astype(np.int64)
                # Item 9: physical K(ring) when --physical-K is set.
                ring_K_arg = None
                if bool(getattr(self.params, "PhysicalK", False)):
                    from .compute.volume_nnls import physical_ring_K
                    try:
                        hkls_df = self.hkl_table.dataframe if hasattr(self.hkl_table, "dataframe") else None
                        if hkls_df is None:
                            import pandas as _pd
                            hkls_df = _pd.read_csv(self.run_dir / "hkls.csv", sep=r"\s+")
                        ring_K_arg = physical_ring_K(
                            hkls_df,
                            wavelength=float(self.params.Wavelength),
                            species=getattr(self.params, "MaterialName", "Ni") or "Ni",
                            B_factor=0.4,
                        )
                        print(f"[pg nnls] using PHYSICAL K(ring) from |F|²·LP·DWF·mult: "
                              f"{ {r: round(v, 3) for r, v in sorted(ring_K_arg.items())} }",
                              flush=True)
                    except Exception as e:
                        print(f"[pg nnls] physical K lookup failed ({e}); "
                              f"falling back to empirical median K", flush=True)
                        ring_K_arg = None
                nnls = compute_nnls_volumes(
                    grain_ids=gid_np, R_naive=R_naive_np,
                    sm_grain_id=sm_g, sm_spot_id=sm_s, sm_ring_nr=sm_r,
                    spot_intensity=spot_intensity, spot_ring=spot_ring,
                    ring_K=ring_K_arg,
                )
                print(
                    f"[pg nnls] {nnls.n_unique_spots:,} spots ("
                    f"{nnls.n_spots_shared:,} shared, "
                    f"{100*nnls.n_spots_shared/max(nnls.n_unique_spots,1):.1f}%) "
                    f"→ NNLS converged in {nnls.nnls_n_iter} iters, "
                    f"rescale = {nnls.rescale_factor:.3f}",
                    flush=True,
                )
                # Replace rad with NNLS-corrected radii
                rad = torch.tensor(nnls.R_nnls, dtype=self.dtype, device=self.device)

        diagnostics = {
            "cluster_sizes": np.asarray(cluster_sizes_diag, dtype=np.int32),
            "n_resolved_hkls": np.asarray(n_resolved_hkls_diag, dtype=np.int32),
            "n_majority_hkls": np.asarray(n_majority_diag, dtype=np.int32),
            "n_residual_tie_hkls": np.asarray(n_tie_diag, dtype=np.int32),
            "n_forward_sim_hkls": np.asarray(n_fs_diag, dtype=np.int32),
        }
        return ProcessGrainsResult(
            ids=ids,
            rep_pos=rep_pos,
            orient_mat=om,
            positions=pos,
            lattice=lat,
            grain_radius=rad,
            confidence=conf,
            strain_lab=strain_lab,
            strain_grain=strain_grain,
            diff_pos_um=diff_pos,
            diff_ome_deg=diff_ome,
            diff_angle_deg=diff_ang,
            rms_error_strain=rms_strain,
            phase_nr=phase_nr,
            stress_lab=stress_lab,
            stress_grain=stress_grain,
            spot_matrix_rows=torch.from_numpy(sm_rows),
            grain_ids_key=grain_ids_key,
            diagnostics=diagnostics,
            sg_nr=self.params.SGNr,
            lattice_reference=self.params.LatticeConstant,
            mode=mode,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_result(pg: "ProcessGrains", params, mode: str) -> ProcessGrainsResult:
    """Build a zero-grain result; used when the run yields nothing."""
    z3 = torch.zeros((0, 3, 3), dtype=pg.dtype, device=pg.device)
    z6 = torch.zeros((0, 6), dtype=pg.dtype, device=pg.device)
    return ProcessGrainsResult(
        ids=torch.zeros(0, dtype=torch.int64),
        rep_pos=torch.zeros(0, dtype=torch.int64),
        orient_mat=z3,
        positions=torch.zeros((0, 3), dtype=pg.dtype, device=pg.device),
        lattice=z6,
        grain_radius=torch.zeros(0, dtype=pg.dtype, device=pg.device),
        confidence=torch.zeros(0, dtype=pg.dtype, device=pg.device),
        strain_lab=z3,
        strain_grain=z3,
        spot_matrix_rows=torch.zeros((0, 12)),
        sg_nr=pg.params.SGNr,
        lattice_reference=pg.params.LatticeConstant,
        mode=mode,
    )


def _build_spot_matrix_rows(
    grains: List[Dict],
    fb,
    hkl_table: HklTable,
    ids_hash: Optional["IDsHash"] = None,
) -> np.ndarray:
    """Build the (n_rows, 12) SpotMatrix rows from per-grain resolved claims.

    Columns: GrainID, SpotID, Omega, DetectorHor, DetectorVert, OmeRaw, Eta,
    RingNr, YLab, ZLab, Theta, StrainError.

    Ring number lookup falls back through (in order):
      1. ``ids_hash.ring_for_spot_id(spot_id)``  — robust, uses SpotID range mapping.
         Required when the legacy / single-grain-per-cluster path runs because
         ``hkl_row`` there is the raw IBF-slot index (0..4999), not a hkls.csv row.
      2. ``hkl_table.integers[hkl_row, 3]``      — only valid for sym-aligned
         hkl_row (Phase 2 spot-aware path).
    """
    rows: List[List[float]] = []
    if fb is None:
        return np.zeros((0, 12))
    fb_arr = np.asarray(fb)
    fb_n_rows = fb_arr.shape[0]
    # Pre-extract the per-hkl ring + theta arrays for fast indexing.
    hkl_int_ring = hkl_table.integers[:, 3].astype(np.int64)
    hkl_theta_deg = hkl_table.real[:, 5] * (180.0 / math.pi)
    n_hkl_rows = hkl_int_ring.shape[0]
    n_grains = len(grains)
    progress_every = max(1, n_grains // 50)
    import time as _time
    _t0 = _time.time()
    for gi, g in enumerate(grains):
        if gi % progress_every == 0:
            elapsed = _time.time() - _t0
            rate = (gi / elapsed) if elapsed > 0 else 0.0
            print(
                f"[pg-sm] grain {gi}/{n_grains}  elapsed={elapsed:.0f}s  "
                f"rate={rate:.1f}/s",
                flush=True,
            )
        rep = g["rep_pos"]
        if rep >= fb_n_rows:
            continue
        seed_rows = np.array(fb_arr[rep], copy=True)         # (5000, 22) in RAM
        # Vectorised SpotID -> row index lookup. Replaces a 5000-iter Python
        # loop per grain (was 836 M iterations across the full peakfit run).
        sid_col = seed_rows[:, 0].astype(np.int64)
        nz = np.flatnonzero(sid_col)
        seed_lookup: Dict[int, int] = dict(
            zip(sid_col[nz].tolist(), nz.tolist())
        )
        for r in g["resolved"]:
            jj = seed_lookup.get(r.spot_id)
            if jj is None:
                continue
            row = seed_rows[jj]
            # Look up ring/theta. The IBF-slot index r.hkl_row can exceed
            # n_hkl_rows (122 for 7 FCC rings) because IBF stores both omega
            # solutions per (h,k,l). Use IDsHash (SpotID → ring) when
            # available, fall back to hkl_table for sym-aligned indices.
            ring_nr = 0
            theta_deg = 0.0
            if ids_hash is not None:
                try:
                    ring_nr = int(ids_hash.ring_for_spot_id(int(r.spot_id)))
                except Exception:
                    pass
            if 0 <= r.hkl_row < n_hkl_rows:
                if ring_nr == 0:
                    ring_nr = int(hkl_int_ring[r.hkl_row])
                theta_deg = float(hkl_theta_deg[r.hkl_row])
            # Eta angle is the AZIMUTH on the detector ring, not a FitBest
            # column. Legacy ProcessGrains.c writes Eta from the InputMatrix-
            # derived eta column; the equivalent here is to recompute it from
            # the lab-frame (YLab, ZLab) the same way grain_observations.py
            # does (eta_deg = atan2(-YLab, ZLab)). Using FitBest col-18 here
            # (a Z-like FitBest diagnostic) silently fills the Eta column
            # with garbage µm-scale values.
            y_lab = float(row[1]); z_lab = float(row[2])
            eta_deg = math.degrees(math.atan2(-y_lab, z_lab))
            rows.append([
                float(g["rep_pos"] + 1),        # GrainID
                float(r.spot_id),               # SpotID
                float(row[3]),                  # Omega
                float(row[7]),                  # DetectorHor (theor pixel)
                float(row[8]),                  # DetectorVert (theor pixel)
                float(row[15]) if seed_rows.shape[1] > 15 else float(row[3]),
                # ^^^ OmeRaw (raw omega pre-correction)
                eta_deg,                        # Eta (deg, recomputed from YLab/ZLab)
                float(ring_nr),                 # RingNr
                y_lab,                          # YLab
                z_lab,                          # ZLab
                float(theta_deg),               # Theta (deg)
                float(row[20]) if seed_rows.shape[1] > 20 else 0.0,
                # ^^^ diffLen residual ≈ StrainError diagnostic
            ])
    if not rows:
        return np.zeros((0, 12))
    return np.asarray(rows, dtype=np.float64)
