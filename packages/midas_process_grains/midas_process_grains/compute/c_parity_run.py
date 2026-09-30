"""End-to-end C-parity driver.

Stage 1 + Pass A + confidence filter, returning the list of kept grains
with the per-grain fields C ProcessGrains writes to Grains.csv.

Strain (Fable + Kenesei) and writers live in callers; this module is the
clustering+merging core only, kept slim so we can validate it in isolation
against C's `GrainIDsKey.csv` and `Grains.csv`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from .c_parity import (
    OPF_OM, OPF_POS, OPF_LATTICE, OPF_DIFF_POS, OPF_DIFF_OME, OPF_IA,
    OPF_RADIUS, OPF_CONFIDENCE,
    Stage1Cluster,
    Stage1Result,
    build_kept_list,
    pass_a_position_dedup,
    stage1_find_internal_angles,
)


@dataclass
class CParityKeptGrain:
    """One row in the eventual Grains.csv. Strain fields are filled later."""
    grain_id: int                       # = SpotID at rep_pos = ids[rep_pos]
    rep_pos: int
    member_positions: np.ndarray
    member_ids: np.ndarray
    orient_mat: np.ndarray              # (3, 3)
    position: np.ndarray                # (3,) X, Y, Z
    lattice: np.ndarray                 # (6,) a, b, c, α, β, γ
    diff_pos: float
    diff_ome: float
    diff_angle: float                   # = IA
    grain_radius: float
    confidence: float


@dataclass
class CParityResult:
    stage1: Stage1Result
    is_dup: np.ndarray
    kept_indices: np.ndarray            # indices into stage1.grain_positions
    kept_grains: List[CParityKeptGrain]


def read_spots_to_index(run_dir: Path) -> np.ndarray:
    """Read SpotsToIndex.csv → 1-D int64 of SpotIDs in OPF row order.

    Skips negative entries (matching C's `if (IDs[nrIDs] < 0) continue;`
    at ProcessGrains.c:456).
    """
    p = Path(run_dir) / "SpotsToIndex.csv"
    ids: List[int] = []
    with open(p) as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            v = int(tok[0])
            if v < 0:
                continue
            ids.append(v)
    return np.asarray(ids, dtype=np.int64)


def run_c_parity_clustering(
    *,
    run_dir: Path,
    opf: np.ndarray,
    process_key: np.ndarray,
    key: np.ndarray,
    space_group: int,
    misori_tol_stage1_deg: float = 0.4,
    misori_tol_passa_deg: float = 0.1,
    pos_tol_passa_um: float = 5.0,
    confidence_min: float = 0.05,
    min_nr_spots: int = 1,
    device: str = "cpu",
) -> CParityResult:
    """Stage 1 + Pass A + kept-list. No strain, no IO.

    Inputs are taken directly so callers can substitute synthetic data in
    tests; the convenience wrapper ``run_c_parity_pipeline_from_disk``
    reads from the run directory.
    """
    # Safety net for a genuinely mismatched set of inputs. This used to fire
    # on EVERY run because FitBest.bin / ProcessKey.bin were read one seed
    # short of OrientPosFit.bin — the C writer pwrites only nSpotsComp
    # records at a full-slot stride, so the last seed leaves a partial slot
    # that the readers truncated away (see io.binary.TailPaddedBinary).
    #
    # The old rationale here — "the dropped trailing seed gets NrIDsPerID=0
    # anyway and contributes nothing" — was WRONG. Measured on the 56,125-seed
    # Ni FF layer: the dropped seed was 56,124 / SpotID 245283 with
    # NrIDsPerID=87, keep_flag set and completeness 0.777, i.e. an ordinary
    # live candidate silently deleted from the grain list. The readers now
    # zero-pad instead, so this branch should no longer trigger; if it does,
    # the inputs really are inconsistent and dropping LIVE seeds is worth
    # shouting about rather than mentioning.
    n_seeds = min(opf.shape[0], process_key.shape[0], key.shape[0])
    if not (n_seeds == opf.shape[0] == process_key.shape[0] == key.shape[0]):
        n_dropped_alive = int((key[n_seeds:, 0] != 0).sum()) if key.shape[0] > n_seeds else 0
        print(f"[c-parity] WARNING: input row counts disagree — truncating to "
              f"{n_seeds:,} (OPF={opf.shape[0]}, PK={process_key.shape[0]}, "
              f"Key={key.shape[0]})", flush=True)
        if n_dropped_alive:
            print(f"[c-parity] WARNING: {n_dropped_alive} of the discarded "
                  f"trailing seeds are ALIVE (keep_flag set) and will not "
                  f"appear in Grains.csv. This is silent data loss — check "
                  f"that the refiner wrote all of FitBest/ProcessKey/"
                  f"OrientPosFit/Key from the SAME invocation.", flush=True)
        opf = opf[:n_seeds]
        process_key = process_key[:n_seeds]
        key = key[:n_seeds]

    # IDs = OPF column 0 (SpotID per OPF row, written by FitPosOrStrains)
    ids = opf[:, 0].astype(np.int64)
    keep_flag = (key[:, 0] != 0)
    nr_ids_per_id = key[:, 1].astype(np.int64)

    print(f"[c-parity] inputs: n_seeds={n_seeds:,}  alive={int(keep_flag.sum()):,}",
          flush=True)
    t0 = time.time()

    # ── Stage 1 ────────────────────────────────────────────────────────────
    stage1 = stage1_find_internal_angles(
        opf=opf, ids=ids, keep_flag=keep_flag,
        nr_ids_per_id=nr_ids_per_id,
        process_key=process_key,
        space_group=space_group,
        misori_tol_rad=math.radians(misori_tol_stage1_deg),
        min_nr_spots=min_nr_spots,
        device=device,
    )

    # ── Pass A ─────────────────────────────────────────────────────────────
    is_dup = pass_a_position_dedup(
        grain_positions=stage1.grain_positions,
        opf=opf, space_group=space_group,
        misori_tol_rad=math.radians(misori_tol_passa_deg),
        pos_tol_um=pos_tol_passa_um,
        device=device,
    )

    kept_indices = build_kept_list(
        grain_positions=stage1.grain_positions,
        is_dup=is_dup, opf=opf, confidence_min=confidence_min,
    )
    print(f"[c-parity] kept after PassA + conf>{confidence_min}: "
          f"{kept_indices.size:,} of {stage1.grain_positions.size:,}",
          flush=True)

    # ── Build kept-grain records ───────────────────────────────────────────
    kept_grains: List[CParityKeptGrain] = []
    for idx in kept_indices:
        cluster = stage1.clusters[idx]
        rep = cluster.rep_pos
        kept_grains.append(CParityKeptGrain(
            grain_id=int(ids[rep]),
            rep_pos=rep,
            member_positions=cluster.member_positions,
            member_ids=cluster.member_ids,
            orient_mat=opf[rep, OPF_OM].reshape(3, 3),
            position=opf[rep, OPF_POS].copy(),
            lattice=opf[rep, OPF_LATTICE].copy(),
            diff_pos=float(opf[rep, OPF_DIFF_POS]),
            diff_ome=float(opf[rep, OPF_DIFF_OME]),
            diff_angle=float(opf[rep, OPF_IA]),
            grain_radius=float(opf[rep, OPF_RADIUS]),
            confidence=float(opf[rep, OPF_CONFIDENCE]),
        ))

    print(f"[c-parity] total time: {time.time()-t0:.1f}s", flush=True)
    return CParityResult(
        stage1=stage1, is_dup=is_dup,
        kept_indices=np.asarray(kept_indices, dtype=np.int64),
        kept_grains=kept_grains,
    )


def resolve_stage1_misori_tol(explicit: Optional[float], params) -> Tuple[float, str]:
    """Stage-1 misorientation tolerance (deg) and where it came from.

    An explicit argument wins; else the parameter file's ``MisoriTol``; else C
    ProcessGrains' own 0.4 deg. Same defect class as MinNrSpots: this used to be
    a hardcoded 0.4 on the c_parity path, so ``MisoriTol`` in the file and the
    CLI's ``--misori-tol`` were both silently ignored in the default mode.
    Measured on 20-ID garnet (HPcat_P2 att5): the log read ``misori_tol = 0.400``
    with ``MisoriTol 1.0`` in the file, and the grain list was the same 657 rows.
    ``raw`` distinguishes "the file set it" from the dataclass default.
    """
    if explicit is not None:
        return float(explicit), "explicit argument"
    if "MisoriTol" in getattr(params, "raw", {}) and params.MisoriTol is not None:
        return float(params.MisoriTol), "from the parameter file"
    return 0.4, "C default; not set in the parameter file"


#: C ProcessGrains' Pass A tolerances (ProcessGrains.c:836-874).
C_PASSA_MISORI_DEG = 0.1
C_PASSA_POS_UM = 5.0


def resolve_passa_tols(misori_explicit: Optional[float], pos_explicit: Optional[float],
                       params) -> Tuple[float, float, str]:
    """Pass A (orientation + position dedup) tolerances and where they came from.

    Explicit arguments win; else the parameter file's ``CParityPassAMisoriTol`` (deg)
    and ``CParityPassAPosTol`` (um); else C ProcessGrains' 0.1 deg and 5 um, so an
    unset file stays bit-identical to C. Opt-in because the defaults are what
    c_parity was validated against EBSD with. The 5 um position gate is far
    tighter than grain-centroid scatter on 20-ID data: bt_20id_jul26b nf_sampleF
    layer 6 kept 246 pairs < 0.1 deg apart (spot Jaccard median 0.61) at
    7-40 um separation. Deliberately NOT ``PassAMisoriTol``: that key means the
    spot-overlap merge (1.0 deg) in the other modes.
    """
    raw = getattr(params, "raw", {})
    src = []
    if misori_explicit is not None:
        m = float(misori_explicit); src.append("misori explicit")
    elif "CParityPassAMisoriTol" in raw:
        m = float(raw["CParityPassAMisoriTol"][0]); src.append("misori from the parameter file")
    else:
        m = C_PASSA_MISORI_DEG; src.append("misori C default")
    if pos_explicit is not None:
        d = float(pos_explicit); src.append("pos explicit")
    elif "CParityPassAPosTol" in raw:
        d = float(raw["CParityPassAPosTol"][0]); src.append("pos from the parameter file")
    else:
        d = C_PASSA_POS_UM; src.append("pos C default")
    return m, d, "; ".join(src)


def run_c_parity_pipeline_from_disk(
    *,
    run_dir: Path,
    out_dir: Path,
    misori_tol_stage1_deg: Optional[float] = None,
    misori_tol_passa_deg: Optional[float] = None,
    pos_tol_passa_um: Optional[float] = None,
    confidence_min: Optional[float] = None,
    min_nr_spots: Optional[int] = None,
    write_spot_matrix: bool = True,
    write_diagnostics: bool = True,
    device: str = "cpu",
    paramstest: Optional[Union[str, Path]] = None,
) -> CParityResult:
    """End-to-end C-parity replica.

    Reads paramstest, OPF, Key, ProcessKey, FitBest from ``run_dir``;
    runs Stage 1 + Pass A + confidence filter; writes Grains.csv,
    GrainIDsKey.csv, and (if FitBest available) SpotMatrix.csv to
    ``out_dir`` in C ProcessGrains format.

    ``write_diagnostics`` additionally emits
    ``out_dir/processgrains_diagnostics.h5`` with the signed per-spot
    residual decomposition (``/residuals``), the same schema the spot-aware
    and ``legacy`` modes write. It costs no extra FitBest I/O — the rows are
    already in RAM for the strain solve — but holds ~11 float64 per matched
    spot until the table is assembled; pass ``False`` on a memory-tight run.

    ``paramstest`` names the parameter file to read; it defaults to
    ``run_dir/"paramstest.txt"``. Pass the file the caller was actually
    given — hardcoding the name silently discards a differently-named
    parameter file (the same defect fixed in ``run_v4_pipeline``).

    ``confidence_min`` defaults to the parameter file's ``Completeness``
    (the key C ProcessGrains applies), falling back to 0.05 when the file
    does not set it.

    Returns the :class:`CParityResult` for callers that want to inspect
    the kept grains in memory.
    """
    from ..io.binary import (materialize, read_fit_best, read_key,
                             read_orient_pos_fit, read_process_key)
    from ..io.ids_hash import load_ids_hash
    from ..params import read_paramstest_pg
    from .c_parity_emit import (
        gather_per_grain_spot_data,
        write_grains_csv,
        write_grain_ids_key,
        write_spot_matrix_csv,
        load_input_extra_info_matrix,
    )

    rd = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ps_path = Path(paramstest) if paramstest else rd / "paramstest.txt"
    print(f"[c-parity] loading inputs from {rd} (params: {ps_path})", flush=True)
    t0 = time.time()
    params = read_paramstest_pg(ps_path)
    if min_nr_spots is None:
        # The user's cluster-size floor lives in the parameter file. This used
        # to be hardcoded to 1 by the CLI, so `MinNrSpots 3` — propagated into
        # the file by the pipeline precisely so it would be honoured — was
        # silently discarded. On the datasetA Ni layer that is 23138 grains
        # instead of ~6180; C ProcessGrains shows the same split, 23710 at
        # MinNrSpots=1 against 6147 at 3.
        # `raw` distinguishes "the file set it" from the dataclass default.
        if "MinNrSpots" in getattr(params, "raw", {}):
            min_nr_spots = int(params.MinNrSpots)
        else:
            min_nr_spots = 1                 # C ProcessGrains' own default
        print(f"[c-parity] min_nr_spots={min_nr_spots} "
              f"({'from ' + ps_path.name if 'MinNrSpots' in getattr(params, 'raw', {}) else 'C default; not set in ' + ps_path.name})",
              flush=True)

    misori_tol_stage1_deg, _src = resolve_stage1_misori_tol(misori_tol_stage1_deg, params)
    print(f"[c-parity] misori_tol_stage1={misori_tol_stage1_deg} deg ({_src})", flush=True)
    misori_tol_passa_deg, pos_tol_passa_um, _psrc = resolve_passa_tols(
        misori_tol_passa_deg, pos_tol_passa_um, params)
    print(f"[c-parity] pass_a misori<{misori_tol_passa_deg} deg AND |dpos|<{pos_tol_passa_um} um "
          f"({_psrc})", flush=True)

    if confidence_min is None:
        confidence_min = (0.05 if params.Completeness is None
                          else float(params.Completeness))
        print(f"[c-parity] confidence_min={confidence_min} "
              f"(from Completeness)" if params.Completeness is not None
              else f"[c-parity] confidence_min={confidence_min} (default; "
                   f"no Completeness in {ps_path.name})", flush=True)
    opf = np.array(read_orient_pos_fit(rd))
    key = np.array(read_key(rd))
    print(f"[c-parity] loading ProcessKey into RAM …", flush=True)
    # materialize(), not np.array(): read_process_key may return a per-seed
    # view when the C writer left a short final slot, and that view refuses
    # np.asarray so a 49 GB FitBest can never be copied by accident. The
    # ProcessKey full load IS intended here (~1.1 GB int32 at 56 k seeds) --
    # the clustering indexes it randomly across seeds.
    pk = materialize(read_process_key(rd))
    print(f"[c-parity] inputs loaded  [{time.time()-t0:.1f}s]", flush=True)

    res = run_c_parity_clustering(
        run_dir=rd, opf=opf, process_key=pk, key=key,
        space_group=int(params.SGNr),
        misori_tol_stage1_deg=misori_tol_stage1_deg,
        misori_tol_passa_deg=misori_tol_passa_deg,
        pos_tol_passa_um=pos_tol_passa_um,
        confidence_min=confidence_min,
        min_nr_spots=min_nr_spots,
        device=device,
    )

    # Side outputs.
    ih_path = rd / "IDsHash.csv"
    if not ih_path.exists():
        # Strain is the primary scientific output; refusing is the only safe
        # response. IDsHash.csv supplies the reference d-spacing d₀ per ring,
        # and Kenesei strain is (d_obs − d₀)/d₀ — substituting zeros (which is
        # what this used to do) pegs every grain at the ±0.01 bound and gives
        # RMSErrorStrain ~1e36, in a run that is otherwise correct. Observed on
        # datasetA and shade_LSHR: 100% of grains, no warning, valid positions.
        raise FileNotFoundError(
            f"{ih_path} not found. It carries the per-ring reference d-spacing "
            f"used for the Kenesei strain gauge; without it every strain value "
            f"is meaningless. It is written by midas_transforms fit_setup / "
            f"Pipeline.dump — re-run the transforms stage, or pass a run "
            f"directory that has it."
        )
    ids_hash = load_ids_hash(ih_path)

    fb = None
    try:
        fb = read_fit_best(rd)
        print(f"[c-parity] FitBest: {fb.shape}", flush=True)
    except FileNotFoundError:
        print(f"[c-parity] no FitBest.bin — strain will be Fable-only", flush=True)

    # Build the FitBest cache ONCE — both Grains.csv (Kenesei) and
    # SpotMatrix.csv use it, saving the second ~22 k × 80 KB NFS round-trip
    # over FitBest.bin.
    spot_cache = gather_per_grain_spot_data(
        res.kept_grains, fb,
        distance_um=float(params.Lsd),
        wavelength_a=float(params.Wavelength),
        ids_hash=ids_hash,
        collect_residuals=write_diagnostics,
    )

    write_grains_csv(
        out_path=out_dir / "Grains.csv",
        kept_grains=res.kept_grains,
        opf=opf, fb=fb,
        lattice_reference=np.array(params.LatticeConstant, dtype=np.float64),
        distance_um=float(params.Lsd),
        wavelength_a=float(params.Wavelength),
        space_group=int(params.SGNr),
        ids_hash=ids_hash,
        device=device,
        spot_cache=spot_cache,
    )
    write_grain_ids_key(
        out_path=out_dir / "GrainIDsKey.csv",
        kept_grains=res.kept_grains,
    )

    # Post-fit per-spot table: SpotMatrix's *Post columns and the sidecar's
    # /residuals both come from it. Optional -- an older run has none.
    fbf = None
    if write_spot_matrix or write_diagnostics:
        try:
            from ..io.binary import read_fit_best_final
            fbf = read_fit_best_final(rd)
        except FileNotFoundError:
            print("[c-parity] no FitBestFinal.bin — SpotMatrix post-fit "
                  "columns stay NaN and the sidecar has no post-fit /residuals",
                  flush=True)

    # Relative peak-fit misfit per SpotID (FitRMSE / IMax of the merged spot):
    # SpotMatrix col 28 and the sidecar's spot_rel_fit_rmse. None on older runs.
    rel_tab = None
    if write_spot_matrix or write_diagnostics:
        from ..io.csv import load_rel_fit_rmse
        rel_tab = load_rel_fit_rmse(rd)
        if rel_tab is None:
            print("[c-parity] no OrigSpotID / Radius_*.csv link — RelFitRMSE "
                  "stays NaN", flush=True)

    if write_spot_matrix and fb is not None:
        iaeif = rd / "InputAllExtraInfoFittingAll.csv"
        if iaeif.exists():
            im = load_input_extra_info_matrix(iaeif)
            # SpotDiagnostics gives the un-found expected spots (the
            # completeness deficit); FitBestFinal gives the post-fit
            # prediction. Both optional — an older run has neither, and the
            # corresponding columns stay NaN rather than 0.0.
            sd = None
            try:
                from ..io.spot_diag import load_spot_diag
                sd = load_spot_diag(rd)
            except FileNotFoundError:
                print("[c-parity] no SpotDiagnostics.bin — SpotMatrix will "
                      "carry no un-found-expected rows", flush=True)
            write_spot_matrix_csv(
                out_path=out_dir / "SpotMatrix.csv",
                kept_grains=res.kept_grains, fb=fb, input_matrix=im,
                spot_cache=spot_cache, spot_diag=sd, fb_final=fbf,
                rel_fit_rmse=rel_tab,
            )
        else:
            print(f"[c-parity] no InputAllExtraInfoFittingAll.csv — "
                  f"skipping SpotMatrix.csv", flush=True)

    if write_diagnostics:
        write_residual_diagnostics(
            out_path=out_dir / "processgrains_diagnostics.h5",
            kept_grains=res.kept_grains,
            spot_cache=spot_cache,
            fb_final=fbf,
            ids_hash=ids_hash,
            rel_fit_rmse=rel_tab,
        )

    print(f"[c-parity] DONE: {len(res.kept_grains):,} grains → {out_dir}",
          flush=True)
    return res


def write_residual_diagnostics(
    *,
    out_path: Path,
    kept_grains: List[CParityKeptGrain],
    spot_cache: Optional[list],
    fb_final=None,
    ids_hash=None,
    rel_fit_rmse=None,
) -> Optional[Path]:
    """Decompose the post-fit and pre-fit residuals and write the sidecar.

    ``rel_fit_rmse`` (the ``load_rel_fit_rmse`` table) adds, in each group,
    ``spot_rel_fit_rmse``: the relative peak-fit misfit of every ``spot_table``
    row, same order (NaN where unknown).

    Two groups, one per refiner table (see
    :mod:`compute.residual_decomposition`):

    - ``/residuals`` -- POST-fit, from ``fb_final`` (``FitBestFinal.bin``),
      at each grain's representative seed row ``rep_pos``. Needs ``ids_hash``
      for ring numbers. Absent (with a printed warning) when ``fb_final`` is
      ``None``: an older run has no post-fit table, and the pre-fit one is
      NOT substituted under the post-fit name.
    - ``/residuals_prefit`` -- PRE-fit, from the ``"resid_prefit"`` blocks
      that :func:`c_parity_emit.gather_per_grain_spot_data` collected from
      ``FitBest.bin`` with ``collect_residuals=True``.

    In both, ``grain_idx`` is the index into ``kept_grains``, i.e. **Grains.csv
    row order**, so the per-grain arrays line up with the CSV row-for-row.

    Returns the path written, or ``None`` when neither table has rows.
    """
    from ..io.consolidated import write_diagnostics_arrays
    from .residual_decomposition import (
        SPOT_RESIDUAL_COLS,
        build_residual_table,
        decompose_residuals,
        summarize_residuals,
    )

    n_grains = len(kept_grains)
    diagnostics = {}

    post = None
    if fb_final is not None and ids_hash is not None:
        post = build_residual_table(
            [g.rep_pos for g in kept_grains], fb_final, ids_hash.ring_for_spot_ids,
        )
    if post is not None and len(post):
        diagnostics["residuals"] = decompose_residuals(post, n_grains)
        diagnostics["residuals_spot_table"] = post
        print(summarize_residuals(diagnostics["residuals"], "residuals"), flush=True)
    else:
        print("[c-parity] no post-fit residuals (no FitBestFinal.bin"
              + ("" if ids_hash is not None else " or no IDsHash") + ") — "
              "processgrains_diagnostics.h5 gets /residuals_prefit only",
              flush=True)

    blocks = []
    if spot_cache is not None:
        for cache in spot_cache:
            if cache is None:
                continue
            blk = cache.get("resid_prefit")
            if blk is not None and len(blk):
                blocks.append(blk)
    if blocks:
        pre = np.concatenate(blocks, axis=0)
        diagnostics["residuals_prefit"] = decompose_residuals(pre, n_grains)
        diagnostics["residuals_prefit_spot_table"] = pre
        print(summarize_residuals(diagnostics["residuals_prefit"], "residuals_prefit"),
              flush=True)

    if rel_fit_rmse is not None:
        from ..io.csv import rel_fit_rmse_for
        for grp in ("residuals", "residuals_prefit"):
            if grp in diagnostics:
                diagnostics[grp]["spot_rel_fit_rmse"] = rel_fit_rmse_for(
                    diagnostics[grp + "_spot_table"][:, 1], rel_fit_rmse)

    if not diagnostics:
        print("[c-parity] no per-spot residuals available "
              "(no FitBest.bin?) — skipping processgrains_diagnostics.h5",
              flush=True)
        return None

    # cluster_sizes is the ONLY per-grain integer diagnostic c_parity
    # measures: the number of seeds Stage 1 + Pass A merged into this grain.
    # The spot-aware pipeline's other counters (n_resolved_hkls,
    # n_majority_hkls, n_residual_tie_hkls, n_forward_sim_hkls) describe
    # per-hkl conflict resolution that c_parity does not perform, so they are
    # omitted rather than written as zeros a reader would take for measured.
    diagnostics["cluster_sizes"] = np.array(
        [len(g.member_positions) for g in kept_grains], dtype=np.int32,
    )
    write_diagnostics_arrays(
        out_path,
        diagnostics=diagnostics,
        n_grains=n_grains,
        mode="c_parity",
        int_keys=("cluster_sizes",),
    )
    counts = ", ".join(
        f"{grp} {diagnostics[grp + '_spot_table'].shape[0]:,}"
        for grp in ("residuals", "residuals_prefit") if grp in diagnostics
    )
    print(f"[c-parity] diagnostics: spot residuals ({counts}) over "
          f"{n_grains:,} grains → {out_path}  "
          f"(columns: {','.join(SPOT_RESIDUAL_COLS)})", flush=True)
    return out_path
