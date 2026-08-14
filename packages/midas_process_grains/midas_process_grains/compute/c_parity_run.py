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
    # ProcessKey.bin can be one row short of OPF due to C pwrite alignment.
    # Truncate everything to the common safe length; the dropped trailing
    # seed (if any) gets NrIDsPerID=0 anyway and contributes nothing.
    n_seeds = min(opf.shape[0], process_key.shape[0], key.shape[0])
    if not (n_seeds == opf.shape[0] == process_key.shape[0] == key.shape[0]):
        print(f"[c-parity] truncating to common length {n_seeds:,} "
              f"(OPF={opf.shape[0]}, PK={process_key.shape[0]}, "
              f"Key={key.shape[0]})", flush=True)
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


def run_c_parity_pipeline_from_disk(
    *,
    run_dir: Path,
    out_dir: Path,
    misori_tol_stage1_deg: float = 0.4,
    misori_tol_passa_deg: float = 0.1,
    pos_tol_passa_um: float = 5.0,
    confidence_min: Optional[float] = None,
    min_nr_spots: Optional[int] = None,
    write_spot_matrix: bool = True,
    device: str = "cpu",
    paramstest: Optional[Union[str, Path]] = None,
) -> CParityResult:
    """End-to-end C-parity replica.

    Reads paramstest, OPF, Key, ProcessKey, FitBest from ``run_dir``;
    runs Stage 1 + Pass A + confidence filter; writes Grains.csv,
    GrainIDsKey.csv, and (if FitBest available) SpotMatrix.csv to
    ``out_dir`` in C ProcessGrains format.

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
    from ..io.binary import read_orient_pos_fit, read_process_key, read_key, read_fit_best
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
    pk = np.array(read_process_key(rd), copy=True)
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

    if write_spot_matrix and fb is not None:
        iaeif = rd / "InputAllExtraInfoFittingAll.csv"
        if iaeif.exists():
            im = load_input_extra_info_matrix(iaeif)
            write_spot_matrix_csv(
                out_path=out_dir / "SpotMatrix.csv",
                kept_grains=res.kept_grains, fb=fb, input_matrix=im,
                spot_cache=spot_cache,
            )
        else:
            print(f"[c-parity] no InputAllExtraInfoFittingAll.csv — "
                  f"skipping SpotMatrix.csv", flush=True)

    print(f"[c-parity] DONE: {len(res.kept_grains):,} grains → {out_dir}",
          flush=True)
    return res
