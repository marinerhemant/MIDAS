"""Null-gated seed augmentation for seeded PF indexing.

Seeded point-by-point indexing can only map orientations that are in its seed,
and the merged-FF seed is what survives process-grains' clustering of the
merged-FF indexer's solutions. On ESRF ma5608 that clustering kept 568 of
~2,300 distinct solutions and dropped real crystals: added back, 12 of the
dropped orientations won 4,888 voxels, three of them 60-deg twin domains that
own ~2/3 of their spots, while an omega-shuffled run's orientations won none.

This module adds such orientations back, gated by a null built from the same
indexer: a candidate is added only if its completeness is higher than the best
completeness any distinct orientation reaches in an omega-shuffled run of the
same merged spots (spots permuted in omega within each ring; every marginal the
indexer sees is kept, only the position-omega pairing that encodes orientation
is destroyed). On ma5608 the shuffled runs never exceeded 0.812-0.841 while all
12 real winners sat at 0.89-0.94.

Workflow
--------
1. ``shuffle_ff_inputs(src_layer, dst_layer)`` writes an omega-shuffled copy of
   a merged-FF layer's InputAll.csv / InputAllExtraInfoFittingAll.csv. Bin and
   index it with the same binary and parameters as the real run.
2. ``augment_seed(real_indexbest, [null_indexbest...], grains_csv, out_csv)``
   writes a Grains.csv = the original rows + the gated candidates.
``build_null_run`` / ``augment_from_ff_layer`` do both from a merged-FF seed layer
(the pipeline's seeding stage calls the latter when ``seeding.augment_ff_layer``
is set). All are exposed on the command line: ``python -m midas_pipeline.seeding.augment``.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

LOG = logging.getLogger("midas_pipeline.seeding.augment")
PathLike = Union[str, Path]


@dataclass
class Candidates:
    om: np.ndarray            # (n, 9) orientation matrices, row-major
    completeness: np.ndarray  # (n,)
    ia: np.ndarray            # (n,) internal angle (deg), IndexBest col 1


def read_candidates(indexbest: PathLike) -> Candidates:
    """Every solution in a consolidated ``IndexBest_all.bin`` (FF or PF)."""
    from midas_index.io.consolidated import read_index_best_all, split_records_by_voxel
    recs = split_records_by_voxel(read_index_best_all(str(indexbest)))
    rows = [np.asarray(r) for r in recs if r is not None and len(r)]
    if not rows:
        return Candidates(np.zeros((0, 9)), np.zeros(0), np.zeros(0))
    R = np.concatenate(rows, 0)
    return Candidates(R[:, 2:11].copy(), R[:, 15] / np.maximum(R[:, 14], 1.0), R[:, 1].copy())


def _min_miso(om: np.ndarray, ref: np.ndarray, sg: int) -> np.ndarray:
    """(len(om),) smallest misorientation (rad) of each row of ``om`` to any row of ``ref``."""
    from midas_stress.orientation import misorientation_om_batch
    out = np.full(len(om), np.inf)
    if len(ref) == 0 or len(om) == 0:
        return out
    for r in ref:
        out = np.minimum(out, np.asarray(
            misorientation_om_batch(om, np.tile(r, (len(om), 1)), sg)).ravel())
    return out


def distinct_far(c: Candidates, seed_om: np.ndarray, sg: int, tol_deg: float = 1.0) -> Candidates:
    """One orientation per ``tol_deg`` cluster (greedy by completeness; the set
    de-duplicated against ITSELF), then those > ``tol_deg`` from every seed row."""
    from midas_stress.orientation import misorientation_om_batch
    tol = np.radians(tol_deg)
    kept: List[int] = []
    for i in np.argsort(-c.completeness, kind="stable"):
        if kept and np.asarray(misorientation_om_batch(
                np.tile(c.om[i], (len(kept), 1)), c.om[kept], sg)).min() < tol:
            continue
        kept.append(int(i))
    k = np.array(kept, dtype=int)
    far = _min_miso(c.om[k], seed_om, sg) >= tol if len(k) else np.zeros(0, bool)
    k = k[far]
    return Candidates(c.om[k], c.completeness[k], c.ia[k])


def null_gate(null_sets: Sequence[Candidates]) -> float:
    """The highest completeness any distinct null orientation reaches."""
    vals = [float(s.completeness.max()) for s in null_sets if len(s.completeness)]
    return max(vals) if vals else float("inf")


def _seed_table(grains_csv: PathLike):
    lines = open(grains_csv).read().splitlines(keepends=True)
    head = [l for l in lines if l.startswith("%")]
    rows = [l for l in lines if not l.startswith("%") and l.strip()]
    om = np.array([[float(x) for x in r.split()[1:10]] for r in rows])
    return head, rows, om


def write_augmented(grains_csv: PathLike, out_csv: PathLike, added_om: np.ndarray) -> int:
    """Original Grains.csv rows + one row per added orientation, cloned from a
    median original row: new GrainID, O11..O33 replaced, X/Y/Z zeroed (PF
    assigns the position per voxel). Returns the number of rows written."""
    head, rows, _ = _seed_table(grains_csv)
    tmpl = rows[len(rows) // 2].split()
    gid0 = max(int(float(r.split()[0])) for r in rows)
    out = list(head)
    if out and out[0].startswith("%NumGrains"):
        out[0] = f"%NumGrains {len(rows) + len(added_om)}\n"
    out += rows
    for k, om in enumerate(added_om):
        f = list(tmpl); f[0] = str(gid0 + 1 + k)
        for j, v in enumerate(np.asarray(om).ravel()):
            f[1 + j] = f"{v:.6f}"
        f[10] = f[11] = f[12] = "0.000000"
        out.append("\t".join(f) + "\n")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(out_csv).write_text("".join(out))
    return len(rows) + len(added_om)


def augment_seed(real_indexbest: PathLike, null_indexbests: Sequence[PathLike],
                 grains_csv: PathLike, out_csv: PathLike, *, sg: int,
                 tol_deg: float = 1.0) -> dict:
    """Write ``out_csv`` = ``grains_csv`` + real merged-FF orientations that
    (a) are > ``tol_deg`` from every seed row and from each other, and (b) beat
    the best completeness of any such orientation in the omega-shuffled runs."""
    if not null_indexbests:
        raise ValueError("augment_seed needs at least one omega-shuffled run: the gate IS the null")
    _, _, seed_om = _seed_table(grains_csv)
    real = distinct_far(read_candidates(real_indexbest), seed_om, sg, tol_deg)
    nulls = [distinct_far(read_candidates(p), seed_om, sg, tol_deg) for p in null_indexbests]
    gate = null_gate(nulls)
    keep = real.completeness > gate
    n = write_augmented(grains_csv, out_csv, real.om[keep])
    rep = dict(seed_rows=len(seed_om), real_distinct_far=int(len(real.completeness)),
               null_distinct_far=[int(len(s.completeness)) for s in nulls], gate=gate,
               added=int(keep.sum()), rows_written=n,
               added_completeness=real.completeness[keep].tolist(), added_ia=real.ia[keep].tolist())
    LOG.info("seed augmentation: gate %.4f (max null completeness over %d run(s)); %d of %d distinct "
             "far real orientations added; %s rows -> %s", gate, len(nulls), rep["added"],
             rep["real_distinct_far"], n, out_csv)
    return rep


# columns permuted together, per ring, in a merged-FF layer's CSVs
_OMEGA_COLS_INPUTALL = (2,)
_OMEGA_COLS_EXTRA = (2, 8, 13)          # Omega, OmegaIni, OmegaDetCor
_RING_COL = 5


def shuffle_ff_inputs(src_layer: PathLike, dst_layer: PathLike, seed: int = 20260922) -> dict:
    """Omega-shuffled copy of a merged-FF layer's InputAll.csv and
    InputAllExtraInfoFittingAll.csv: one permutation per ring, applied
    identically to every omega column of both (row-aligned) files. Everything
    else in ``src_layer`` must be copied/linked by the caller (paramstest,
    hkls, ring tables) before binning and indexing ``dst_layer``."""
    src, dst = Path(src_layer), Path(dst_layer); dst.mkdir(parents=True, exist_ok=True)
    ha = open(src / "InputAll.csv").readline(); hx = open(src / "InputAllExtraInfoFittingAll.csv").readline()
    A = np.loadtxt(src / "InputAll.csv", skiprows=1, ndmin=2)
    X = np.loadtxt(src / "InputAllExtraInfoFittingAll.csv", skiprows=1, ndmin=2)
    if not np.array_equal(A[:, 4], X[:, 4]):
        raise ValueError("InputAll and InputAllExtraInfoFittingAll are not row-aligned (SpotID)")
    rng = np.random.default_rng(seed)
    As, Xs = A.copy(), X.copy()
    for r in np.unique(A[:, _RING_COL]):
        idx = np.flatnonzero(A[:, _RING_COL] == r); p = rng.permutation(idx)
        for c in _OMEGA_COLS_INPUTALL: As[idx, c] = A[p, c]
        for c in _OMEGA_COLS_EXTRA: Xs[idx, c] = X[p, c]
    np.savetxt(dst / "InputAll.csv", As, fmt="%.6f", header=ha.strip(), comments="")
    np.savetxt(dst / "InputAllExtraInfoFittingAll.csv", Xs, fmt="%.6f", header=hx.strip(), comments="")
    moved = float(np.mean(As[:, 2] != A[:, 2]))
    LOG.info("omega-shuffled %d spots in %d rings (%.1f%% moved) -> %s", len(A),
             len(np.unique(A[:, _RING_COL])), 100 * moved, dst)
    return dict(n_spots=int(len(A)), moved_fraction=moved)


_NULL_COPY = ("paramstest.txt", "hkls.csv", "SpotsToIndex.csv", "positions.csv")


def build_null_run(ff_layer: PathLike, dst: PathLike, *, seed: int = 20260922, n_cpus: int = 8,
                   indexer: Optional[PathLike] = None) -> Path:
    """Omega-shuffled twin of a merged-FF seed run, indexed exactly like it.

    ``ff_layer`` is the FF layer directory whose Grains.csv is the seed: it must
    hold InputAll.csv, InputAllExtraInfoFittingAll.csv and the indexing stage's
    ``paramstest_index_comp.txt``. The shuffled spots are binned with
    ``midas_transforms.bin_data`` and indexed with the same C indexer and
    parameter file (every folder key rewritten to ``dst``, so nothing points back
    at the real run). Returns ``dst/Output/IndexBest_all.bin``."""
    import re
    import shutil
    import subprocess
    src, dst = Path(ff_layer), Path(dst)
    pfile = src / "paramstest_index_comp.txt"
    if not pfile.is_file():
        raise FileNotFoundError(f"{pfile} missing: the null run must reuse the real run's indexer parameters")
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "Output").mkdir(parents=True); (dst / "Results").mkdir()
    shuffle_ff_inputs(src, dst, seed)
    for f in _NULL_COPY:
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
    t = pfile.read_text()
    t = re.sub(r"^OutputFolder .*$", f"OutputFolder {dst}/Output", t, flags=re.M)
    t = re.sub(r"^ResultFolder .*$", f"ResultFolder {dst}/Results", t, flags=re.M)
    t = t.replace(str(src), str(dst))
    (dst / "paramstest_index_comp.txt").write_text(t)
    if str(src) in t:
        raise RuntimeError("real-run path leaked into the null run's parameters")
    from midas_transforms.bin_data import bin_data
    bin_data(result_folder=dst, out_dir=dst, device="cpu", dtype="float64", write=True)
    # The null must be binned exactly like the real run: a midas_transforms whose ring-slot layout differs
    # from the one the real run (and so the indexer) used makes the indexer mis-address nData.bin or crash.
    # nData.bin is the fixed bin grid (Data.bin's size follows the spots' omegas, so it moves with the shuffle).
    f = "nData.bin"
    if (src / f).is_file() and (src / f).stat().st_size != (dst / f).stat().st_size:
        raise RuntimeError(f"{f}: null run binned to {(dst / f).stat().st_size} bytes, real run {(src / f).stat().st_size}; "
                           "midas_transforms and the indexer that made the real run differ in bin layout")
    if (src / "nData.bin").is_file() and (dst / "RingSlots.csv").is_file() != (src / "RingSlots.csv").is_file():
        raise RuntimeError("RingSlots.csv present in one of real/null runs only: bin layouts differ")
    if indexer is None:
        from midas_index import backend_c
        indexer = backend_c.binary_path()
    n_seeds = sum(1 for l in open(dst / "SpotsToIndex.csv") if l.strip())
    env = dict(__import__("os").environ, OMP_NUM_THREADS=str(n_cpus))
    with open(dst / "index_out.txt", "w") as out, open(dst / "index_err.txt", "w") as err:
        subprocess.run([str(indexer), "paramstest_index_comp.txt", "0", "1", str(n_seeds), str(n_cpus)],
                       cwd=str(dst), env=env, stdout=out, stderr=err, check=True)
    ib = dst / "Output" / "IndexBest_all.bin"
    if not ib.is_file():
        raise RuntimeError(f"null indexing wrote no {ib}; see {dst}/index_err.txt")
    LOG.info("null run: %d seeds indexed on omega-shuffled spots -> %s", n_seeds, ib)
    return ib


def augment_from_ff_layer(ff_layer: PathLike, work_dir: PathLike, *, sg: int, n_null: int = 1,
                          tol_deg: float = 1.0, n_cpus: int = 8) -> dict:
    """Build ``n_null`` omega-shuffled null runs of ``ff_layer`` and write
    ``work_dir/Grains_augmented.csv`` = ff_layer/Grains.csv + gated orientations."""
    ff_layer, work_dir = Path(ff_layer), Path(work_dir)
    nulls = [build_null_run(ff_layer, work_dir / f"null_{i}", seed=20260922 + i, n_cpus=n_cpus)
             for i in range(max(1, int(n_null)))]
    rep = augment_seed(ff_layer / "Output" / "IndexBest_all.bin", nulls, ff_layer / "Grains.csv",
                       work_dir / "Grains_augmented.csv", sg=sg, tol_deg=tol_deg)
    rep["out_csv"] = str(work_dir / "Grains_augmented.csv")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m midas_pipeline.seeding.augment",
                                 description="Null-gated seed augmentation for seeded PF indexing.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shuffle", help="write omega-shuffled merged-FF inputs")
    s.add_argument("src_layer"); s.add_argument("dst_layer"); s.add_argument("--seed", type=int, default=20260922)
    a = sub.add_parser("augment", help="write an augmented Grains.csv")
    a.add_argument("--real", required=True, help="real merged-FF IndexBest_all.bin")
    a.add_argument("--null", required=True, nargs="+", help="omega-shuffled run(s) IndexBest_all.bin")
    a.add_argument("--grains", required=True, help="seed Grains.csv (process-grains output)")
    a.add_argument("--out", required=True); a.add_argument("--sg", type=int, required=True)
    a.add_argument("--tol-deg", type=float, default=1.0)
    f = sub.add_parser("from-ff", help="build omega-shuffled null run(s) of a merged-FF seed layer and augment its Grains.csv")
    f.add_argument("ff_layer"); f.add_argument("work_dir"); f.add_argument("--sg", type=int, required=True)
    f.add_argument("--n-null", type=int, default=1); f.add_argument("--n-cpus", type=int, default=8)
    f.add_argument("--tol-deg", type=float, default=1.0)
    ns = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if ns.cmd == "shuffle":
        print(shuffle_ff_inputs(ns.src_layer, ns.dst_layer, ns.seed))
    elif ns.cmd == "from-ff":
        rep = augment_from_ff_layer(ns.ff_layer, ns.work_dir, sg=ns.sg, n_null=ns.n_null,
                                    tol_deg=ns.tol_deg, n_cpus=ns.n_cpus)
        print({k: v for k, v in rep.items() if k not in ("added_completeness", "added_ia")})
    else:
        rep = augment_seed(ns.real, ns.null, ns.grains, ns.out, sg=ns.sg, tol_deg=ns.tol_deg)
        print({k: v for k, v in rep.items() if k not in ("added_completeness", "added_ia")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
