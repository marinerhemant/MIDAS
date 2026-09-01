"""Byte-parity Python port of ``NF_HEDM/src/Mic2GrainsList.c``.

Reads a ``.mic`` text file (output of :mod:`parse_mic`), clusters its
voxels into unique grains by orientation similarity (and optionally
also by spatial proximity), and writes a ``Grains.csv`` consumable by
:program:`GenSeedOrientationsFF2NFHEDM`.

All orientation / misorientation math goes through
:mod:`midas_stress.orientation` (the canonical port of MIDAS C's
``GetMisorientation.h``).

Usage
-----

CLI mirror::

    midas-nf-mic2grains <ParamFile> <MicFile> <OutFile>
                        [doNeighborSearch=0] [nCPUs=...] [minConfOverride=...]

Python::

    from midas_nf_pipeline.mic2grains import mic_to_grains, Mic2GrainsParams
    mic_to_grains(Mic2GrainsParams(
        param_file='ps.txt', mic_file='Au.mic', out_file='Grains.csv',
        do_neighbor_search=0,
    ))
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Mic2GrainsParams:
    """Mirror of the Mic2GrainsList CLI argument bundle."""
    param_file: str | Path
    mic_file: str | Path
    out_file: str | Path
    do_neighbor_search: int = 0
    n_cpus: int = 1                                 # informational only
    min_conf_override: Optional[float] = None


@dataclass
class _ParsedParams:
    """Subset of the param-file keys we actually use."""
    sg_nr: int = 225
    max_angle_deg: float = 1.0
    min_conf: float = 0.04
    lattice_params: List[float] = field(default_factory=lambda: [0.0] * 6)


def _read_param_file(path: str | Path) -> _ParsedParams:
    """Replicate the C ``Parse Parameter File`` block (lines 130-164)."""
    p = _ParsedParams()
    with open(path, "r") as f:
        for raw in f:
            if not raw or raw[0] == "#":
                continue
            if raw.startswith("LatticeParameter"):
                tokens = raw.split()
                if len(tokens) >= 7:
                    p.lattice_params = [float(x) for x in tokens[1:7]]
                continue
            if raw.startswith("SpaceGroup"):
                tokens = raw.split()
                if len(tokens) >= 2:
                    p.sg_nr = int(tokens[1])
                continue
            if raw.startswith("MaxAngle"):
                tokens = raw.split()
                if len(tokens) >= 2:
                    p.max_angle_deg = float(tokens[1])
                continue
            if raw.startswith("MinFracAccept"):
                tokens = raw.split()
                if len(tokens) >= 2:
                    p.min_conf = float(tokens[1])
                continue
            if raw.startswith("MinConfidence"):
                tokens = raw.split()
                if len(tokens) >= 2:
                    p.min_conf = float(tokens[1])
                continue
    return p


def _read_mic_text(path: str | Path, min_conf: float) -> tuple[float, np.ndarray]:
    """Read the mic text, return (TriEdgeSize, voxel_array (N, 11)).

    Skip ``%`` header lines (parse ``%TriEdgeSize`` if present); discard
    rows with fewer than 11 floats or with confidence < min_conf.
    """
    tri_edge_size = 0.0
    rows: List[List[float]] = []
    with open(path, "r") as f:
        for raw in f:
            if not raw:
                continue
            if raw[0] == "%":
                if raw.startswith("%TriEdgeSize"):
                    parts = raw.split()
                    if len(parts) >= 2:
                        try:
                            tri_edge_size = float(parts[1])
                        except ValueError:
                            pass
                continue
            tokens = raw.split()
            if len(tokens) < 11:
                continue
            try:
                vals = [float(t) for t in tokens[:11]]
            except ValueError:
                continue
            if vals[10] < min_conf:
                continue
            rows.append(vals)
    arr = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 11), dtype=np.float64)
    return tri_edge_size, arr


# ---------------------------------------------------------------------------
#  Orientation conversions — strictly via midas_stress.
# ---------------------------------------------------------------------------

def _eulers_to_oms(eulers: np.ndarray) -> np.ndarray:
    """``(N, 3)`` Bunge ZXZ Euler (rad) → ``(N, 9)`` row-major orient mats.

    Direct call to :func:`midas_stress.orientation.euler_to_orient_mat_batch`
    (no other wrappers).
    """
    from midas_stress.orientation import euler_to_orient_mat_batch
    eul = np.asarray(eulers, dtype=np.float64)
    if eul.ndim == 1:
        eul = eul[None, :]
    return np.asarray(euler_to_orient_mat_batch(eul), dtype=np.float64).reshape(-1, 9)


def _miso_om_batch_rad(om_a_flat: np.ndarray, om_b_flat: np.ndarray, sg_nr: int) -> np.ndarray:
    """Batched misorientation in radians via midas-stress's torch path.

    Both inputs are ``(N, 9)`` row-major flat OMs. We pass torch tensors
    so midas-stress dispatches to the vectorised kernel.
    """
    import torch
    from midas_stress.orientation import misorientation_om_batch
    a = torch.as_tensor(np.ascontiguousarray(om_a_flat, dtype=np.float64))
    b = torch.as_tensor(np.ascontiguousarray(om_b_flat, dtype=np.float64))
    return misorientation_om_batch(a, b, sg_nr).detach().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
#  Clustering kernels (global vs spatial)
# ---------------------------------------------------------------------------

def _global_merge(
    oms_flat: np.ndarray, xs: np.ndarray, ys: np.ndarray,
    max_angle_rad: float, sg_nr: int,
) -> List[Tuple[np.ndarray, float, float, int]]:
    """Walk grains in confidence-descending order; for each unused grain,
    flag every remaining unused grain whose misorientation is below
    ``max_angle_rad`` as part of the same grain.

    Returns a list of ``(orient_mat_9, x, y, n_voxels)`` per unique grain.
    """
    n = oms_flat.shape[0]
    used = np.zeros(n, dtype=bool)
    out: List[Tuple[np.ndarray, float, float, int]] = []
    for i in range(n):
        if used[i]:
            continue
        used[i] = True
        # Compare grain i against every remaining unused grain in one batch.
        rest = np.nonzero(~used)[0]
        n_vox = 1
        if rest.size > 0:
            om_i = np.broadcast_to(oms_flat[i:i + 1], (rest.size, 9)).copy()
            misos = _miso_om_batch_rad(om_i, oms_flat[rest], sg_nr)
            close = misos < max_angle_rad
            close_idx = rest[close]
            used[close_idx] = True
            n_vox += int(close.sum())
        out.append((oms_flat[i], float(xs[i]), float(ys[i]), n_vox))
    return out


def _spatial_merge(
    oms_flat: np.ndarray, xs: np.ndarray, ys: np.ndarray,
    max_angle_rad: float, sg_nr: int,
    tri_edge_size: float,
) -> List[Tuple[np.ndarray, float, float, int]]:
    """Region-growing BFS in 3×3 spatial bins of side ``tri_edge_size *
    1.01``; an edge is added when both spatial distance² <
    ``(tri_edge * 2)²`` and orientation misorientation < ``max_angle_rad``.
    Mirrors NF_HEDM/src/Mic2GrainsList.c:325-437.
    """
    n = oms_flat.shape[0]
    if n == 0:
        return []
    if tri_edge_size <= 1e-6:
        # Defensive fallback: behave like the C does when TriEdgeSize is missing.
        return _global_merge(oms_flat, xs, ys, max_angle_rad, sg_nr)

    bin_size = tri_edge_size * 1.01
    min_x, min_y = float(xs.min()), float(ys.min())
    max_x, max_y = float(xs.max()), float(ys.max())
    dim_x = int((max_x - min_x) / bin_size) + 2
    dim_y = int((max_y - min_y) / bin_size) + 2

    bins: List[List[int]] = [[] for _ in range(dim_x * dim_y)]
    bx_arr = ((xs - min_x) / bin_size).astype(np.int64)
    by_arr = ((ys - min_y) / bin_size).astype(np.int64)
    for i in range(n):
        bins[by_arr[i] * dim_x + bx_arr[i]].append(i)

    dist_thresh_sq = (tri_edge_size * 2.0) ** 2
    used = np.zeros(n, dtype=bool)
    out: List[Tuple[np.ndarray, float, float, int]] = []

    from collections import deque
    for seed in range(n):
        if used[seed]:
            continue
        used[seed] = True
        q: "deque[int]" = deque([seed])
        seed_om = oms_flat[seed]
        n_vox = 1
        while q:
            cur = q.popleft()
            cx, cy = xs[cur], ys[cur]
            bx, by = int((cx - min_x) / bin_size), int((cy - min_y) / bin_size)
            cand: List[int] = []
            for ny in range(by - 1, by + 2):
                if ny < 0 or ny >= dim_y:
                    continue
                for nx in range(bx - 1, bx + 2):
                    if nx < 0 or nx >= dim_x:
                        continue
                    for j in bins[ny * dim_x + nx]:
                        if used[j]:
                            continue
                        dx = xs[j] - cx
                        dy = ys[j] - cy
                        if dx * dx + dy * dy < dist_thresh_sq:
                            cand.append(j)
            if not cand:
                continue
            cand_arr = np.asarray(cand, dtype=np.int64)
            om_seed = np.broadcast_to(seed_om[None, :], (cand_arr.size, 9)).copy()
            misos = _miso_om_batch_rad(om_seed, oms_flat[cand_arr], sg_nr)
            for jj, miso in zip(cand_arr, misos):
                if used[jj]:
                    continue
                if miso < max_angle_rad:
                    used[jj] = True
                    q.append(int(jj))
                    n_vox += 1
        out.append((seed_om, float(xs[seed]), float(ys[seed]), n_vox))
    return out


# ---------------------------------------------------------------------------
#  Output writer
# ---------------------------------------------------------------------------

#: The 24 columns a Mic2GrainsList Grains.csv actually carries, in order.
#: Kept as a module constant so tests (and any downstream tool) can assert the
#: header against the same list the writer uses.
GRAINS_COLUMN_NAMES: List[str] = (
    ["GrainID"]
    + [f"O{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
    + ["X", "Y", "Z", "a", "b", "c", "alpha", "beta", "gamma",
       "DiffPos", "DiffOme", "DiffAngle", "GrainRadius", "Confidence"]
)


def _write_grains_csv(
    out_path: str | Path,
    unique: List[Tuple[np.ndarray, float, float, int]],
    *,
    sg_nr: int,
    max_angle_deg: float,
    min_conf: float,
    lattice_params: List[float],
    do_neighbor_search: int,
    tri_edge_size: float,
    mic_file: str | Path,
) -> None:
    """Write the 9-line header + per-grain rows.

    The DATA rows match ``NF_HEDM/src/Mic2GrainsList.c:440-482`` byte-for-byte.
    The COLUMN-HEADER line (line 2) deliberately does NOT: the C wrote prose,

        %GrainID OrientMat(9) X Y Z LatC(6) 0 0 0 Radius Confidence

    which names no columns, so every name-driven reader in the tree
    (``midas_process_grains.io.read``, ``midas_pipeline.seeding.handoff``,
    ``midas_dct_tt.chain``, ``midas_stress.io``, ``midas_plotting.grains``)
    raised on NF seed files even though the 24 data columns were already
    correctly aligned with the ``ProcessGrains`` schema. We now write the real
    column names instead.

    **This is safe for the positional consumers** -- checked before changing:
    the FF indexer reads ``GrainsFile`` with ``fgets`` for ``%NumGrains``,
    then *eight* content-ignored ``fgets`` skips, then a positional ``sscanf``
    (``FF_HEDM/src/IndexerOMP.c:2358-2383``, and its Python port
    ``midas_index/io/csv.py:read_grains_csv``). The header block is still
    exactly nine lines and the data rows are untouched, so both see exactly
    what they saw before. ``tests/test_mic2grains.py::test_header_matches_c``
    already compares only header lines 0 and 3-8, skipping this one.

    Column mapping (24 columns, space-separated, left-aligned with the
    ``ProcessGrains`` schema)::

        0        GrainID
        1..9     O11..O33     orientation matrix, row-major
        10..12   X Y Z        um; NF is one layer, so Z is 0
        13..18   a b c alpha beta gamma   (the reference lattice, unrefined)
        19..21   DiffPos DiffOme DiffAngle
                 Always 0: NF has no FF refinement residual to report. The C
                 wrote three literal zeros here and we keep them, because
                 anything else would change what the positional readers see.
        22       GrainRadius  um, from the voxel count
        23       Confidence   always 1 (grains below MinConfidence are dropped
                 upstream, so every emitted grain is fully confident by
                 construction)

    Per-row format string::

        %d %.12lf %.12lf %.12lf %.12lf %.12lf %.12lf %.12lf %.12lf %.12lf
        %.6lf %.6lf 0
        %.6lf %.6lf %.6lf %.6lf %.6lf %.6lf
        0 0 0 %.6lf 1
    """
    voxel_area = tri_edge_size * tri_edge_size * math.sqrt(3.0) / 4.0
    with open(out_path, "w") as f:
        f.write(f"%NumGrains {len(unique)}\n")
        # Real column names, not prose -- see the docstring. Space-separated
        # to match the data rows; every reader in the tree splits the header
        # on whitespace.
        f.write(" ".join(["%GrainID"] + GRAINS_COLUMN_NAMES[1:]) + "\n")
        f.write(f"%Generated by Mic2GrainsList from {mic_file}\n")
        f.write(f"%SpaceGroup {sg_nr}\n")
        f.write(f"%MaxAngle {max_angle_deg:.4f}\n")
        f.write(f"%MinConfidence {min_conf:.4f}\n")
        f.write(
            f"%LatticeParameter {lattice_params[0]:.6f} {lattice_params[1]:.6f} "
            f"{lattice_params[2]:.6f} {lattice_params[3]:.6f} "
            f"{lattice_params[4]:.6f} {lattice_params[5]:.6f}\n"
        )
        f.write(f"%DoNeighborSearch {do_neighbor_search}\n")
        f.write(f"%TriEdgeSize {tri_edge_size:.6f}\n")

        for i, (om, gx, gy, n_vox) in enumerate(unique):
            area = n_vox * voxel_area
            radius = math.sqrt(area / math.pi) if area > 0 else 0.0
            om_str = " ".join(f"{v:.12f}" for v in om)
            lat_str = " ".join(f"{v:.6f}" for v in lattice_params)
            f.write(
                f"{i + 1} {om_str} "
                f"{gx:.6f} {gy:.6f} 0 "
                f"{lat_str} "
                f"0 0 0 {radius:.6f} 1\n"
            )


# ---------------------------------------------------------------------------
#  Top-level entry
# ---------------------------------------------------------------------------

def mic_to_grains(args: Mic2GrainsParams) -> int:
    """Run the full Mic2GrainsList pipeline. Returns ``nUnique``."""
    p = _read_param_file(args.param_file)
    if args.min_conf_override is not None:
        p.min_conf = float(args.min_conf_override)

    tri_edge_size, voxels = _read_mic_text(args.mic_file, p.min_conf)
    if voxels.shape[0] == 0:
        # Empty mic → write only the header.
        _write_grains_csv(
            args.out_file, [],
            sg_nr=p.sg_nr,
            max_angle_deg=p.max_angle_deg,
            min_conf=p.min_conf,
            lattice_params=p.lattice_params,
            do_neighbor_search=int(args.do_neighbor_search),
            tri_edge_size=tri_edge_size,
            mic_file=args.mic_file,
        )
        return 0

    # Sort by confidence descending — stable sort to preserve the C's
    # qsort-by-confidence-only ordering of the *seed* list (intra-tie
    # ordering can vary between unstable qsort and stable Python sort,
    # but the merge result's contents and grain count don't depend on
    # the intra-tie order because every same-orientation voxel within
    # the tolerance gets absorbed regardless).
    eulers = voxels[:, 7:10]
    confs = voxels[:, 10]
    xs = voxels[:, 3]
    ys = voxels[:, 4]
    order = np.argsort(-confs, kind="stable")
    eulers = eulers[order]
    xs = xs[order]
    ys = ys[order]

    oms_flat = _eulers_to_oms(eulers)

    do_neighbor = int(args.do_neighbor_search)
    if do_neighbor and tri_edge_size <= 1e-6:
        do_neighbor = 0

    max_ang_rad = p.max_angle_deg * math.pi / 180.0
    if do_neighbor:
        unique = _spatial_merge(oms_flat, xs, ys, max_ang_rad, p.sg_nr, tri_edge_size)
    else:
        unique = _global_merge(oms_flat, xs, ys, max_ang_rad, p.sg_nr)

    _write_grains_csv(
        args.out_file, unique,
        sg_nr=p.sg_nr,
        max_angle_deg=p.max_angle_deg,
        min_conf=p.min_conf,
        lattice_params=p.lattice_params,
        do_neighbor_search=do_neighbor,
        tri_edge_size=tri_edge_size,
        mic_file=args.mic_file,
    )
    return len(unique)


# ---------------------------------------------------------------------------
#  CLI helper (for the umbrella ``midas-nf-pipeline`` entry point)
# ---------------------------------------------------------------------------

def main_cli(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="midas-nf-mic2grains",
        description="Cluster a .mic file into unique grains "
                    "(byte-parity port of Mic2GrainsList.c).",
    )
    ap.add_argument("param_file")
    ap.add_argument("mic_file")
    ap.add_argument("out_file")
    ap.add_argument(
        "do_neighbor_search", nargs="?", type=int, default=0,
        help="0 = global orientation merge (default), 1 = spatial BFS.",
    )
    ap.add_argument(
        "n_cpus", nargs="?", type=int, default=1,
        help="Informational; preserved for argv parity with the C CLI.",
    )
    ap.add_argument(
        "min_conf_override", nargs="?", type=float, default=None,
        help="Optional command-line MinConfidence override.",
    )
    ns = ap.parse_args(argv)
    n_unique = mic_to_grains(Mic2GrainsParams(
        param_file=ns.param_file,
        mic_file=ns.mic_file,
        out_file=ns.out_file,
        do_neighbor_search=int(ns.do_neighbor_search),
        n_cpus=int(ns.n_cpus),
        min_conf_override=ns.min_conf_override,
    ))
    print(f"Values written: {n_unique} unique grains found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
