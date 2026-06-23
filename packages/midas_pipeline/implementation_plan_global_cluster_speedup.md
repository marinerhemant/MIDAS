# find_grains global_cluster speedup — implementation plan

## Problem
`find_grains/_cluster.py::global_cluster` is O(N²) greedy single-link clustering of
per-voxel best orientations by misorientation < `maxAngle` (default 1°). It relies on
"a seed marks many neighbours → remaining set shrinks fast." **Deformed** samples spread
per-voxel orientations beyond 1° → few marked per seed → degrades to full N²/2
misorientation evals on **CPU numpy**. For the 220N CP-Ti run (N=16641 voxels) this is 7h+.

Confirmed: backend is numpy/CPU; deformed-specific because intragranular spread defeats the
marking shortcut. General bug: scales badly with voxel count AND deformation.

## Constraint
**Byte-parity** with the current CPU reference (same greedy clusters, same `voxel_to_unique`
and `unique_key_arr`). Strategy: change ONLY how the "close" relation is computed; keep the
exact greedy marking loop (i=0..N, j>i, first-seed-wins, best-conf representative).

## Fix 1 — GPU-tiled full adjacency (quick win, parity-safe)
- Compute the N×N boolean "close" relation (`misorientation < maxAngle`) once, on GPU, in
  row-tiles via `misorientation_om_batch` (already torch-capable). fp64 for threshold parity.
- Run the existing greedy marking on the boolean adjacency (CPU, cheap bool ops — no
  misorientation recompute). Identical result to reference.
- Memory: N×N bool = 277 MB (N=16641); tile rows to bound the symmetry intermediate.
- ~100× constant speedup → 7h → minutes.

## Fix 2 — Fundamental-zone binning (algorithmic, O(N·k))
- Convert per-vox OMs → quats → fundamental zone. Bin FZ quats into a coarse grid (cell ≈
  maxAngle). For each voxel, candidate neighbours = same + adjacent bins (conservative: a
  superset of all voxels within maxAngle, incl. FZ-boundary symmetry images).
- Build a SPARSE close adjacency from candidate pairs only → same greedy marking → parity.
- Avoids N² entirely; scales to large/deformed maps.

## API
- Keep `global_cluster(...)` as the reference (rename internal use to `_global_cluster_reference`
  for tests; public name unchanged).
- Add `global_cluster_fast(..., device=None, use_binning=True)` that builds the adjacency via
  binning (Fix 2) and/or GPU tiles (Fix 1), then runs the shared greedy marker
  `_greedy_from_adjacency(close, confs, keys)`.
- `find_grains_single` calls `global_cluster_fast`; env/arg to force the reference.

## Verification
- Unit test `test_global_cluster_parity.py`: synthetic OMs (tight clusters + 1–3° spread +
  random), assert `voxel_to_unique` and `unique_key_arr` IDENTICAL across
  reference / GPU-full / FZ-binned, for cubic (225) and hexagonal (194) SGs.
- Then deploy to copland, re-run the stuck `global_cluster` step from the consolidated bin
  files to produce the 220N Grains.csv.
