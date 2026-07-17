# P2-10: new-pipeline completeness ≈ 0.5× legacy — code-inspection findings

Status: **code trace complete (2026-07-16); data confirmation deferred to
copland** (user decision). Symptom: same data/params, new-pipeline mean
per-voxel completeness 0.45 vs legacy 0.81 on the Wenxi CP-Ti layer, with
orientations identical (0.30° median, 97.7% < 5° including the voxels the
`Completeness 0.4` gate drops).

## Where each side computes completeness

**New pipeline** (the value gated by find_grains and written to
`Result_OrientPos_voxel_<v>.csv` col 26):
`midas_fit_grain/scan_driver.py:417-430` —

```
n_expected = spots_final.valid.sum()          # post-refinement forward
completeness = n_matched / max(n_expected, 1)
```

`spots_final.valid` (midas_diffract `forward.py:880-945, 1285-1300`) =
**Ewald-solution exists × eta ≥ min_eta × frame ∈ [0, n_frames) ×
on-detector (full panel)**.

**Legacy C** (`FF_HEDM/src/FitOrStrainsScanningOMP.c:397-407, 96-99`):
`CalcDiffractionSpots(Lsd, MinEta, OmegaRanges, nOmegaRanges, hkls, nhkls,
BoxSizes, &nTspots, ...)` → nExpected = theoretical spots passing
**Ewald × MinEta × OmegaRanges WINDOWS × per-window BoxSizes**.

## Divergence candidates (ranked)

1. **Omega-window + BoxSizes filters are absent from the Python
   denominator.** C excludes theoretical spots outside the measured
   omega windows and outside the per-window detector box; Python's
   `frame_ok` only bounds [0, n_frames). Bites hard on shadow-gapped
   multi-`OmegaRange` data (SOH: 299°/360° coverage → ≥1.2×) and on any
   run with BoxSizes smaller than the panel. For a full-circle,
   full-panel Wenxi run this is a no-op — so it cannot alone explain
   1.8× there, but it IS a real convention difference to fix regardless.
2. **The legacy 0.81 may not be the C refiner's number at all.** In the
   legacy pf_MIDAS chain the per-voxel completeness that reaches the
   final map can come from `IndexerScanningOMP`'s frac (numerator/
   denominator over the ring subset used for indexing, ring-filtered by
   `SpotsToIndex`), while the new pipeline reports the REFINER's
   completeness over all configured rings. Different ring universes →
   systematically different denominators. Check which file the legacy
   0.81 was read from (`microstructure_pf.h5` field provenance).
3. **n_frames / frame_ok mismatch** — if the model was built with the
   OmeBinSize-fallback step (P0-3, now fixed) the frame window could be
   wrong in either direction. Post-P0-1/P0-3 paramstest carries explicit
   OmegaStart/OmegaStep, so this candidate is closed for new runs but
   may explain historical numbers.
4. Friedel double-count is NOT a suspect on inspection: both sides count
   both Ewald branches per hkl row, and hkls.csv carries the same full
   symmetry orbits for both.

## Decisive experiment (copland, ~30 min)

For ~10 voxels of the Wenxi layer (recon at
`/home/s20a/sharma_analysis/...` per PF_PFODF_FIX_REQUIREMENTS assets):

1. Dump Python's `spots_final` (y, z, ω, ring, valid) per voxel — one
   extra `model(...)` call in a scratch script.
2. Run the C `CalcDiffractionSpots` on the same (OM, pos, lattice) —
   `FitOrStrainsScanningOMP` linked as in the smoke build — and dump
   `TheorSpots`.
3. Diff the two lists. The spots present in Python-valid but absent from
   C-nTspots identify the missing filter directly (their ω/box values
   will cluster in the shadow gaps or outside BoxSizes if candidate 1;
   whole rings if candidate 2).

## Interim guidance (already documented in the handoff corrections)

- New-pipeline completeness ≈ 0.5× legacy on Wenxi-class data;
  a `Completeness 0.4` gate silently drops ~42% of voxels legacy keeps.
  **Use 0.2 with the new pipeline until the convention is unified.**
- The likely fix once confirmed: apply the OmegaRanges/BoxSizes masks to
  `n_expected` in `scan_driver.py` (cfg already carries both), and/or
  report both "refiner completeness" and "indexer frac" explicitly so the
  two conventions stop sharing one name.
