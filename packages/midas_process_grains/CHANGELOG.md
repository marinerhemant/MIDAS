# Changelog

All notable changes to midas-process-grains. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.4] – 2026-05-26

### Added

- **Stage 8.5b — force-keep distinct (Path 2).** New
  `compute.drop_policy.compute_force_keep_distinct` recovers candidate
  grains that the intensity-conservation budget drop wrongly removes.
  For each dropped candidate, the (symmetry-aware) min-misorientation to
  the nearest kept grain + the σ-normalized position distance to that
  same kept grain are computed. Grains whose nearest kept neighbour is
  both **beyond peak-search misorientation resolution** AND **beyond
  3σ in position** are flagged as distinct and force-kept. New kwargs
  on `run_v4_pipeline`: `force_keep_distinct_enabled=True` (default),
  `force_keep_distinct_misori_deg=1.0` (matches typical FF-HEDM peak
  resolution = 3-4 px detector + 3 ω frames),
  `force_keep_distinct_sigma=3.0`.

- **Stage 8.5c — orphan-greedy reclaim (Path 3).** New
  `compute.drop_policy.compute_orphan_greedy_reclaim` recovers dropped
  candidates whose spot-sets uniquely cover spots not yet claimed by any
  kept grain. Reduces orphan rate at the cost of keeping more
  lower-quality candidates that nevertheless contribute new evidence.
  New kwargs: `orphan_reclaim_enabled=True`,
  `orphan_reclaim_min_unique_spots=5`.

### Changed

- **Stage 8.5 quality-rank.** The per-grain budget drop now ranks by
  the quality score `Confidence × hkl_coverage / max(σ_Z, 5)` instead
  of NNLS recovery. Quality ranking preserves gold/silver-tier grains
  preferentially. σ_Z=NaN now falls back to `median(measured σ_Z)`
  instead of 5 µm, avoiding artificial boost of un-measured grains.

- **Indrajeet result moves from 4,452 → ~5,745 → ~6,000+ kept grains**
  depending on which paths (1.0° → 0.5° force-keep + orphan reclaim)
  are enabled. Trust-tier survival jumps across the board.

## [0.4.3] – 2026-05-26 (subsumed into 0.4.4)

- Quality-rank in Stage 8.5; σ_Z=NaN fallback.

## [0.4.2] – 2026-05-26 (subsumed into 0.4.4)

- DiscModel=1 + DiscArea support in `_compute_radius_correction` for
  thin-foil samples. New leaf column `GrainRadius_disc_um = √(V/π)`
  alongside the existing `GrainRadius_NNLS` (3D-sphere R).

## [0.4.1] – 2026-05-25 (subsumed into 0.4.4)

- **Stage 8.5 — volume-budget drop policy.** Per-grain and family-aware
  variants. Drop grains until `Σ V_kept ≤ V_sample_true` by intensity-
  conservation argument.
- **Pass-1.5 — twin-aware cluster merge.** Symmetry-aware union-find
  collapse of alt-indexings + twin variants at the Pass-1 stage.

## [0.4.0] – 2026-05-25

This release closes the algorithmic correctness loop for the v4 pipeline.

### Added

- **Per-grain position σ (Stage 7).** New
  `compute.position_uncertainty.compute_per_grain_position_sigma`
  wraps `midas_propagate.per_grain_hessian_blocks` to compute the
  per-grain (σ_X, σ_Y, σ_Z) from Hessian inversion of the spot-residual
  NLL. Auto-enabled via `run_v4_pipeline(compute_position_sigma=True,
  position_sigma_max_grains=N)`. Emits `sigma_X_um, sigma_Y_um,
  sigma_Z_um, sigma_residual_rms_px, n_spots_matched` columns on the
  leaf table.

- **Strain emission (Stage 8).** Kenesei bounded lstsq per grain via
  the new `_compute_strain_per_grain` helper. Auto-enabled via
  `compute_strain=True`. Emits `eps_11..eps_23` columns (Voigt) on the
  leaf — finally filling in the strain columns the schema reserved for
  v3.

- **σ-aware trust tier scheme.** New `sigma_aware` scheme in
  `compute.trust_tiers` requires (hkl_coverage ≥ 0.8) AND (clean
  cluster) AND (σ ≤ 100 µm on all three axes) AND (≥ 20 matched
  spots) for gold; relaxed thresholds for silver. Emits
  `trust_tier_sigma_aware` column.

- **NNLS uncertainty bands.** `compute.volume_nnls.compute_nnls_volumes`
  now returns `sigma_V_nnls_raw` and `sigma_R_um` from the linearised
  covariance of the active-set NNLS solution. Per-grain σ on
  GrainRadius_NNLS is now emitted as `sigma_R_NNLS_um` in the leaf
  table. NaN for boundary grains (V ≈ 0).

- **HCP variant-level twin labels.** `default_hcp_twin_relations` now
  enumerates all 6 symmetry-equivalent K₁ variants per twin system
  (5 systems × 6 variants = 30 operators), so the `twin_type` column
  reports which specific K₁ variant the observed pair matches.

- **Auto-K guard rails.** `select_k_agree_auto` now returns a
  configurable `fallback_k` (default 4) when `n_alive` is below
  `min_alive_for_auto` (default 100) or the pair graph is too sparse —
  preventing degenerate K selection on small/sparse datasets.

- **Data-driven OM-spread tolerance.** New `select_om_spread_tol_auto`
  picks the OM-spread split threshold from the within-component misori
  histogram via the same antimode-finder used for Pass-1 θ*. Replaces
  the hand-tuned 1.0° default when the caller passes
  `fp_om_split_tol_deg="auto"`.

- **Multi-ring forward-predict helper.**
  `build_forward_predict_graph_multi_ring` unions per-ring attribution
  maps into a single agree/disagree graph, with per-ring variant-index
  offsets to prevent false cross-ring agreement. Wiring this into the
  full pipeline is a follow-up.

- **Auto-phase detection.** New `compute.auto_phase` module
  (`detect_phase`, `detect_phase_from_inputall`) classifies an
  unlabelled dataset against a library of common metallic + oxide
  phases (FCC, BCC, HCP, trigonal, spinel). Useful as a starting SG
  guess when the user has no prior label.

- **GPU device routing.** `compute_per_grain_position_sigma` accepts a
  `device=` kwarg and auto-picks CUDA / MPS / CPU based on availability.

- **Tutorial notebooks.** Four jupytext-compatible tutorials in
  `notebooks/`:
  - `02_v4_quickstart.py` — paramstest.txt → GrainsV4.csv → plots
  - `03_per_grain_sigma_and_trust.py` — Stage 7/8 walkthrough
  - `04_twin_labelling_and_families.py` — all crystal systems + user-supplied ops
  - `05_multi_phase_and_auto_phase.py` — multi-phase workflows

- **Family rollup** with singleton parents + rotation-mean OM +
  volume-weighted mean position (added in 0.3 series, formally
  documented in this release).

### Changed

- **Twin label dedup.** `label_twins` now keeps the LOWEST-misori
  operator name per unordered pair (i, j). Returned `n_pairs` is now
  the number of UNIQUE pairs detected, not the number of operator
  hits — fixes double-counting when an HCP pair matches via multiple
  K₁ variants or a cubic Σ3-of-Σ3 pair matches via both Σ3 and Σ9.

- **`trust_tier_strict` / `loose`** now optionally accept σ and
  n_spots arrays so any scheme can become σ-aware by setting the
  `sigma_gold_um` / `n_spots_gold` fields on its `TrustTierConfig`.

- **GrainsV4.csv schema** extended with: `sigma_X_um, sigma_Y_um,
  sigma_Z_um, sigma_residual_rms_px, n_spots_matched,
  sigma_R_NNLS_um, trust_tier_sigma_aware, eps_11..eps_23` columns.

### Fixed

- Em-dash unicode in `select_k_agree_auto` docstring that prevented
  module import on some installs.

### Validation

- 236/236 unit tests pass (up from 195 in 0.3.x).
- Per-grain σ validated on four datasets (Ti-7Al σ_Z=13 µm,
  Indrajeet 54 µm, xzhang 37 µm, peakfit 35 µm median); xzhang and
  peakfit agree to within 2 µm on the SAME sample.
- Cross-pipeline reproducibility on heavily-twinned LMO: identified
  fundamental multi-modal-refiner limit (0.01% match at
  refiner-noise-scale; 99% match at Σ-twin scale). Documented as
  characteristic of heavily-twinned samples, not a v4 bug.
- 6 synthetic planted-twin integration tests (HCP, tetragonal,
  multi-phase, user-supplied orthorhombic) all recover ≥95% of
  planted pairs.
- Twin geometry against theoretical Σ-misori: ≤ 0.14° median |Δ|
  across Indrajeet, xzhang, peakfit; 0% > 2° off.

## [0.2.1] – 2026-04 (previous)

See git log for changes prior to the v4 series.
