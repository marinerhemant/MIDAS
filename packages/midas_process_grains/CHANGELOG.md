# Changelog

All notable changes to midas-process-grains. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.10.0] – 2026-08-21

### Closed open question — the post-fit spot loss is a GATE-BOUNDARY effect

- 0.03 % of matched spots (66 of 201,657 on the reference layer, across 50 of
  1,988 seeds) have no post-fit match in `FitBestFinal.bin`. **Explained, not a
  defect**: they sit at the boundaries of the forward model's own generation
  gates, so when the fit moves orientation and lattice the reflection stops
  being generated and the observed spot has no candidate left.
  - within 2x `MinEta` of the eta pole: **33.3 %** of lost vs **1.2 %** of kept
    (28x enrichment); at |omega| > 170 deg: **34.8 %** vs **5.9 %** (5.9x).
  - Not the matching caps: 0 % of lost spots are near the 1.0 deg
    internal-angle cap or the 5 deg omega window.
  - Not "the fit discards bad spots": their median pre-fit `DiffLen` is
    *lower* than the population (571.8 vs 593.9 um). The one elevated
    statistic is the omega residual (0.314 vs 0.131 deg), which is the tell —
    those are the spots nearest the omega boundary.

### Added

- **Reads the 33-column `OrientPosFit.bin`** written by midas-fit-grain
  >= 0.9.0, which appends the same-estimator pre/post error triples (cols
  27-29 pre pos/ome/angle, 30-32 post) while leaving cols 22-24 untouched.
  Width is **sniffed**, not assumed: `Key.bin`'s seed count is authoritative
  when present, arithmetic decides when only one width divides the file, and
  a SpId-sentinel content check breaks the genuine tie (any multiple of 297
  doubles divides by both 27 and 33 — 11x27 == 9x33). A file matching
  neither is rejected rather than guessed at, because a wrong width shifts
  every column with no exception anywhere.
  `ORIENT_POS_FIT_LAYOUT` gains the six names; asking for them on a 27-wide
  file raises `IndexError`, which is the intent.
  - **Cols 22-24 are documented as the historical MIXTURE they are** (22
    post-fit, 23/24 pre-fit). Prefer 27-32 for any before/after comparison.
- **`read_fit_best_final`** for `Output/FitBestFinal.bin` — the post-fit
  per-spot records, same layout/stride/tail behaviour as `FitBest.bin`.
  The two are **not row-aligned** (the post-fit matcher can select a
  different spot set), so join on SpotID, never by row index.
- **`io/spot_diag.py`** — canonical reader for `SpotDiagnostics.bin`, the only
  artifact recording expected spots that were **not found**. Version-aware
  (v1 vs v2, see below), with `unmatched()`, `completeness()` and
  `completeness_by_ring()`. On the reference FF layer: 55,593 voxels,
  6,384,978 predicted spots, **654,736 un-found**, mean completeness 0.8975,
  and per-ring completeness 1:0.972 2:0.927 3:0.905 4:0.901 **5:0.779** —
  the outer ring is where spots are lost, which nothing previously exposed.
  - v1 files carry a defect: col 5 is `theorGx` on matched rows but a real
    theoretical-spot id on unmatched ones (measured col5==col6 on
    41,118/41,118 matched rows). `SpotDiag.col5_is_theor_spot_id` reports
    whether it is trustworthy; `summary()` warns on v1.

### Changed

- 23 new tests (`test_spot_diag.py`, `test_orient_pos_fit_width.py`); suite
  364 -> **388**.

## [0.9.3] – 2026-08-21

### Fixed

- **The LAST seed was silently deleted from every c-omp run.**
  `FitUnified.c` pwrites only `nSpotsComp` records per seed at a full-slot
  stride (`:2297` FitBest.bin, `:2136` ProcessKey.bin), so both files end
  mid-slot. `read_fit_best` and `read_process_key` floor-divided by the
  stride and truncated the remainder — dropping the final seed, silently, on
  every run. Their docstrings described this as tolerated slack, and the
  `c_parity_run` truncation guard rationalised it as *"the dropped trailing
  seed gets NrIDsPerID=0 anyway and contributes nothing"*, which was false.
  - Measured on a fresh 56,125-seed Ni FF layer: FitBest.bin = 56,124 full
    slots **+ 87 rows**, ProcessKey.bin = 56,124 **+ 87 ints** — the same
    seed 56,124, `nSpotsComp = 87`. That seed is **alive**: SpotID 245283,
    `keep_flag` set, completeness 0.777, meanRadius 27.3 µm, while
    `OrientPosFit.bin` and `Key.bin` both saw it at 56,125 rows.
  - The short final slot is now zero-padded and returned
    (`io.binary.TailPaddedBinary`), matching what
    `midas_fit_grain.io_binary.read_fit_best` has always done for its
    materialising path. All four readers now agree at 56,125 seeds and the
    `c_parity_run` "truncating to common length" branch no longer fires.
  - After the fix that seed is a grain: `Grains.csv` ID 245283, DiffPos
    404.402644, radius 27.346476, confidence 0.776786 — each matching
    `OrientPosFit` cols 22/25/26 — with **87** SpotMatrix rows = its
    `nSpotsComp`, and it obeys the per-spot/per-grain identities to 1.4e-16.
  - Full slots stay zero-copy memmap views; only the final short slot is
    materialised (≤880 KB FitBest, ≤20 KB ProcessKey). The view **refuses**
    whole-file `np.asarray` rather than silently truncating or copying 49 GB
    — use `io.binary.materialize()` where a full load is genuinely intended
    (ProcessKey clustering is; FitBest never is). That guard immediately
    caught one such call site in `c_parity_run`.
  - Integrity check tightened: a trailing remainder that is not a whole
    number of records is now rejected as torn. The previous
    `size > n_full*slot + slot` check was unreachable — `divmod` guarantees
    the remainder is below one slot.
  - 12 new tests (`tests/test_tail_padded_binary.py`); suite 355 → 367. No
    existing fixture could have caught this: `conftest.tiny_run_dir` writes
    "the full MAX_N_HKLS=5000 slots zero-padded", i.e. always an exact
    multiple.

## [0.9.2] – 2026-08-21

### Added

- **`c_parity` now writes the residual sidecar.** The signed per-spot residual
  decomposition added in 0.6.0 was only ever wired into `pipeline.run`, so the
  *default* mode — the one `midas-pipeline` and `midas-ff-pipeline` run —
  returned from `run_c_parity_pipeline_from_disk` without producing
  `processgrains_diagnostics.h5` at all. Nothing errored; `Grains.csv` was
  correct and every downstream diagnostic silently degraded to descriptive-only
  (documented as a standing limitation in `manuals/ff-hedm/phase-3-run.md`).
  `c_parity` now emits the same `/residuals` schema as the spot-aware and
  `paper_claim` paths.
  - Costs **no extra FitBest I/O**: `gather_per_grain_spot_data` already holds
    the full `(5000, 22)` seed block for the Kenesei strain solve, and the
    decomposition reads cols 1,2,3 / 7,8,9 / 19 of the same array. Measured on
    the from-scratch Ni FF layer (24,090 grains, 49.4 GB FitBest): gather
    38.1 s, the memmap read still dominating.
  - `compute/residual_decomposition.build_spot_residual_block` is the new
    vectorised core; `build_spot_residual_row` is now a wrapper over it, so the
    `c_parity` and spot-aware callers cannot drift (guarded by a
    scalar-vs-block equality test).
  - `io/consolidated.write_diagnostics_arrays` is the one implementation of the
    sidecar schema; `write_diagnostics_h5` is a thin adapter for callers that
    have a `ProcessGrainsResult`.
  - `c_parity` writes `/diagnostics/cluster_sizes` (seeds merged per grain) and
    **omits** `n_resolved_hkls` / `n_majority_hkls` / `n_residual_tie_hkls` /
    `n_forward_sim_hkls` rather than zero-filling them — those describe per-hkl
    conflict resolution `c_parity` does not perform, and a zero is
    indistinguishable from a measurement.
  - `--no-diagnostics-h5` is now honoured by `c_parity` (it was accepted and
    ignored). `mode="physics"` still has no residuals — `v4_pipeline` never
    reads FitBest, so there is no obs-vs-predicted table to decompose.
  - Validated on a **from-scratch** Ni FF run (`ff_refiner_prepost`,
    2026-08-21: raw .cbf → zip_convert → peakfit → transforms → index →
    refine, current binaries), 2,390,948 spot residuals over 24,090 grains.
    Per-grain means reproduce `Grains.csv` `DiffOme`/`DiffAngle` to **1.4e-15
    / 1.2e-15** over all 55,593 written seeds, 0 exceptions; the
    radial/tangential rotation is orthonormal to 3.5e-7.
    `utils/midas_ff_report.py` renders its full figure set (including
    `residuals.png`) on a default run. 7 new tests
    (`tests/test_c_parity_residuals.py`).
  - **Earlier numbers in this entry were withdrawn.** They were measured on
    a `datasetA` recon (`.../nb_ni_recon/LayerNr_1`), which is a *mixture* of runs:
    `GrainRadius == 1.0000` on all 24,318 grains (the signature of a build
    predating the `meanRadius` fix), a `paramstest.txt` whose `OutputFolder`
    does not match where its own `FitBest.bin` sits, and May file dates that
    predate commit `06dd3241` (2026-08-07) — the very refiner staging the
    numbers described. Do not re-cite anything measured there.

### Fixed

- **`result.write(diagnostics_h5=True)` no longer blanks a populated sidecar.**
  A `c_parity` result is rebuilt from `Grains.csv` and so carries no
  `diagnostics`; writing it over the sidecar c_parity had just written would
  have replaced 2.4 M rows of residuals with zero-length arrays, and the run
  would still have reported success.

### Known upstream behaviour (not introduced here, not fixed here) — ESTABLISHED

- **The per-spot residuals in `FitBest.bin` are evaluated at the INDEXER SEED,
  before any fitting**, as are `Grains.csv` `DiffOme` and `DiffAngle`. Only
  `DiffPos` is post-fit. So the `/residuals` sidecar describes the seed
  geometry on c-omp runs, and `DiffPos` cannot be re-derived from it.
  - Mechanism: `SpotsComp` (→ FitBest.bin) is filled only inside
    `CalcAngleErrors`, whose unconditional call sites (`FitUnified.c:1804`,
    `:1828`) both pass `Ini = ConcatPosEulLatc(Pos0, Euler0, LatCin)` = the
    seed. The post-fit re-match at `:1939` is inside the `MIDAS_FG_REMATCH`
    macro, gated on `getenv(...)` and off by default. `ErrorFin[0]`
    (`:2036-2040`) is `FitErrors12D(FinalResult)/nSpotsComp`, whereas
    `ErrorFin[1]/[2]` are carried over from the pass-2 seed evaluation.
  - Measured over **all 55,593** written seeds of the from-scratch run:
    mean(per-spot ω residual)/`DiffOme` and mean(per-spot IA)/`DiffAngle` are
    both exactly 1 (~6 ulp); mean(per-spot `DiffLen`)/`DiffPos` = **1.7111**
    (p05 1.315, p95 2.213); mean(`DiffLen`)/`IniErr` = 1.0000 and
    `DiffPos`/`FinalErr` = 1.0000, both to the log's print precision.
    `FinalErr > IniErr` in **0/55,593** grains — the 1.71× *is* the
    improvement refinement achieves.
  - Convention-free confirmation: `hypot(YExp, ZExp)` is the theoretical ring
    radius, a pure function of the lattice used to build the theoretical
    spots. Across 257,700 spots from 2,500 grains whose *refined* `a` spans
    4.3e-3 relative, its per-ring spread is **3e-16** (3 distinct floats per
    ring) where refined-`a` spread would move ring 1 by ~416 µm; and those
    radii match `hkls.csv`, generated at the seed `LatticeParameter`, to
    <5e-07 µm.
  - **ESTABLISHED** — mechanism lens survived a dedicated refutation attempt
    (which also killed the competing `FitErrors12D` id-pairing explanation:
    worth 0.0%), and an independent reproduction lens reproduced every
    number from raw files with 0 exceptions. Residual gap: the env-var state
    is inferred from the absence of its output lines, not directly asserted.

## [0.6.0] – 2026-07-16

### Added

- **Signed per-spot residual decomposition** (`compute/residual_decomposition.py`).
  Per-spot signed dY, dZ, radial, tangential, wrapped dOme + internal angle,
  collected inside the existing FitBest pass in
  `pipeline.py::_build_spot_matrix_rows` (now returns a tuple; single caller
  updated). Aggregates — per-grain median/MAD, per-ring dR/R ppm, 30° eta
  bins, global scalars — plus a gzip float32 per-spot table are written to
  `processgrains_diagnostics.h5:/residuals` (`io/consolidated.py`). The run
  log warns when |median dR/R| > 200 ppm on a ring — the signature of a
  wrong reference lattice (a₀) being absorbed as fake hydrostatic strain.
  `Grains.csv` `DiffPos`/`DiffOme` are now decomposable from this table.
  `mode="legacy"` (no FitBest) emits empty residuals by design.
  Validated: 7 new tests (`tests/test_residual_decomposition.py`), full suite
  290 pass; production run on datasetB recon_3580_003 (1.66 M-row spot table)
  — diagnosed the −850 ppm reference-lattice offset that recalibration then
  removed (mean hydrostatic +850.5 → +7.6 µε).

## [0.4.6] – 2026-05-26

### Fixed

- **v4 pipeline: Stage 6 NNLS got placeholder radii on c-omp refiner outputs.**
  The c-omp `midas_fitgrain` (FitUnified.c) writes `meanRadius=1.0` to
  OrientPosFit col 25 as a deliberate placeholder — PF mode needs col 25 = 1,
  and `midas_process_grains.pipeline` already compensates when it emits
  `Grains.csv` (averages per-spot `GrainRadius` from `Radius_StartNr_*.csv`
  over each grain's matched spots; see pipeline.py:595-605). `v4_pipeline`
  was reading OPF col 25 directly into `rep_radius_naive` and propagating
  the 1.0 placeholder through Stage 6 sizing → Stage 8.5/8.5b/8.5c
  (volume-budget drop / force-keep distinct / orphan reclaim).
  Effect on a fresh c-omp recon (datasetA Ni nb_ni_recon): median R = 1.0
  µm, ΣV = 1.03 × 10⁵ µm³, packing = 0.01 %, zero drops engaged.
  **Fix:** mirror pipeline.py's recovery — detect when OPF col 25 is all
  1.0, then recompute `rep_radius_naive[i]` as the mean of per-spot
  `GrainRadius` from `InputAllExtraInfoFittingAll.csv` over each
  candidate's matched-spot set from `Results/ProcessKey.bin`. Vsample
  correction is then applied to the recomputed values. ~3 s overhead on
  56 k candidates.
  Validated on datasetA Ni c-omp recon — now matches legacy_fresh: median
  R = 27.71 µm, ΣV = 4.25 × 10⁹ µm³, packing 425 %, 17,723 drops engaged.

## [0.4.5] – 2026-05-26

### Changed

- **Stage 8.5c orphan reclaim now requires fractional uniqueness.** Default
  `orphan_reclaim_min_unique_fraction=0.5` — a reclaim candidate must have
  ≥50 % of its spots be NEW (unique to the orphan set). The earlier v0.4.4
  default (`min_unique_spots=5` only) admitted grains that contributed 5
  unique spots while ALSO re-claiming 30+ redundant ones, ballooning per-spot
  multiplicity (median 3, mean 4) and ΣV (packing 328 %). The new fractional
  guard restores intensity-conservation: on datasetA Ni, packing
  328 % → 127 %, kept count 16,846 → ~5,600, orphan rate 1.4 % → 32 %,
  median multiplicity 3 → 2. Set to 0.0 to restore v0.4.4 behaviour.

- **New kwarg** `orphan_reclaim_min_unique_fraction` on `run_v4_pipeline`.

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

- **datasetA result moves from 4,452 → ~5,745 → ~6,000+ kept grains**
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
  datasetA 54 µm, datasetF 37 µm, peakfit 35 µm median); datasetF and
  peakfit agree to within 2 µm on the SAME sample.
- Cross-pipeline reproducibility on heavily-twinned LMO: identified
  fundamental multi-modal-refiner limit (0.01% match at
  refiner-noise-scale; 99% match at Σ-twin scale). Documented as
  characteristic of heavily-twinned samples, not a v4 bug.
- 6 synthetic planted-twin integration tests (HCP, tetragonal,
  multi-phase, user-supplied orthorhombic) all recover ≥95% of
  planted pairs.
- Twin geometry against theoretical Σ-misori: ≤ 0.14° median |Δ|
  across datasetA, datasetF, peakfit; 0% > 2° off.

## [0.2.1] – 2026-04 (previous)

See git log for changes prior to the v4 series.
