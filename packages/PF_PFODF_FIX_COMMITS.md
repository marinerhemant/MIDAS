# Commit drafts — PF_PFODF_FIX_REQUIREMENTS campaign (2026-07-16)

Nine commits, per-package (items sharing files cannot be split further).
Suggested order below respects dependencies (transforms first — pipeline
tests import it). All suites FULLY green at HEAD of this series:
transforms 192✓ (all 9 C-regression tests incl. the two previously
failing golden tests — fixed, see NOTES), pipeline 300✓, ff_pipeline 55✓,
process_grains 292✓, fit_grain 69✓, peakfit 63✓, zipper 10✓,
pf_odf 48✓, grain_odf 28✓.

---

## Commit 1 — midas-transforms 0.8.0

Files: `packages/midas_transforms/**` (incl. new tests
`test_bin_data_chunk_parity.py`, `test_paramstest_new_keys.py`,
`test_returncode_origspotid_flow.py`, `test_fit_setup_intensity_filter.py`)

### Detailed
```
midas-transforms 0.8.0: paramstest full geometry, InputAllExtra schema, chunked binning

P0-1  paramstest.txt now carries the FULL raw-frame geometry: txFit
      (was never emitted -> raw-frame consumers got tx=0, ~0.27 deg
      in-plane rotation = 3-4e3 ue fake strain), all p0..p14 (C wrote
      p0..p3), canonical v2 distortion names for calibrate-v2-native
      archives, and OmegaStart/OmegaStep (written only when the step is
      known; consumers must never infer it from shadow-gapped
      OmegaRange spans). Refined DoFit=1 values now actually reach the
      *Fit keys (previously only pt.Lsd was set and the writer
      preferred the raw LsdFit). Reader parses everything back,
      including v2-name reconstruction into dist_coeffs_v2.
      Smoke-verified: IndexerOMP + FitPosOrStrainsOMP behave
      IDENTICALLY on legacy vs new-keys files (gated test).

N1/E2 InputAllExtra header now matches the data it labels: cols 13-17
      were one slot left of reality (the "IntInt" reads that cost the
      Ni campaign half a day were omega angles); cols 11/12 renamed
      YRawPx/ZRawPx (they hold raw detector pixels == peaksearch
      YCen/ZCen, not lab um). Rename-only: every in-tree consumer is
      positional; binary strides untouched.

N2+E3 ONE append-only schema change: peakfit's per-peak returnCode now
      survives merge (Result col 17, sticky-first-nonzero over merged
      constituents) -> calc_radius (col 25) -> InputAllExtra col 19;
      and OrigSpotID (merge-space SpotID) is carried through the TWO
      renumbering stages (calc_radius renumbers 1..N and duplicates
      two-ring spots; fit_setup re-sorts + renumbers again) into
      Radius col 24 and InputAllExtra col 18. A SpotID join across
      these spaces silently paired random spots and invalidated two
      emerson analyses. Readers accept legacy widths (pad -1 =
      unknown); the base reader returns the 18 legacy cols so the
      ExtraInfo.bin 16-double stride can never widen; C-golden parity
      tests scoped to the legacy 17 Result cols.

N3    Binning pair expansion is spot-chunked (MIDAS_BIN_PAIR_CHUNK,
      default 2^28 pairs) with memory hygiene and per-ring pack+concat
      in the scanning writer - the all-at-once path OOM-killed at
      250 GB / 60 GiB CUDA on the 135M-spot Ni Layer-3. Bit-identical
      by construction; new FF/PF chunk-parity fixtures prove
      byte-equality (forced 1-pair budget vs unchunked) on all output
      files.

N8    MinIntegratedIntensity fit_setup spot filter (default 0 = off, no
      FF behaviour change), recorded in paramstest.txt so reruns see it
      - replaces the Ni campaign's unrecorded awk-filtering of layer
      CSVs.

Also: the two long-failing test_regression_vs_c bin_data golden tests
are FIXED - they were format-stale since the unified-container change
(f962e7b2: 10-col Spots.bin, uint64-pair Data.bin always), not
value-stale. They now compare value-exactly across the container
change (Spots cols 0-8 byte-level + scanNr==0; Data rowno stream +
nData count/offset tables == C golden + scanno==0) - which also
independently validates the N3 chunked expansion against real C
output on a 27k-spot layer.

Validation: 192 tests pass (all 9 C-regression tests green).
```

### Short
```
transforms 0.8.0: full paramstest geom, schema append, chunked binning
```

---

## Commit 2 — midas-pipeline 0.6.0 (after Commit 1)

Files: `packages/midas_pipeline/**` (incl. new tests
`test_positions_materialization.py`, `test_only_dependency_guard.py`,
`test_scan_fanout.py`, `test_transforms_layer_extra.py`,
`test_workdir_and_binning_device.py`)

### Detailed
```
midas-pipeline 0.6.0: PF silent-corruption fixes + per-scan fan-out

P0-2  positions.csv is materialized at layer setup from ScanGeometry
      (file order = acquisition order, never overwriting a pre-seeded
      file, + root copy) and a missing file is now a HARD error in PF
      mode - previously every early stage soft-skipped and the run
      exited 0 having done nothing (hit independently by the SOH and
      Ni campaigns). _pf_scans no longer re-sorts positions ascending
      (that silently reversed the scan<->Y pairing for descending
      acquisitions); convention unified with stages/indexing.py:
      file order everywhere, the indexer sorts its own grid.

E2(d) The PF per-scan y-offset looked up the legacy-C column name
      "YOrig(NoWedgeCorr)" - which no header variant ever contained -
      so the second lab-frame shift was a silent no-op. Now shifts
      YLab + YOrigDetCor (present at col 9 in old AND new headers),
      never YRawPx, and fails loud on an alien header.

N4    --only allowlists are validated against a per-mode stage
      dependency graph (deps satisfied by selection, provenance-
      complete records, or --skip); an unmet dependency is a hard
      error. Both campaigns got broken recons from the same handoff
      doc's incomplete allowlist, each omitted stage soft-skipping.

N5/N6 peakfit + transforms + zip_convert fan out per scan
      (--scan-workers / --zip-workers, default 1 = serial legacy).
      Every scan is claimed atomically (midas_log/claims/, stale
      local claims broken by dead-pid check) so two independent
      runners cooperate instead of racing; peakfit round-robins CUDA
      devices and splits --n-cpus-local between workers.

N7    --binning-device overrides --device for binning only (its pair
      expansion OOMs on GPU first at dense-PF scale).

N9    zip_convert fails hard when ALL scans fail (a broken env fails
      every scan identically; the run used to "succeed" on WARNINGs).

N11   --scan-work-dir separates writable per-scan work dirs from a
      read-only RawFolder (collaborator data); pre-built raw zips are
      still honoured. Replaces the symlink-farm workaround.

N12   merge_overlaps stub docs corrected: the cross-frame merge is NOT
      "pending" - it is the byte-exact C-parity port
      midas_transforms.merge_overlapping_peaks, executed inside the
      transforms stage. The stale docstring mis-led two campaigns.

Also: merge_scans _MERGED_HEADER fixed (same N1-class mislabeling) and
per-scan CSV readers accept the 20/21-col appended schema.

Validation: 300 tests pass (24 new).
```

### Short
```
pipeline 0.6.0: PF hard-errors, positions materialize, scan fan-out
```

---

## Commit 3 — midas-process-grains 0.6.0

Files: `packages/midas_process_grains/**` (incl. new
`compute/residual_decomposition.py`, tests
`test_residual_decomposition.py`, `test_rms_error_strain.py`,
`test_d0_advisory.py`)

### Detailed
```
midas-process-grains 0.6.0: residual decomposition, real RMSErrorStrain, d0 advisory

E0  Signed per-spot residual decomposition (dY/dZ/radial/tangential/
    wrapped dOme + internal angle) collected inside the existing
    FitBest pass; per-grain median/MAD, per-ring dR/R ppm, 30-deg eta
    bins, global scalars + gzip float32 spot table written to
    processgrains_diagnostics.h5:/residuals. Grains.csv DiffPos/
    DiffOme are now decomposable. Ran in production on emerson
    recon_3580_003 (1.66M rows) - diagnosed the -850 ppm reference-
    lattice offset that recalibration then removed (+850.5 -> +7.6 ue
    mean hydrostatic). legacy mode emits empty residuals by design.

E1  Grains.csv RMSErrorStrain is the real per-grain strain-solver RMS
    residual (ue) instead of a hardwired 0 users read as "perfect
    strain fit". Verified on a strained synthetic: planted +1000 ue ->
    recovered 923 ue diagonal, RMSErrorStrain 54 ue.

E7  d0 ADVISORY: when the per-ring |median dR/R| > 200 ppm flag trips
    on a cubic reference, print the free-standing-recovered a0
    (midas_stress recover_d0_cubic_free_standing, volume+confidence
    weighted) and the exact LatticeConstant line to paste. Never
    auto-applied - loaded samples need recover_d0 + stiffness +
    applied stress. E2E test reproduces the emerson numbers (planted
    -850 ppm -> recovered a0 3.5969).

Validation: 292 tests pass. Deploy note: copland s20iduser has no
PyPI - install with pip --no-build-isolation --no-deps.
```

### Short
```
process-grains 0.6.0: /residuals diag, real RMSErrorStrain, d0 advisory
```

---

## Commit 4 — midas-fit-grain 0.5.4

Files: `packages/midas_fit_grain/**`

### Detailed
```
midas-fit-grain 0.5.4: absorb new paramstest keys (E6)

Extend the unknown-key ignore set with everything midas-transforms
>= 0.8.0 writes (txFit, p6..p14, v2 distortion names, OmegaStart/
OmegaStep, LsdFit-family, WeightMask/WeightFitRMSE, RingToIndex,
NoSaveAll) plus MinIntegratedIntensity, so downstream parsing stops
warning on the completed geometry. No behavioural change.
```

### Short
```
fit-grain 0.5.4: known-keys list for new paramstest keys (E6)
```

---

## Commit 5 — midas-peakfit 0.4.3

Files: `packages/midas_peakfit/**`

### Detailed
```
midas-peakfit 0.4.3: cap worker BLAS/OpenMP threads (N10)

With num_procs > 1 the orchestrator now setdefaults OMP/OPENBLAS/MKL/
NUMEXPR/VECLIB thread counts to 1 before creating its pool - without
it, workers x default-all-cores oversubscribed the box (Ni Layer-3:
load 28/64, frame rate collapsing 22 -> 4-5 f/s until the caller
exported OMP_NUM_THREADS=1 by hand). Explicit user settings always win
(setdefault).
```

### Short
```
peakfit 0.4.3: orchestrator caps worker OMP/BLAS threads (N10)
```

---

## Commit 6 — midas-zipper 0.1.4

Files: `packages/midas_zipper/**`

### Detailed
```
midas-zipper 0.1.4: tqdm dependency + honest omegaStep (N9, E5)

N9  tqdm added to install deps - a fresh env made ff_zip.py exit 1 on
    every scan (import at :488) while zip_convert marched on and the
    run "succeeded" with 0 zips built.

E5  The '-omegaStep' CLI default 0.0 no longer lands verbatim in the
    zarr as analysis_parameters/omegaStep (false scan metadata - the
    emerson archive said 0.0 with an actual 0.25). An explicit CLI
    value is promoted onto the canonical OmegaStep key; the false 0.0
    is dropped entirely so metadata readers never see a fabricated
    step.
```

### Short
```
zipper 0.1.4: add tqdm dep; never write false omegaStep 0.0
```

---

## Commit 7 — midas-pf-odf 0.1.0 (LOCAL-ONLY package — currently untracked;
commit applies if/when the package joins the repo)

### Detailed
```
midas-pf-odf 0.1.0: raw-frame correctness + productized mini-calibration

P0-3 geometry_from_paramstest resolution order: caller kwargs >
     explicit OmegaStart/OmegaStep keys > single-OmegaRange span;
     multiple OmegaRange lines with no explicit step now ERROR instead
     of inferring 74/1440 = 0.0514 deg from the first shadow-gapped
     span (every SOH forward anchor landed on empty frames). Coverage
     always spans ALL ranges; a false "OmegaStep 0.0" key is ignored.

P1-4 Detector distortion plumbed into the forward
     (apply_distortion=True, distortion_from_paramstest): paramstest
     p0..p14 are v1-ordered - p3 is phi4, a PHASE in degrees - mapped
     via midas_distortion.v2_coeffs_from_named; v2 names take
     precedence. Measured effect on SOH: 4-166 um ring shifts,
     ~100 ue on Stage-1 strain.

P1-5 ring_numbers override on build_model_from_paramstest (indexing
     rings != optimal strain rings; SOH indexing used 3,4 while rings
     1,2 are 3-4x brighter and fully on-detector).

P1-6 calibrate_raw_frame_geometry + layer_model_factory: the
     empirical-Jacobian damped Gauss-Newton mini-calibration
     (bump y_BC/z_BC/tx/ty/tz/Lsd/omega0 in the ACTUAL model ->
     convention-proof derivatives -> MAD-robust damped LSQ), which
     collapsed SOH residuals 7.06 -> 1.4 px RMS. Synthetic recovery
     test: planted BC shift recovered to 0.5 px with RMS collapse.

P2-7 Hard per-pixel saturation mask (design: mask-only; threshold from
     UpperBoundThreshold via saturation_threshold_from_paramstest or
     explicit kwarg) through assemble_grain_patch_data -> the data MSE
     AND the closed-form per-spot amplitude. SOH rings 1/2 are 96-98%
     saturated; fitting Gaussian splats against flat-tops floored the
     loss and ran the strain away (4237 -> 5959 ue with more steps).

P2-8 Sparse-grain smoothness: neighbor_edges_from_grid_ij builds the
     4-neighbour edge list for real (sparse) grains; the dense
     grid_shape path now raises an actionable error instead of
     "shape [41,41,3] invalid for size 3936".

P2-9 spread_init accepts per-voxel (G,) warm-starts (pooled to
     per-region means via the region map) - the Stage-2 -> Stage-3
     hand-off crashed on reshape before.

Validation: 48 tests pass (17 new).
```

### Short
```
pf-odf 0.1.0: omega guard, distortion, mini-calib, saturation mask
```

---

## Commit 8 — midas-grain-odf 0.2.0 (LOCAL-ONLY package — currently untracked)

### Detailed
```
midas-grain-odf 0.2.0: spread-DOF guardrails + robust spread stats (E4)

(a) A spread DOF (sigma_theta / sigma_eps) finishing PINNED at its
    physical ceiling now warns loudly and sets
    {strain,orientation}_spread_pinned on the result - the emerson
    failure: at synthetic-tuned default LRs both DOFs shot to their
    clamps and the correlation loss WORSENED (-25.4 -> -4.0), i.e. the
    fit was invalid but looked converged.

(b) lr_strain_spread / lr_orientation_spread accept "auto": the SGD lr
    is scaled from the first gradient so the first step moves the
    parameter by ~20% of its own scale (init, floored at 0.5% of the
    ceiling). Recovers a planted sigma_eps without hand-tuning where
    the previous ceiling-relative target overshot into the zero-clamp.

(c) particle_spread_stats / GrainFitResult.spread_stats report the
    weighted-MEDIAN spread + weight-within-X-deg alongside wRMS - wRMS
    scales with theta_max (prior-dominated tails: 1.75x at 2x trust
    region vs 1.24x for the median), so cross-setting comparisons must
    use the median.

Validation: 28 tests pass (3 new) + 1 expected xfail.
```

### Short
```
grain-odf 0.2.0: spread clamp-pin flags, auto LR, robust spread stats
```

---

## Commit 9 — docs (manuals + investigation + requirements inventory)

Files: `manuals/FF_Analysis.md`, `manuals/FF_Parameters_Reference.md`,
`manuals/PF_Analysis.md`, `packages/P2_10_COMPLETENESS_INVESTIGATION.md`,
`packages/PF_PFODF_FIX_REQUIREMENTS.md`, `implementation_plan.md`,
`packages/PF_PFODF_FIX_COMMITS.md`
(+ `packages/midas_pf_odf/dev/PF_HEDM_NEW_DATASET_HANDOFF.md` if tracked)

### Detailed
```
docs: PF/FF manuals for the fix campaign; P2-10 investigation

- FF_Analysis: correct the stale /all_spots column attrs (cols 11/12
  are raw detector px; ExtraInfo.bin uses a DIFFERENT order); document
  the three SpotID spaces + the OrigSpotID/ReturnCode bridges; add the
  verified raw-frame conventions (frame layout, um->px map, raw vs
  DetCor tens-of-px under large distortion); document
  processgrains_diagnostics.h5:/residuals.
- FF_Parameters_Reference: new paramstest geometry keys (txFit,
  p0..p14 v1-order warning, v2 names, OmegaStart/OmegaStep,
  MinIntegratedIntensity).
- PF_Analysis: new operability guarantees + knobs (positions
  materialization + hard error, --only dependency guard,
  --scan-workers/--zip-workers/--binning-device/--scan-work-dir,
  MinIntegratedIntensity, zip_convert all-fail hard error).
- P2_10_COMPLETENESS_INVESTIGATION.md: code-trace of the 0.5x-legacy
  completeness convention (python denominator lacks C's OmegaRanges/
  BoxSizes filters; legacy 0.81 provenance suspect) + the decisive
  copland experiment; interim guidance: gate at 0.2.
- pf-odf handoff doc: all 8 corrections applied (positions.csv,
  refine-backend python, --skip not --only, host-specific activate,
  0.40x zip ratio, pgrep -u scoping, legacy-h5-not-input,
  completeness note) + notes on which sections the new pipeline
  version supersedes.
```

### Short
```
docs: fix campaign manuals, ID spaces, P2-10 completeness trace
```

---

## NOTES for the committer

- The two long-failing bin_data golden tests were FORMAT-stale, not
  value-stale: since f962e7b2 FF bin_data always writes the unified
  container (10-col Spots.bin, uint64 (rowno, scanno) Data.bin pairs)
  while the tests still read the legacy layouts (27135×10 doubles / 9
  = the misleading 30150-row "count"). Fixed in Commit 1 by comparing
  values across the container change; the C goldens themselves are
  fine and confirm full FF value-parity.
- midas_pipeline test_recon_fbp SIGABRT: observed exactly once while a
  parallel `cmake -j8` build was saturating the machine; 9/9 green on
  re-runs (MIDAS_TOMO is an OpenMP+FFTW binary — transient resource
  exhaustion). Environmental flake, no code change.
- Deployment reconcile after committing (per PF_PFODF_FIX_REQUIREMENTS
  deployment matrix): copland s20iduser midas_env still runs
  transforms 0.7.2 (old mislabeled header) — redeploy 0.8.0 with
  `pip install --no-build-isolation --no-deps` (no PyPI on that
  account); toro/s20hedm workspace carries the pre-commit patched
  transforms — align to the committed 0.8.0; alleppey s1iduser
  PYTHONPATH overlay (~s1iduser/emerson_odf/pkgs) — refresh the
  overlaid packages.
- The live Ni run on toro already executes the N1/N3 code this series
  commits; the additional schema columns (N2/E3) change per-scan CSVs
  going forward — safe (append-only, positional consumers) but
  regenerate nothing mid-run.
```
