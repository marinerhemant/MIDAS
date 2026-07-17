# Implementation plan — PF_PFODF_FIX_REQUIREMENTS.md, all items

Source: `packages/PF_PFODF_FIX_REQUIREMENTS.md` (3 campaigns, 2026-07-14→16).
Goal: land **every** item without breaking FF or PF. Strategy: 4 phases,
each phase = independently committable, full test suite green per touched
package, FF blast-radius table consulted per item. Rules enforced throughout:

- **Rename-only header fixes OK; never reorder/insert columns** (consumers positional, C mmaps fixed strides).
- New columns **append-at-END only**, `INPUTALL_EXTRA_NCOLS` + all readers bumped in the same commit, `Spots.bin`/`ExtraInfo.bin` strides untouched.
- New paramstest keys → E6 known-keys lists updated in the same commit.
- Defaults preserve FF behaviour exactly (new knobs off/inherit).
- Every commit: package tests green; two commit-message drafts; **user runs git commit**.

---

## Phase 0 — Lock in the APPLIED-uncommitted work (stops repo/deployment divergence)

**Commit A — midas-transforms: N1 complete + E2 + N3, version bump**
1. **N3 FF parity fixture (gate, do first):** new test in
   `midas_transforms/tests/` proving Data.bin/nData.bin **byte-equality** on an
   FF-shaped layer (scan_nr=0): forced multi-chunk vs single-chunk vs golden
   fixture (extend the existing SaveBinDataScanning C-fixture pattern to FF
   mode). N3 code is already applied; this test is the commit gate.
2. **E2(a):** rename InputAllExtra header cols 11/12 → `YRawPx ZRawPx`.
3. **E2(b):** reconcile 18-name repo header vs 19-col deployed data (`DetID`)
   against ground truth `fit_setup/core.py:396-410`; header must match data
   exactly (19 names if fit_setup writes DetID).
4. **E2(d):** fix `midas_pipeline/stages/transforms.py:204-205`
   `"YOrig(NoWedgeCorr)"` lookup → real column name + **fail loud** on missing
   column (currently a silent no-op on the PF y-offset). Same commit as the
   rename. (midas_pipeline may need its own commit if versioned separately —
   keep the rename+lookup change atomic in one push.)
5. Bump midas-transforms version.

**Commit B — midas-process-grains 0.6.0 (E0), already tested (290 pass)**
- Commit as-is + update `manuals/` (`/residuals` schema = `SPOT_RESIDUAL_COLS`,
  DiffPos/DiffOme decomposable, legacy mode emits empty residuals by design).

*Gate: transforms + process_grains + pipeline test suites green; FF parity
fixture green on CPU (and MPS single-chunk).*

## Phase 1 — P0 correctness (silent-corruption class)

**Commit C — P0-1 + E6: paramstest writer emits txFit, p0..p14, OmegaStart/OmegaStep**
- `midas_transforms/params.py:326-331,566-580`: emit `txFit`, all distortion
  coeffs, `OmegaStart`/`OmegaStep`.
- E6: extend known-keys lists (`midas_fit_grain/config.py:317-327`, analogous
  in process_grains) so nothing warns.
- Smoke test: FitUnified + midas_indexer parse a paramstest carrying all new
  keys (doc explicitly requests this C-parser smoke — will run the binaries).
- Test: writer round-trip asserts tx + p0..p14 + omega keys present.

**Commit D — P0-2 + corroboration: positions.csv materialization, PF-scoped hard error**
- Materialize `positions.csv` at layer setup (`midas_pipeline/pipeline.py:196-199`)
  from `ScanGeometry.pf_uniform` (file order = acquisition order, sign per
  `--scan-step`).
- Missing positions = **hard error only when `scan_mode=="pf"`** (FF guard —
  FF writes its own 1-row file late).
- Unify/document the sort convention (`_pf_scans._positions_for_layer` sorts;
  `stages/indexing.py:147` reads file-order) per the acquisition-order-for-
  filter / sorted-for-grid convention.

**Commit E — P0-3: omega-step inference guard (midas_pf_odf)**
- `io.py:159-222`: prefer explicit OmegaStart/OmegaStep keys; with multiple
  `OmegaRange` lines and no explicit key → error loudly, never infer from spans.
- Test: 3-range shadow-gapped paramstest → raises; explicit keys → 0.25.

**Commit F — N2 + E3 as ONE append-only schema change**
- Append `returnCode` (from peakfit col 18) and `OrigSpotID` (merge-space ID,
  mapping exists at fit_setup write time) at END of InputAll/InputAllExtra.
- Bump `INPUTALL_EXTRA_NCOLS`; update every reader with a col-count assert in
  the same commit; binaries untouched. Loud docstrings on both writers about
  the two SpotID spaces. `FitErrCode` header name becomes honest.
- Tests: schema round-trip; OrigSpotID join vs merge output on a fixture.

**Commit G — E1: RMSErrorStrain real value**
- `midas_process_grains/pipeline.py:793-810`: capture
  `PerSpotStrainResult.residual_norm` per grain (µε, legacy C convention).
- Test: non-zero on strained synthetic.

**Commit H — N4 + N9: fail-loud operability**
- N4: per-mode stage-dependency check in `midas_pipeline/pipeline.py:239-260`
  — error/loud-warn when `--only` omits required stages (per-mode graph, not a
  shared allowlist).
- N9: add `tqdm` to midas_zipper deps; zip_convert fails hard when ALL scans fail.

*Gate: full test suites of transforms, pipeline, pf_odf, process_grains,
zipper, fit_grain green; C smoke test green.*

## Phase 2 — P1 usability & throughput

**Commit I — P1-4 + P1-5: pf_odf model correctness knobs**
- Distortion: read p-coeffs + RhoD, map via
  `midas_distortion.v2_coeffs_from_named` (paramstest is **v1-ordered**, p3 =
  phi4 phase-degrees — use core.py:126-180 authoritative map), opt-in
  `apply_distortion=True` kwarg on `build_model_from_paramstest`.
- `ring_numbers: Sequence[int] | None = None` kwarg (None = current behaviour).
- Tests: v1→v2 mapping sanity (no meter-scale shifts), ring override.

**Commit J — P1-6: productize GN raw-frame mini-calibration**
- Fetch `mini_calib.py` from alleppey (`/home/s20a/sharma_analysis/mini_calib.py`)
  → port into midas_pf_odf as `calibrate_raw_frame_geometry(ds, cache, …)`.
- Tests: synthetic geometry perturbation recovered (bump tx/BC/Lsd, verify
  convergence to truth); torch-diff + CPU/MPS per repo policy.

**Commit K — N5 + N6 + N10: per-scan fan-out + thread hygiene**
- peakfit stage: per-scan fan-out across GPUs/processes with per-scan claim
  (lockfile/rename) — no two-runner race. `stages/peakfit.py:66-120`.
- transforms + zip_convert stages: same fan-out (I/O-aware workers for
  zip_convert).
- N10: orchestrator caps worker OMP/BLAS threads itself.
- FF guard: `_run_ff` single-call path untouched; shared helpers
  signature-stable. Tests: CPU fan-out determinism (same outputs as serial).

**Commit L — N7 + N8 + N11 + E5: knobs & metadata**
- N7: per-stage device override for binning (default = inherit).
- N8: `MinIntegratedIntensity` in fit_setup (default 0/off; written to
  paramstest only when set; E6 lists already extended in Commit C — add key).
- N11: `WorkFolder`/`--scan-work-dir` distinct from RawFolder
  (`_pf_scans.py:218-236`, `stages/zip_convert.py:67`).
- E5: midas_zipper derives omegaStep from data/params when flag absent; never
  writes false 0.0 (omit/NaN otherwise).

*Gate: package suites green; serial-vs-fanout output equality test green.*

## Phase 3 — P2 research-design items (each needs a design decision from you)

**Commit M — P2-7: saturation-aware peakshape loss** ← current pf-odf blocker
- Per-pixel saturation mask (weight 0 where measured ≥ UpperBoundThreshold)
  through `assemble_grain_patch_data` → loss. **Design Qs for you:** mask-only
  vs mask + per-voxel intensity weights; threshold source (paramstest key vs
  kwarg). Test: synthetic flat-top blob — strain bias removed vs unmasked.

**Commit N — P2-8 + P2-9: sparse smoothness + spread hand-off**
- Scatter-based neighbour smoothness on `grid_ij` (no dense reshape) —
  `lambda_smooth>0` works on sparse grains.
- Stage-2→3: accept per-voxel `spread_init`, pool by `spread_region_map`.
- Tests: sparse-grain regulariser finite + reduces roughness; handoff shapes.

**Commit O — E4: grain-odf spread-DOF guardrails**
- Clamp-detector (warn/flag after N steps pinned), LR auto-scale from initial
  gradient magnitude, weighted-median spread (+weight-within-X°) reported
  alongside wRMS.

**Commit P — E7: d0 advisory in process_grains** (design sign-off: wording)
- When /residuals ring-offset flag trips (>200 ppm): print recovered a₀ via
  `recover_d0_cubic_free_standing` + exact `LatticeConstant` paste line.
  **Advisory only, never auto-apply.**

**Commit Q — N12: merge_overlaps** (pick one)
- Port legacy `MergeOverlappingPeaks` bounding-box merge, **or** quantify
  peakfit's omega-tail merge vs legacy on a reference layer and document.

**Investigation R — P2-10: completeness 0.5× root-cause** (may end as doc, not code)
- Trace nExpected in `midas_index`/find_grains (Friedel factor-2 suspect) on
  the Wenxi/SOH data. Outcome: fix convention OR document "new 0.2 ≈ legacy
  0.4" + rescale param guidance.

## Phase 4 — Docs + deployment reconcile

1. `midas_pf_odf/dev/PF_HEDM_NEW_DATASET_HANDOFF.md`: all 8 §Corrections.
2. `manuals/`: /residuals schema (done in B), two SpotID spaces (until E3
   ships it's moot — update to point at OrigSpotID), raw-frame convention
   notes (frame layout, µm→px map, raw≠DetCor warning), RMSErrorStrain
   semantics, new paramstest keys.
3. **Deployment matrix reconcile** (read-only ssh survey first):
   - copland s20iduser: transforms 0.7.2 (old header) → redeploy bumped wheel
     (`--no-build-isolation --no-deps`, no PyPI on that account).
   - toro/s20hedm: carries patched transforms → align to committed version.
   - alleppey s1iduser: PYTHONPATH overlay → refresh overlay.
   - Each redeploy proposed to you before execution.

---

## Verification limits (explicit)

- **No local GPU**: CUDA paths tested CPU+MPS locally; CUDA validation on
  alleppey/sentosa or deferred to you.
- **Real-data final gates** (SOH strain sanity post-P2-7, Ni run continuity,
  emerson header re-read) run remotely; I stage, you approve execution.
- The live Ni run on toro uses the Phase-0 code — Phase 0 commits exactly what
  it runs (plus header completion); no behaviour change to that run.

## Order rationale

Phase 0 first per commit-foundation-before-anything (repo is source of truth
with 3 hosts diverging). Phase 1 before Phase 2 because schema/params commits
(C, F) define names every later item consumes. Phase 3 last: needs your design
input and real-data validation. Docs/deploy last so they describe shipped code.
