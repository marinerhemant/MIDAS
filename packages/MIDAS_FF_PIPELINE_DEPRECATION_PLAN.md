# `midas-ff-pipeline` → `midas-pipeline` consolidation plan

**Goal:** retire `midas-ff-pipeline` as a separate PyPI package. All FF-HEDM
orchestration goes through `midas-pipeline run --scan-mode ff`.

**Why:** `midas-pipeline` already covers ~90 % of ff-pipeline's surface
(every stage has a twin), but four pieces are unique to ff-pipeline and
five more have skeletal stubs in midas-pipeline. Two packages with
overlapping functionality is a maintenance trap (the `px`-injection bug
that bit Indrajeet's notebook was exactly this kind of "lives in two
places, both half-right" failure mode).

---

## Pre-flight scope

Source we're moving / replacing:

| File | Lines | Status in midas-pipeline |
|---|---|---|
| `stages/cross_det_merge.py` | 365 | **stub (18 lines)** — needs full port |
| `stages/global_powder.py` | 325 | **stub (18 lines)** — needs full port if any user wires it on |
| `testing.py` | 577 | **missing** — used by `midas_pipeline/notebooks/_build.py` |
| `cli.py` | 692 | partial — missing `auto`-knob resolution (`--group-size auto`, `--shard-gpus auto`, `--cpu-shards auto`) and `simulate` subcommand |
| `discovery.py` | 161 | **missing** — `-batch` / multi-layer auto-discovery |
| `seeding.py` | 103 | **missing** — NF→FF seeding resolver |
| `eta_coverage.py` | 324 | **missing** — per-detector η coverage |
| `sr_midas.py` | 125 | **missing** — super-resolution peak-search integration |
| `reprocess.py` | 115 | **missing FF flavor** — pipeline has a recon-focused `reprocess` already |
| `dispatch.py` | 104 | partial — pipeline has `dispatch.py` but no parsl-config loader |
| `detector.py` | 169 | **missing** — DetectorConfig (multi-detector json/ paramstest loader) |
| `eta_coverage.py` | 324 | **missing** |
| `provenance.py` | 203 | similar surface in midas-pipeline; verify schema parity |
| stages we keep (zip_convert, hkl, peakfit, transforms, binning, etc.) | various | already in midas-pipeline |
| tests (~1000 lines) | 11 files | port the ones that test unique ff-pipeline functionality |
| notebooks (7) | — | one notebook per concept; port to `midas_pipeline/notebooks/` |
| dev scripts | 2 | low priority |

**Reverse-dep blast radius:** `midas-suite` only. Two other places reference
it (a notebook builder and a parity gate script).

---

## Phased plan

### Phase 1 — Multi-detector + batch (HIGHEST PRIORITY)

The two pieces a real FF user can't do without:

- [ ] Port `cross_det_merge.py` (365 lines) — replace the stub. Tests:
  port `tests/test_cross_det_merge.py` (133 lines).
- [ ] Port `discovery.py` (161 lines) — batch mode `--batch` /
  `--raw-folder` / multi-layer auto-discovery. Tests:
  port `tests/test_discovery.py` (52 lines).
- [ ] Port `detector.py` (169 lines) — DetectorConfig loader (JSON +
  paramstest). Tests: port `tests/test_detector.py` (73 lines).
- [ ] Wire batch mode + DetectorConfig into `midas_pipeline/cli.py`
  (new `--batch`, `--raw-folder`, `--detectors-json` flags).
- [ ] Parity check: `pip install midas-pipeline; midas-pipeline run
  --scan-mode ff --batch ...` produces byte-identical Grains.csv vs
  ff-pipeline on the synthetic fixture.

### Phase 2 — Testing fixture (UNBLOCKS NOTEBOOKS)

- [ ] Port `testing.py` (577 lines: `generate_synthetic_dataset`).
  Currently `midas_pipeline/notebooks/_build.py` imports from
  `midas_ff_pipeline.testing` — break the dependency.
- [ ] Re-export from ff-pipeline as a back-compat shim so old notebooks
  still work during deprecation window.

### Phase 3 — Utilities

- [ ] Port `seeding.py` (NF→FF resolver, 103 lines). Tests: port
  `test_seeding.py` (89 lines).
- [ ] Port `eta_coverage.py` (324 lines). Per-detector η coverage.
- [ ] Port `sr_midas.py` (125 lines). Optional super-resolution.
- [ ] Port `reprocess.py` (115 lines) — FF flavor; coexist with
  pipeline's existing recon reprocess.
- [ ] Port `global_powder.py` full implementation (325 lines).
- [ ] Port parsl-config loader (`dispatch.py` cluster configs from
  `~/.midas/parsl_configs/`).

### Phase 4 — CLI parity

- [ ] Port `--group-size auto`, `--shard-gpus auto`, `--cpu-shards auto`
  auto-resolution logic (currently explicit-only in midas-pipeline).
- [ ] Port `simulate` subcommand (synthetic dataset generator wired
  into CLI).
- [ ] Port `inspect` subcommand depth (pipeline has it, but ff has
  richer detector/layer listing).
- [ ] Notebooks: port all 7 from `midas_ff_pipeline/notebooks/` to
  `midas_pipeline/notebooks/` (adapt `PipelineConfig` calls to use
  `ScanGeometry.ff()` builder).

### Phase 5 — Deprecation release

- [ ] **midas-ff-pipeline 0.4.0**: import-time `DeprecationWarning`
  ("use `midas-pipeline run --scan-mode ff` instead; this package
  will be removed in 1.0.0"). CLI prints the same warning on invocation.
  All functionality still works.
- [ ] **midas-suite 0.4.0**: floor `midas-pipeline>=0.5.0` (the version
  that includes the ported Phase 1-4 work); drop or downgrade the
  `midas-ff-pipeline` dep to a soft pin.
- [ ] README + docs across the repo: every reference to
  `midas-ff-pipeline` updated to `midas-pipeline run --scan-mode ff`.

### Phase 6 — Removal

- [ ] **midas-ff-pipeline 1.0.0**: empty package with just the
  deprecation warning + a single re-export shim that delegates to
  `midas-pipeline`. Document in changelog.
- [ ] **midas-suite 1.0.0** (or 0.5.0): drop `midas-ff-pipeline`
  entirely from deps and extras.
- [ ] Eventually delete `packages/midas_ff_pipeline/` from the repo.

---

## Risk register

1. **API divergence.** `PipelineConfig` signatures don't match
   (`result_dir/zarr_path/detectors_json` vs `result_path/scan/...`).
   Any user notebook on the library API breaks. Mitigation: keep
   `midas-ff-pipeline.PipelineConfig` working as an adapter that
   internally builds the midas-pipeline equivalent.
2. **CLI flag drift.** Auto-resolve knobs (`auto`) don't exist in
   midas-pipeline. Users running `--group-size auto` get a parse
   error. Mitigation: Phase 4 ports these.
3. **Provenance schema.** Both packages write `midas_state.h5` per
   layer with stage hashes. Verify byte-identical schema before
   removing ff-pipeline (a user with a half-complete run started on
   ff-pipeline must be resumable on midas-pipeline).
4. **Testing fixture C-binary dependency.** `generate_synthetic_dataset`
   calls C `ForwardSimulationCompressed` (requires built MIDAS_HOME).
   Tests skip when unavailable; CI parity must be maintained.
5. **NF→FF seeding semantics.** `patch_params_with_grains` mutates
   paramstest in-place under specific filename conventions. Behavior
   must match exactly or NF→FF workflows produce different grain sets.

---

## Validation gate (before Phase 5 release)

1. Synthetic-fixture parity: `generate_synthetic_dataset(...)` →
   both pipelines produce **byte-identical** `Grains.csv`.
2. Multi-detector test: 4-detector pinwheel layout (ff-pipeline
   notebook 03) reproduces in midas-pipeline.
3. Batch mode test: 3-layer auto-discovered run produces 3 result
   directories with consistent Grains.csv.
4. NF→FF seeding test: NF Grains.csv → FF result matches reference.
5. Resume test: kill mid-pipeline run, resume from saved state, end
   result identical.
6. All ff-pipeline `tests/` (993 lines) port to midas-pipeline and pass.

---

## Effort estimate (revised)

| Phase | Work | Est. |
|---|---|---|
| 1 | cross_det_merge + batch + detector + tests | 2-3 weeks |
| 2 | testing.py port + back-compat | 3-4 days |
| 3 | seeding + eta_coverage + sr_midas + reprocess + global_powder + dispatch | 2 weeks |
| 4 | CLI auto-resolve + simulate + notebooks | 1-2 weeks |
| 5 | Deprecation release (warnings, docs, version bumps) | 3-5 days |
| 6 | Removal + suite cleanup | 2-3 days |
| **Total** | | **~8 weeks** |

---

## Execution log (live)

### 2026-05-29 (today, this session)
- Plan written.
- **Phase 5 partial:** ff-pipeline 0.4.0 with `DeprecationWarning` shipped
  on import + CLI. Declares the move publicly.
- **Phase 2:** port `testing.py` → `midas_pipeline.testing`. Re-export
  shim in ff-pipeline. Identity-tested (same function object on both
  import paths). `_build.py` notebook fixture continues to work
  transparently.
- **Phase 1 deferred:** attempted to port `cross_det_merge.py` (365 lines,
  full multi-detector impl), but it requires plumbing `DetectorConfig`
  + `detectors`/`is_multi_detector`/`detector_dir()`/`stage_dir()` /
  `merged_paramstest` into `midas_pipeline.stages._base.StageContext`,
  which is a bigger surgery (~ 3 new modules + StageContext extension +
  Pipeline._make_context wiring) than fits one safe session. The current
  midas-pipeline `cross_det_merge` stub remains a no-op for
  single-detector runs (which is the common demo path). Logged as
  follow-up at top of Phase 1.

### Deferred (post-meeting work)
- Phase 1 remainder: batch discovery, DetectorConfig wiring
- Phase 3: seeding, eta_coverage, sr_midas, reprocess, global_powder full impl
- Phase 4: CLI auto-resolve, simulate subcommand, notebook ports
- Phase 5 remainder: midas-suite 0.4.0 with floor bump + docs
- Phase 6: ff-pipeline 1.0.0 removal
