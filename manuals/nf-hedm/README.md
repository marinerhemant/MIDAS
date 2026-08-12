# NF-HEDM Reconstruction Runbook — survey → calibrate → reconstruct → report

**Use this doc to start a fresh session on a near-field dataset this pipeline has never
seen.** Paste it in together with `LAB_NOTEBOOK.md`, then give three lines:

```
Data folder:     <ABSOLUTE PATH>     # the image tree, e.g. /gdata/dm/1ID/<year>/<bt>/data/nf/
Metadata folder: <ABSOLUTE PATH>     # or "find it" — see §3a
Sample material: <e.g. gold cubes / Ti-7Al / unknown, tell me from the data>
```

Everything else the agent works out or asks for. **The order in §0 is not optional** — it
was confirmed with the instrument scientist, and getting it wrong is itself a documented
failure mode.

**Scope.** Everything here assumes **1-ID**: §3a–§3g and the `2047 − index` BC convention
are that beamline and that detector chain only. 20-ID HT-HEDM is a different acquisition,
detector and file format — see §3h, which lists two blockers that must be cleared before
that data can enter the pipeline at all.

**On any other beamline, stop and ask rather than adapting a recipe.** The array→lab
mapping has to be re-derived, not inherited, and getting it wrong **mirrors the
microstructure invisibly** — the same silent failure mode as the ω sign, with nothing in
the `.mic` that shows it (§3h).

Not a tutorial. Follow the steps in order; each one names the file to read, the command to
run, the field to look at, and the branch to take.

**Companion: `LAB_NOTEBOOK.md`.** This file says what to do. The notebook records
what was found on the `bt_1id_jul26` campaign, how each claim was measured, and which
attractive ideas were **retracted** — read its §5 before re-opening any question, and its
§4b before touching calibration. Findings are summarised here only where they change what
you should type.

### The doc set — what to read when

**This file is the spine, and the only one you need loaded the whole time.** Everything
else is opened when you reach it. Section numbers are continuous across the set.

| File | Holds | Read it |
|---|---|---|
| **`README.md`** (this) | scope gate, install gate, the order (§0), hard rules, halt conditions | always — start here |
| [`phase-0-setup.md`](phase-0-setup.md) | §1, §1a, §1b — environment, beamline vs your own machine | before touching data |
| [`phase-1-metadata.md`](phase-1-metadata.md) | §2–§4b — ω sign, metadata folder, scan definition, energy, distance | first, and §2 cannot be checked later |
| [`phase-2-frames.md`](phase-2-frames.md) | §5–§5e — look at the raw frames, counting regime, whether rings exist at all | before building anything |
| [`phase-3-geometry.md`](phase-3-geometry.md) | §6–§7f — BC from `DetZBeamPos`, `Lsd` from spots, calibrant refinement | the long one; the plateau lives here |
| [`phase-4-run.md`](phase-4-run.md) | §8–§8m — the pipeline, the nine-command route, denoising, `SumFrames` | when launching |
| [`phase-5-read-report.md`](phase-5-read-report.md) | §9–§9d, §11–§12c — the `.mic`, validation buckets, report, done-means | when a result exists |
| [`PARAMETERS.md`](PARAMETERS.md) | §10–§10i — every parameter key, who reads it, units | when writing `params.txt` |
| [`DIAGNOSIS.md`](DIAGNOSIS.md) | symptom → discriminating test → cause → lever | **when something looks wrong** — indexed by symptom, not by step |
| [`RUNBOOK.md`](RUNBOOK.md) | §R1–§R3 — where it runs, what healthy looks like *with conditions*, pick-up point | on resume, and before quoting any number as "normal" |
| [`LAB_NOTEBOOK.md`](LAB_NOTEBOOK.md) | evidence, measurement ledger, **retracted claims** | before re-investigating anything |
| [`ENVELOPE.md`](ENVELOPE.md) | what this measurement can and cannot determine, sorted by whether anything can be done about it | before promising an answer, and **before suggesting a different measurement** |

**Citation convention.** A bare `§n` means *this doc set* — use the table above to find
which file. A reference to the notebook is always written `Lab Notebook §n`, because the
notebook has its own numbering that collides with these.


Citations are `path:line` relative to `$MIDAS = /Users/hsharma/opt/MIDAS`. Read them with
absolute paths (`/Users/hsharma/opt/MIDAS/<path>`). Every non-obvious claim carries one,
and `utils/doc_citation_check.py` (wired into the pre-commit hook) fails the commit when a
cited file, line or symbol no longer exists — so a citation here points at real code.
**It cannot check the claim, only the pointer:** the line is right, the sentence about it
may still have gone stale.
Claims that are convention, or that could not be verified, are flagged inline and listed
again in §11. **Do not promote a §11 item to a fact.**

Maintained code = four Python packages: `midas_nf_pipeline`, `midas_nf_preprocess`,
`midas_nf_fitorientation`, `midas_hkls`, plus the viewer `gui/nf_qt.py`. `NF_HEDM/` is
soft-deprecated C; only its example paramfile and seed cache are used here. Versions —
**tree and shared env currently differ, see §1.**

---

## STOP — read this before touching anything

### When to stop and come back with a question

**"Get back to me if you get stuck" does not fire here.** The failures in this pipeline do
not feel like being stuck. A wrong ω sign mirrors the microstructure with confidence
unchanged. A wrong geometry reaches confidence **1.0000** — that is a *plateau*, and
`ty` seeds 2° apart all reach it (hard rule 14). A re-seeded refinement ratchets the tilts
1°/pass while confidence stays high (rule 15). In each case the run finishes and looks
right.

So the trigger is not confusion. **Halt on these named conditions, whether or not anything
seems wrong:**

| Condition | Why you cannot decide it yourself |
|---|---|
| par field 9 is **not** `aero` | no other value's ω sign has ever been established here (§2, §11) |
| no folder with `FileCount.txt` + `fastsweep_Emon.txt` + `*_SequenceOfEvents.log` | you cannot write a paramfile at all (§3a) |
| the data is **20-ID / HDF5 / Bluesky** | different acquisition, format and scaling; two blockers are still open (§3h) |
| any beamline **other than 1-ID** | the `2047 − index` BC convention encodes *this* detector; getting it wrong mirrors the microstructure invisibly (§3h) |
| any package **below floor** after §1 | `SumFrames` units inverted; a mixed resolve is silently wrong (§1, §8j) |
| `fit_axis(...).is_reliable` is **False** | the shadow tracker refused — branch on it, do not override (§6e) |
| both ybc routes fail | ybc is **not measurable from this scan**; inheriting it is a decision, not a default (§6e) |
| the specimen is extended / irregular and you need a particle position | `position_candidates_um` and `triangulate` both assume compact; mask the annulus instead (§6e, §6i) |
| calibration reached confidence 1.0 and you are about to accept it | confidence is not an acceptance criterion — check `BoxSize` and an external constraint first (rules 14, 17; §7b) |
| `mic2grains` radii are needed but `EdgeLength ≪ GridSize` | the radii describe the probe triangle, not the cell — a measured 500× discrepancy (§8a) |
| this document and the tree **disagree** | report it; do not work around it (§1) |

When you halt, say which row fired, what you measured, and what you would need in order to
proceed. Everything not blocked by it should still be finished first.

### Hard rules

1. **Determine the ω sign convention first (§2).** Field 9 of `<beamtime>_NF.par`. If it
   is `aero`, then **ω_MIDAS = −ω_aero** and the paramfile needs the *negated* sweep. Get
   this wrong and the reconstruction is **mirrored**, which is **not detectable from the
   `.mic` alone**. This is step 1 of every new dataset, no exceptions.

   > **This rule has a method at 1-ID only.** There is no `.par` file at 20-ID, and this
   > doc set contains **no procedure for establishing the ω sign there** — `exchange/theta`
   > is recorded per frame, but nothing here establishes whether its sign matches
   > ω_MIDAS's convention. Masking all four sign candidates and letting the reconstruction
   > settle handedness (`PARAMETERS.md`, Lab Notebook §7) resolves the *mask position*; it
   > does **not** determine the sign for the paramfile or the forward model. Because a
   > wrong sign mirrors the microstructure with confidence unchanged, no downstream check
   > in this set — not confidence, not the neighbour-misorientation coherence test — can
   > catch it. **Treat 20-ID ω sign as undetermined and halt.** Filling this gap needs a
   > determination at the beamline, not a re-reading of these files.
2. **The TIFF tree does not contain the metadata (§3).** `/gdata/dm/1ID/<year>/<beamtime>/data/nf/`
   holds only images. Distances, ω, energy, exposure live in a *separate* acquisition-log
   folder. Find it or stop.
3. **Energy comes from `fastsweep_Emon.txt` field 10, and nowhere else (§3, §4).** Two
   other fields look like energy and are wrong.
4. **Never count spots off a raw max-projection (§5).** It is dominated by cosmic rays.
   Use the temporal-median + LoG path.
5. **`midas-nf-pipeline run` IS the supported route (§8a)** — ten orchestrator defects
   that made it unusable are fixed. Pass `--fit-gpus 0,1` or the fit uses one GPU while
   the rest idle. (Older notes saying "do not use it" are stale; lab notebook §2.)
6. **`--all-layers` is mandatory** on `process-images`; without it only the last detector
   distance's bits survive (`process_images/pipeline.py:229-243`,
   `process_images/cli.py:57-60`).
7. **Read `TriEdgeSize` from column 5 of a data row, never from the `%TriEdgeSize` header**
   (§9a).
8. **`/grains/` in the consolidated H5 is not grains (§9c).** Use `/maps/grain_id` or run
   `mic2grains`.
9. **Units: µm, degrees, Å** (Å for wavelength and lattice parameters only). Output Euler
   angles are **radians**; so are `.map.kam` and `.map.grod`.
10. **DetZ ≠ Lsd (§4).** Only *differences* between DetZ readbacks are trustworthy.
11. **`BC` pixel convention is `2047 − raw_index`, on BOTH axes (§6b).** `ybc = 2047 − col`,
    `zbc = 2047 − row`, where `raw` is what `tifffile.imread` returns. Validated against an
    operator reading to 0.3 px. The constant is **2047, not 2048** — one pixel matters at
    `BCTol 0.2`.
12. **Never borrow the beam tilt β between beamtimes (§6f).** Measure it from that
    beamtime's own DetZBeamPos scan. Borrowing it was wrong by 62× in y.
13. **BC comes from DetZBeamPos; Lsd comes from spots (§6a).** Neither measurement can give
    the other's quantity. Run DetZBeamPos first.
14. **Confidence 1.0 does NOT mean the geometry is right (§7b).** It is a *plateau*: on real
    Au data, `ty` seeds 2 deg apart all reach exactly 1.0000. Never close out a calibration
    on the confidence number alone.
15. **Never re-seed a refinement with its own output (§7b).** `TiltsTol` is relative to the
    seed, so iterating ratchets the tilts ~1 deg per pass while confidence stays high.
    Use `NumIterations` inside ONE invocation instead.
16. **`-multiGridPoints` does not fix an under-determined geometry on a single-crystal
    calibrant (§7b).** All voxels are one grain, so N voxels give one grain's constraint.
17. **Check `BoxSize` before blaming the geometry (§7d).** Unset, it costs exactly the last
    few percent of confidence (0.949153 vs 1.000000) and looks like a small geometry error.
18. **On weak signal, fix the REDUCTION before the geometry (§8f).** Denoising the
    median-corrected residual and dropping the threshold to ~0.7 σ was worth 3.6× the
    voxels at C ≥ 0.9; a converged geometry refinement was worth +0.005 FracOverlap.
    **Set that threshold with `BlanketSigma`, not `BlanketSubtraction`** — the latter was
    an int and could not express a sub-σ step at all (§8k).
19. **Compare reconstructions by field, never by checksum (§8g).** `MicFileBinary` records
    carry a per-voxel `RunTime`, so two bit-identical *physics* results have different md5s.

The rules above are about distrusting the *data* and the *code*. These four are about
distrusting your own run, and they are the ones a context-free session skips:

20. **Suspect success.** Confidence 1.0 is a *plateau*, not a verdict (hard rule 14); the
    orchestrator's `Wrote 1 grains` does not mean one grain (§8a); `MicrostructureBinary.mic`
    reads all-zero mid-run by design (§8a); a phase that indexes nothing exits cleanly
    (§8a). Ask what the stage would look like if it had silently no-opped, then check that
    specific thing.
21. **Debug your own configuration before the data or the physics.** Order: a version
    below floor (§1) → a key whose units changed (`SumFrames`, §8j) → a dropped or
    misspelled key (`LatticeConstant` vs `LatticeParameter`, §10b) → a sign or array
    convention (§2, §6b) → only then the sample. Lab Notebook §4c and §6a record claims
    that were **retracted** once the mundane cause surfaced.
22. **Never take a number from a name.** Not the energy from a filename (§4a), not the
    frame count from the `DoVolume(...)` arguments — those are what was *requested*, the
    per-frame log is what was *written* (§3g). Not the raster from a folder name: a
    companion pipeline measured a folder called `10x10um_0p25umStepSize` as 20.000 × 14.142 µm
    because the sample sat at 45° to the beam
    (`LaueMatching/scripts/pipeline/Laue_Handbook.md`, Phase 0).
23. **Do not reimplement what a `midas_*` package already does.** §3h says it directly for
    the HDF5 reader — *"Reuse `midas_calibrate_v2.io.readers.read_image` … do not write a
    new reader"* — and §8a for the pre-allocated outputs. Grain segmentation by
    misorientation → `midas_stress.misorientation` (§8a); structure factors →
    `midas_hkls` (§8l).

### Traps that silently corrupt results

| Trap | Symptom if missed | Where |
|---|---|---|
| `aero` ω sign | mirrored microstructure, plausible confidence | §2 |
| `NF.par` field 29 looks like energy | wrong wavelength → wrong ring radii | §4 |
| `<beamtime>.spe` `#U Energy:` is stale | same | §4 |
| Duplicated image number at sweep boundaries | one sweep silently loses a frame | §3d |
| DetZ readback used as absolute Lsd | systematically wrong geometry | §4 |
| Cosmic rays in a max-projection | fictitious spot counts, fictitious "coverage" | §5 |
| `%TriEdgeSize 0.000000` header | `mic2grains` silently falls back to global merging; grain radii collapse to 0 | §9a |
| `Confidence == 0` rows dropped from text `.mic` | row *i* of `.mic` ≠ voxel *i* | §9a |
| four mislabelled H5 datasets | reading the wrong quantity under a plausible name | §9c |
| `LatticeConstant` instead of `LatticeParameter` | `KeyError` in the HKL stage | §10b |
| BC left in raw array indices | mirrored geometry; **invisible in y**, because BC sits near the detector centre so a flip still looks plausible | §6b |
| `2048 − index` instead of `2047 − index` | 1 px BC error, i.e. 5× `BCTol` in z | §6b |
| β borrowed from another beamtime's paramfile | per-distance BC wrong by tens of px | §6f |
| centroiding `1 − T` over the whole illuminated band | axis scatter 66 px instead of 0.2 px, and it looks like real data | §6e |
| row-permutation "null" in a spot-matching test | null silently re-runs the real analysis and passes | §6i |
| confidence 1.0 read as "geometry solved" | a whole beamtime reconstructed on the wrong plateau | §7b |
| a refinement driving confidence to 1.0, checked with maxC/median | **those statistics are blind to the plateau failure.** On `nf_sampleD` pass C gave maxC 1.000 with 40 % of the disc indexing and the MEDIAN also up (0.229 → 0.368) — the plateau signature. Test the **orientation field** instead: misorientation between spatial NEIGHBOURS vs RANDOM pairs (0.23° / 78 % < 5° vs 40.98° / 4.5 % here ⇒ real grains). A wrong plateau gives a spatially random orientation field | lab notebook §8h |
| chance confidence computed from the lit-pixel fraction | with `hits_d.prod(dim=0)` ANDing 1.65 % and 1.40 %, independent-pixel arithmetic predicts 2.3e-4; the truth was **~1000× higher** because observed and predicted spots cluster in the same regions. **Read the floor off the MEDIAN over the search volume** | lab notebook §8h |
| triangulated `Lsd` used as the final geometry on a WIDE sample | it is a **seed**, not the answer. On `nf_sampleD` triangulation was 211 µm off; after refinement δ landed 6.8 µm from the previous campaign's. Triangulate → refine → then quote | §6i-ter, lab notebook §8h |
| refinement re-seeded from its own output | tilts drift ~1 deg/iteration, confidence stays high | §7b |
| `GridPoints` given 6 tokens instead of a 12-column `.mic` row | parses fine, refines nothing | §7c |
| `BoxSize` parsed but not applied | calibrant plateaus at 0.949 instead of 1.000 | §7d |
| blob size compared using radius from the GRID ORIGIN | sample offset misread as a geometry difference | §7e |
| md5 of `MicFileBinary` used to check reproducibility | `RunTime` differs every run; always "fails" | §8g |
| voxel-count blow-up in `screen()` | 1704 GiB allocation on a full grid | §8h |
| assuming `EdgeLength` must equal `GridSize` | **RETRACTED** — `EdgeLength` is an independent, supported knob (`hex_grid/grid.py:97-153`); small probe triangles on a coarse grid are intentional, and the voxel count never changes. Forcing them equal made triangles 10 µm and cost ~94 GiB/voxel. Lab notebook R2 | §10e |
| `EdgeLength` ≪ `GridSize` with `mic2grains -doNeighborSearch 1` | merge threshold is `2·TriEdgeSize` while neighbours are `GridSize/2` apart ⇒ **every voxel its own grain**; grain areas describe the probe, not the cell | §10e |
| `MinMisoNSaves` left at its **1.0 default** with `SaveNSolutions 1` | a per-window symmetry misorientation dominates runtime, AND a later higher-confidence solution is silently discarded | lab notebook §2 |
| 20-ID HDF5 assumed to be ×64 scaled | **the encoding is PER-CAMPAIGN, not per-detector.** `nfdev_jul26` is 10-bit stored ×64 (max 65472); `bt_20id_jul26b` on the SAME detector serial is 12-bit unscaled (max 4092, unique values 0,2,4,6,8,10,12,16,…). Dividing the second by 64 turns "threshold 2" into "threshold 128" and thresholds the **pedestal** — the background then looks like signal | §3h |
| ring / powder analysis on a coarse-grained NF sample | an NF spot lands at *grain position* + `Lsd·tan(2θ)·d̂`, so rings are smeared by the **illuminated sample width**. On `nf_sampleD` (247 µm wide) that is 2.0× the 111→200 spacing ⇒ **no rings exist**, and any `Lsd` or lattice parameter fitted to the radial profile is meaningless | §5e |
| BC carried over from an earlier campaign at the same beamline | the beam stripe moved **57 px = 31 µm** between `nfdev_jul26` and `bt_20id_jul26b`. Re-measure zbc every campaign | §6d |
| `shadow.track_shadow` left at its `band_frac=0.30` default at 20-ID | tracker wanders into the beam's dim wings; axis is wrong by **+100 to +130 px** and the amplitude comes back clipped. `band_frac=0.70` reproduces the known Au axis to **0.41 px**. `fit_axis(...).is_reliable` correctly returns False — **branch on it** | §6e |
| moving-shadow ybc attempted on an extended specimen | works only for a COMPACT particle. An irregular specimen's deepest-dip centre does not trace a rigid sinusoid (shadow width swung 56→886 px with ω on `nf_sampleD`) and `fit_axis` refuses at every setting | §6e |
| `triangulate` used on a wide sample as if it were a point | the model assumes a point source at BC. Perturbation ≈ (sample half-width)/(typical spot radius): 11 % for a 247 µm specimen vs 3 % for a 70 µm cube. Symptom is the **y-vs-z split** rising (142 µm vs 57 µm) | §6i |
| `BlanketSubtraction ≈ 0.7 σ` on photon-starved data | the residual is >99 % exactly zero so **MAD = 0**; 0.7 σ collapses to the code's σ floor and admits the whole single-count floor | §5d, lab notebook §7b |
| NLM combined with a σ-derived (sub-ADU) threshold | NLM smears isolated single counts into 4-px clusters and **manufactures** spots | §5d |
| assuming one calibrant cube | `nfdev_jul26` has **two** Au cubes, one on-axis and one 497 µm off; a fit that averages them returns a wrong geometry | lab notebook §7d |
| 1-ID `2047 − index` BC convention reused at another beamline | mirrored microstructure, invisible in the `.mic` | §3h |
| a σ-denominated threshold set through `BlanketSubtraction` | it was an **int**: on NLM-denoised data (σ_MAD ~0.27 counts) the smallest legal value is already 3.7σ, so every sub-σ recommendation in §8f/§8k was unwritable. Use **`BlanketSigma`** | §8k |
| a tomo mask built from `position_candidates_um` on unfamiliar geometry | the mapping was wrong **in form**, not sign — the true position is 90° away, so neither the returned point nor its antipode is near the particle. A candidate-point mask returned exactly **0.0000 at every off-axis voxel**, which reads identically to "the particle is absent". Two campaigns pin `θ = −φ − 90°` (`8a5f0184`); old form was out by 985 µm, corrected form predicts both to 3 µm. **Sweep the annulus — the radius is convention-free** | §6e |
| version floors or key semantics taken from this document rather than the tree | in the six days after this file was written, `SumFrames` inverted its units and a new threshold key appeared | §1 |

---

## 0. THE ORDER — do these steps in this sequence

Confirmed with the instrument scientist, 2026-08-01. The section numbers show where each
step is documented. **Getting the order wrong is itself a failure mode**: on `nfdev_jul26`
an operator-supplied constant was written straight into `Lsd` and a full reduction plus a
reconstruction were launched before the geometry existed (lab notebook §7f, F0).

| # | Step | Where | Notes |
|---|---|---|---|
| 0 | **BC per distance.** Use a separate `DetZBeamPos` scan if one exists; otherwise use the direct beam on the detector. Also: use the sample **shadow** to establish how many particles there are, and get the initial `tx` guess. | §6a-§6f, §3h | zbc from the beam stripe; **ybc needs the shadow** — the beam's horizontal width is slit-defined (§6e) |
| 1 | **Distance triangulation.** The only external input needed is **ΔD, the change in distance between successive positions.** | §6i / §6i-bis | Absolute `Lsd` is NEVER taken from the DetZ readback (hard rule 10) |
| 2 | **Process all images → `SpotsInfo.bin`.** | §8b step 5, §3h | **Geometry-independent** (§8e), so it can run in parallel with steps 0-1 |
| 3 | **Parameter optimisation on ONE voxel** known to be inside the sample (from the shadow). | §7c, §7d | See the two hard rules below |
| 4 | **Run one full layer.** With multiple particles, build an artificial **tomo mask** so a generous `Rsample` does not cost a grid full of empty space. | §8b, §10e | Mask recipe in §10e |
| 5 | **Multi-point optimisation**, iterative, keep the best. | §7b(2) | Sample voxels from **different grains** or this degenerates — see below |
| 6 | **Full recon on a real sample**, then multi-point optimisation. | §8 | |

**Two hard rules govern step 3, and both have bitten:**

- **Iterate INSIDE one invocation** (`NumIterations`), never by re-seeding with the previous
  output. `TiltsTol` is relative to the current seed, so re-seeding ratchets the tilts
  ~1°/pass **while confidence stays high** (hard rule 15, §7b(3)).
- **Confidence ≈ 1 is not an acceptance criterion.** It is a *plateau*: `ty` seeds 2° apart
  all reached exactly 1.0000 (hard rule 14, §7b(1)). And check `BoxSize` first — unset, a
  calibrant plateaus at 0.949153 for reasons that have nothing to do with the geometry
  (§7d).

**Step 5 needs more than one grain.** §7b(2) established that `-multiGridPoints` cannot
break the degeneracy on a single-crystal calibrant, because N voxels of one grain give one
orientation's worth of constraint. If the calibrant happens to contain **two** particles
(as `nfdev_jul26` does — lab notebook §7d), deliberately draw voxels from **both**;
otherwise step 5 reduces to the documented negative.

**Prefer an external check over a self-consistency one.** Where two particles exist, their
separation measured from **absorption** is independent of any diffraction fit, so
"does the fitted geometry reproduce that separation?" is a real test in a way that
"did confidence reach 1?" is not.

---
