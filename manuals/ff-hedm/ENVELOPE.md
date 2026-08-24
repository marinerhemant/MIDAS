# FF-HEDM — measurement envelope

**Instrument:** 1-ID, single monolithic GE panel, one layer
**Last checked:** 2026-08-23 · **Owner:** Hemant Sharma (hsharma@anl.gov)

> Part of the **FF-HEDM doc set**. Spine: [`README.md`](README.md). Contract: `~/opt/beamreport/DOCS_SPEC.md` §6 (separate repo, not under `$MIDAS`).

What this measurement can and cannot determine, and which of those is changeable. Read it
before promising an answer, and before suggesting a different measurement.

> **Not the scope gate.** The scope gate in the spine says whether these *recipes* apply to
> your data. This file says whether the *measurement* can answer the question. A dataset can
> be squarely in scope and still unable to support what is being asked of it.

---

## 1. Fixed — cannot change this cycle

No suggestions here. State the consequence and the substitute.

| Property | Value | Provenance | What it makes unobtainable | Substitute |
|---|---|---|---|---|
| Beam shape (`trend.amplitude_growing`, `trend.periodic`, `param.residual_correlated`) | line or box, not a point | station configuration | **Position along the beam is weakly constrained.** Large `DiffPos` and \|Δy\| with small angular residuals is a geometry property, not a defect. | Orientation and in-plane position stay trustworthy and should be reported as such. Do not "fix" the position spread. |
| Detector count | one monolithic GE panel | spine scope gate | Multi-panel merging is a no-op here; `cross_det_merge` does nothing. | none — a multi-panel run is a different doc set |
| Layers per run | one | spine scope gate | No through-thickness stacking within a run. | Match and stitch across runs, which is a separate step. |
| Powder calibrant sensitivity to `tx` | zero | `manuals/Reconstruction_Reports.md:170`, [`DIAGNOSIS.md`](DIAGNOSIS.md) | A powder standard **cannot constrain `tx`** (rotation about the beam) at all. Refining it against powder is fitting noise. | Hold `tx` fixed during powder calibration, then refine it from the grains in a second pass. |
| Structure-factor spread across the reflection list | phase-dependent | measured 2026-08-23, `midas_hkls.f2_normalised` | Completeness counts every predicted reflection as **equally obligatory**. A reflection with \|F\|² 1 % of the strongest is scored as a miss exactly like the strongest one, so the achievable number is phase-dependent and two phases are not comparable on it. Al₂O₃ (R-3c): 44 of 238 reflections below 1 % of max, median \|F\|² 0.070. NMC811 (R-3m): median 0.083. fcc Ni: median 0.580 — nearly uniform, unaffected. | Declare the basis (`PhaseAtom`, or `PhaseCIF`) so `hkls.csv` carries an `F2` column, then set `ConfidenceMetric weighted`. **Re-tune both gates first** — `Completeness` is applied at indexing *and* again in process-grains as `ConfidenceTol`. Do **not** reach for `ForbiddenF2Threshold` to cull faint reflections: on Al₂O₃ a 1e-2 cut removes 44 reflections but takes 5 of 27 rings with it, and fewer rings means a worse-conditioned strain tensor. |
| Basis-forbidden reflections | phase-dependent; **zero for many phases** | measured 2026-08-23 | Reflections with \|F\|² = 0 can never produce a spot yet sit in the denominator, capping completeness below 1. This needs **two occupied Wyckoff sites whose contributions cancel** — the space group's own extinction rules are already applied by `generate_hkls`, so hcp Ti and Al₂O₃ (within a 200 mm detector) have **none**. Measured where it does bite: diamond Si 8/106 → ceiling 0.9245; a 2a+2c hexagonal cell 28/242 → 0.8843. | `DropForbiddenReflections 1` with a declared basis. Ring numbers are assigned *before* the filter, so survivors keep the numbers `RingThresh`/`OverAllRingToIndex` refer to — but a ring whose every reflection is forbidden vanishes from `hkls.csv` and must be removed from `RingThresh` by hand. |
| Detector mask / dead area in the denominator | not applied unless enabled | `IndexerUnified.c`, 2026-08-22 | A reflection predicted onto a masked pixel, a panel gap, or off the panel was counted as a reflection that was looked for and **not found**. The mask had no effect on completeness at all — `MaskFile` reached only the per-peak `maskTouched` flag. | Generate the active-area bitset with `midas-transforms detector-mask <zarr>` and set the `BigDetSize` it prints. The spot then leaves **both** sides of the ratio. Off by default because it moves a gating number. |
| Absolute grain **size** scale | a canned constant | `radius/core.py:172`, measured 2026-08-23 | **Nothing in the pipeline measures the illuminated volume.** Grain volume is an intensity ratio against `V_gauge = Hbeam·π·Rsample²`, and `Hbeam`/`Rsample` are deliberately generous SEARCH BOUNDS (hard rule 9), never the specimen; `midas_calibrate_v2` templates write `Rsample 1000 / Hbeam 1000 / Vsample 50000000`. On the FF reference run (`ff_refiner_prepost/result/LayerNr_1`, 6112 grains) there is no `Vsample` line, so `V_gauge = 2000·π·2000² = 2.513e10 µm³` and all 6112 grains sum to **6.5 %** of it. Two runs with different search bounds are not comparable on absolute size. | Relative sizes within a ring are unaffected — use those. For an absolute number, supply a measured shape: `midas_transforms.geometry.SampleShape` (analytic cylinder/box needs no tomography, `geometry.tomo` reads a reconstruction) and `radius.shape_correction.correct_grain_volumes`. It emits `GrainRadius_shape` **alongside** `GrainRadius`, never over it. Do **not** put the measured volume in `Vsample` — that key is in the search-bound family. |
| Per-spot absorption in the grain size | cancels to first order | measured 2026-08-23 | The powder reference `powder_int` is itself a sum of *observed* intensities, so any part of a correction common to a whole ring is already in it and **only the spread survives**. Correcting the numerator alone inflates every volume by `⟨1/A⟩` — ≈1.6× in volume, 17 % in radius at μD ≈ 0.5 — uniformly, in the direction people expect. | `normalise_per_ring` enforces `⟨f⟩_r = 1` by construction, so a uniform correction is bit-exactly no correction. Whether a spread exists at all is a property of the specimen: μD 0.05 on NMC811 at 52 keV is null against a ±2.5 % noise floor; μD 1.63 on bulk Ce at 95 keV is not. Measure μD before building the correction. |
| Saturated reflections | dropped whole, unflagged | `midas_peakfit/seeds.py:156` | One pixel over `UpperBoundThreshold` discards the **entire region** — every peak in it — with no flag column. A saturated reflection is a *strong* one, so the loss reads downstream as incompleteness **and** inflates every grain volume on that ring (it was the brightest contributor to the ring's powder normalisation). | Since 2026-08-22 the count is reported per frame and per run. It is still a loss: re-acquire with more attenuation, or raise `UpperBoundThreshold` if the detector is not actually clipping. |

> **Three different things are called "beam" in the parameters. Do not conflate them.**
>
> | key | what it actually is |
> |---|---|
> | `Hbeam` / `BeamThickness` | the illuminated **slab height**; a physical prior on Z |
> | `BeamSize` | a **refinement position constraint**, not a description of the illuminated volume. Like `Rsample`/`Hbeam` it is a deliberately generous bound (hard rule 9), so a value far larger than the real beam is correct, not a bug |
> | the actual beam **width** on the sample | never a parameter at all — it is an acquisition fact, and it is what caps which grains are reconstructable (§2 and DIAGNOSIS `split.illumination_radial`) |
>
> Reading `BeamSize 1000` as "the beam was 1 mm wide" and concluding the illuminated
> volume is 10× too large is a real trap; it was hit on 2026-08-19.

**Consequence worth stating on any report:** the position spread along the beam is set by
the illumination geometry. A report that treats it as a reconstruction defect is wrong, and
one that "improves" it by loosening bounds is making the answer worse. See the
`Rsample`/`Hbeam` hard rule in the spine.

## 2. Configured — set per run, changeable next time

The only tier where "what could be observed differently" has an answer.

| Parameter | Used | Achievable range | Limited by | What changing it would buy |
|---|---|---|---|---|
| **Ring count / `RingThresh`** | per run, measured from the data | rings visible at this energy and distance | detector extent, energy, saturation | **The strain tensor's conditioning.** Below about six rings the tensor is poorly determined and per-grain strain is indicative only. Adding higher-angle *unsaturated* rings is the single largest improvement available. |
| Azimuthal coverage per ring | per run | set by BC and panel extent | beam centre near a panel edge truncates rings | Only rings with **full azimuthal coverage** are safe defaults. A partially covered ring biases the η-dependent terms. |
| `Hbeam` / `BeamThickness` | per run | the **true per-layer beam** | physics: grains outside the beam cannot diffract | Constrains Z to the illuminated slab. **Never set to the sample dimension** — a 10-layer 100 µm scan carrying `Hbeam 1000` lets Z roam ±500 µm. |
| **Beam WIDTH vs sample width** | fixed by the optics for the run | the beam's horizontal extent | slits / focusing | **Caps which grains are reconstructable at all.** Unlike `Hbeam` this is ω-dependent: the beam is fixed in the lab while the sample turns, so a grain at radial offset *r* is lit only a fraction `f(r) = (2/π)·arcsin(hw/r)` of the rotation. Grains beyond `hw` cannot reach `MinMatchesToAcceptFrac` on real spots and are accepted only on coincidences. A beam narrower than the sample means **only the near-axis core is determined** — the rest needs a translation scan. See DIAGNOSIS, `split.illumination_radial`. |
| `MinMatchesToAcceptFrac` / `Completeness` | per run | — | — | The acceptance bar, as a fraction of a candidate's *predicted* reflections. **The code default is 0.0** (`midas_index/params.py:56`), i.e. omitting the key accepts everything; the registry carries no default at all, only `typical=0.8`, and calibrate-v2 templates write 0.4. This row said "Default 0.5" until 2026-08-23 — 0.5 is what the FF reference run's paramstest happens to carry, not a default. Read it together with the row above: it is a bar on **achievable** completeness, and where illumination caps the achievable value below it, the survivors are the padded ones. |
| Lsd | per run | stage-limited | detector translation range | Angular resolution against ring coverage: further out resolves better and captures fewer rings. |
| Energy | per run (keV) | source + optics | undulator, monochromator | Which rings are accessible, and penetration through the sample. |
| ω step and range | per run | — | acquisition time | Peak sampling in ω, and whether Friedel pairs are available for the position path. |

**Rows deliberately blank.** Detector maximum frame rate, stage travel limits, and the dose
at which a given sample starts to damage are not recorded in this doc set and are not in the
parameter files. Until filled in, a report **will not** propose changing exposure or total
dwell. An undeclared bound produces no counterfactual, by design.

## 3. Intrinsic — the sample or the physics forbids it

No configuration helps.

| Question | Why it is not answerable | Distinguish from |
|---|---|---|
| Absolute hydrostatic strain, without a strain-free reference | The measurement determines lattice *parameters*; converting to strain needs a d0. Any d0 error appears as a uniform hydrostatic offset. | For a **cubic, free-standing** polycrystal this is recoverable from the data itself: equilibrium forces ⟨ε_hydro⟩_V = 0, so the mean *is* the d0 error. Recoverable there, not in general. |
| d0 for a **non-cubic** or loaded sample, from the diffraction alone | The free-standing equilibrium argument does not close. | `midas_stress.recover_d0` works but needs single-crystal stiffness **and** orientations as external input. Not obtainable from the pattern alone. |
| Reducing per-grain strain **scatter** by correcting d0 | The d0 correction is purely isotropic. It moves the baseline and leaves deviatoric strain untouched. | It fixes **bias**, and bias is often the headline (hundreds of MPa). Scatter is set by ring coverage and geometry — a §2 question, not a d0 one. |
| Grain shape | FF recovers centroids, not shapes. | NF-HEDM recovers spatially resolved orientation. Different measurement, different doc set. |
| Grain **Z** to better than ~2–3× the beam height | The vertical coordinate is the badly-conditioned one for a thin beam: a grain's spots move little in Z as it moves in Z. Measured on 20-ID with a 100 µm beam (layer step 0.075 mm, so BH100/OL25 confirmed from the data, not the folder name): grain-Z scatter 153 µm where a uniform slab gives 29, improving to 76 after refining `tx`/`Wedge` (§5h). Only 28 % of grains sat within ±50 µm of the beam plane, rising to 44 %. | This is not a bounding-box artefact — the distribution is peaked at the beam and nowhere near the ±500 µm `Hbeam` bound, so rule 9 is not in play. X and Y are unaffected (271/265 µm before, 273/272 after), which is exactly how you tell a resolution limit from a fit absorbing error. **Use X and Y; treat Z as indicative.** |
| `Lsd` independently of the lattice and λ | The ring radius depends on the combination `Lsd·λ/a`, so any two of the three fix the third. Sweeping the assumed cell on nf709 (9077 grains) moved fitted `Lsd` **linearly** — 249 µm per mÅ — while the cost stayed flat to 0.05 %. Fixing `a` therefore *reports* an `Lsd`, but the number is a restatement of your assumption. | Breakable, but only with extra information: several detector distances whose **relative** travel is known (`midas-calibrate-v2 --mode multi --lsd-offsets`) share one `L0` and one λ, which is what makes λ identifiable rather than asserted. A powder standard breaks the `a` half, not the λ half. |

## 4. Derived limits

What follows arithmetically from §1–2. A report may quote these directly.

| Quantity | Limit | From |
|---|---|---|
| Per-grain strain trustworthiness | **indicative only** at ≲4 rings on line-beam data | §2 ring count; `manuals/Reconstruction_Reports.md:226` |
| Position resolution along the beam | set by beam extent, not by fit quality | §1 row 1 |
| Smallest indexable grain | set by spot intensity against `RingThresh` | §2, measured per run — not a fixed number |

## 5. Did not versus cannot

Skipped on a given run but perfectly possible. These read identically to hard limits in a
parameter file and mean the opposite.

- **`tx` not refined from grains.** A choice, not a limit. The powder pass cannot do it (§1),
  the grain pass can — `midas_joint_ff_calibrate.grain_refine`, on **raw** `SpotMatrix`
  pixels (`midas_joint_ff_calibrate/grain_refine.py:426`). It reports the *residual* roll
  left over from whatever `tx` the reconstruction ran with, so compose
  (`tx_total = tx_applied + tx_reported`) and iterate; each pass recovers only part of what
  remains. Feed the result back through `Parameters.txt` and re-run from `transforms`, the
  stage that applies it — **not** a refiner setting, and absent from both backends by design
  (phase-3 §7).
- **Diagnostics sidecar not written.** If `residuals/spot_table` is absent the report is
  descriptive only. That is a pipeline-version question, not a measurement limit —
  `midas-process-grains` ≥ 0.9.2 writes it from every mode that reads FitBest, the
  default `c_parity` included. Below 0.9.2 `c_parity` wrote no sidecar at all;
  `mode=physics` still writes none.
- **`DiffPos` is not re-derivable from the residual table.** The per-spot residuals
  reproduce the refiner's own FitBest `DiffLen` exactly, but the per-grain `DiffPos`
  column is not their mean, while `DiffOme` and `DiffAngle` *are*.
  **Cause diagnosed 2026-08-21** and since verified: `FitBest.bin`'s per-spot records
  and `Grains.csv`'s `DiffOme`/`DiffAngle` were evaluated at the **indexer seed**,
  before any fitting, while `DiffPos` alone came from the refined parameters — so
  cols 19-21 are a pre/post mixture. `Grains.csv` now also carries
  `DiffPos/Ome/Angle` **Pre** (47-49) and **Post** (50-52), the clean
  same-estimator triples; use those to compare before and after refinement.
  Cols 19-21 are kept unchanged for bug-compatibility.
  Still true and still provisional: `DiffPos` and `DiffPosPost` are **not the same
  estimator** (`FitErrors12D` is id-paired, `CalcAngleErrors` angle-paired). The
  median deviation is 0.0000 µm but the **max is 20.14 µm** (FF; 13.6 µm PF), so do
  not restate the median agreement as agreement.
- **Few rings because of saturation.** Recoverable by re-acquiring with a different exposure
  or attenuation; report as "not acquired", never "not available".

---

**Checklist before this file is trusted**

- [x] Every row has a unit or is explicitly dimensionless
- [ ] Every bound in §2 names what limits it — **three rows still blank** (frame rate, stage travel, damage dose)
- [x] Nothing in §1 or §3 is phrased as a suggestion
- [x] `Last checked` is within the current run cycle
