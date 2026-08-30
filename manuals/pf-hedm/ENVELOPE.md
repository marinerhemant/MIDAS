# pf-HEDM — measurement envelope

**Technique:** scanning 3DXRD / pf-HEDM — one detector, one layer, 2-D voxel grid
**Last checked:** 2026-08-25 · **Owner:** MIDAS maintainers

> Part of the **pf-HEDM doc set**. Spine: [`README.md`](README.md). Contract:
> `~/opt/beamreport/DOCS_SPEC.md` §6 (separate repo).

What this measurement can and cannot determine, and which of those is changeable. Read it
before promising an answer, and before suggesting a different measurement.

> **Not the scope gate.** The scope gate in the spine says whether these *recipes* apply to
> your data. This file says whether the *measurement* can answer the question. A dataset can
> be squarely in scope and still unable to support what is being asked of it.

---

## 1. Fixed — cannot change this run or beamline cycle

No suggestions here. State the consequence and the substitute.

| Property | Value | Provenance | What it makes unobtainable | Substitute |
|---|---|---|---|---|
| Attenuation / exposure set for the run | one setting for all rings | run configuration | **If the low rings saturate and the high rings sit near noise, no single exposure serves both.** Per-voxel strain is signal-limited for the run. | Report orientation/KAM/GROD (robust); flag strain magnitudes provisional. HDR / graded attenuation is a *next-run* change (§2). |
| Detector count | one panel | scope gate | Multi-panel merging is a no-op; `cross_det_merge` does nothing. | A multi-panel run is a different doc set. |
| Layers per run | one | scope gate | No through-thickness stacking within a run. | Stitch across layers, a separate step. |
| ω coverage gaps | blocked spans in `OmegaRange` | run configuration | Fewer reflections per voxel; strain conditioning weaker. **Not a defect.** | Full 360° coverage next run recovers reflections + Friedel pairs. |
| Powder calibrant sensitivity to `tx` | zero | shared with FF | A powder standard **cannot constrain `tx`**. | Hold `tx` fixed in powder calibration, refine from grains after. |

**Consequence worth stating on any report:** on an attenuated scan the per-voxel strain
*magnitude* is set by the illumination/exposure, not by the fit. A report that treats a
noisy per-voxel strain as a calibrated measurement is wrong; one that "improves" it by
loosening bounds is making it worse.

## 2. Configured — set per run, changeable next time

The only tier where "what could be observed differently" has an answer.

| Parameter | Used | Achievable range | Limited by | What changing it would buy |
|---|---|---|---|---|
| **Dynamic range** (HDR / graded attenuation) | single exposure | multi-exposure merge or per-ring attenuation | station configuration | **The single largest strain improvement.** Un-saturating the bright rings and lifting the high rings turns per-voxel strain from provisional to quantitative. |
| **Ring count / selection** | per run | rings visible + unsaturated | detector extent, energy, saturation | Reflections-per-voxel and therefore strain conditioning. pf-odf is identifiability-limited at ≈1 usable reflection/voxel. |
| ω step and range | per run | acquisition time | — | Peak sampling in ω, Friedel-pair availability, reflections per voxel. Closing the gaps helps directly. |
| Beam size / scan step | per run | optics + stage | in-plane spatial resolution (voxel size) | Finer voxels resolve steeper gradients; coarser is faster. |
| **Beam focus: scanning point vs a single line-focus far-field exposure** | scanning | either, same station | flux density and spot crowding | **Measured, same layer, same detector, identical exposure/threshold/transmission: the LF frame is exactly Σ(13 PF frames) × 0.164** — superposition holds, and the line focus delivers **1/6.1** the flux density. Signal ×0.164 with background ×2.1 vs one PF frame ⇒ **signal-to-background 12.8× worse**; only **6 %** of PF's above-threshold signal survives at LF sensitivity (LF found 31 690 spots against PF's 935 620). Going to a line beam to save time costs ~94 % of the peaks. |
| **Scanned field width** (n_scans × step) | per run | stage travel, time | **A grain comparable to or larger than the scanned field has no recoverable shape** — the projections never see it end. Flagged at occupancy > 0.65 (phase 6 §6.6): 2 of 10 grains on the reference campaign, at 0.84 and 0.78. | Widening the field to comfortably exceed the largest grain. This is the *only* one of the shape limits with a known fix. |
| Energy | per run | source + optics | ring accessibility, penetration | Which rings are reachable. |

**Rows deliberately blank.** Detector maximum frame rate, sample damage dose, and stage
travel limits are not recorded in this doc set. Until filled in, a report **will not**
propose changing exposure/dwell against them — an undeclared bound produces no
counterfactual, by design.

## 3. Intrinsic — the sample or the physics forbids it

No configuration helps.

| Question | Why it is not answerable | Distinguish from |
|---|---|---|
| Strain **inside** a crack/void | Broken or absent material does not diffract; those voxels have no signal to fit and read as low/undetermined, not zero. | The **process zone around** the crack does diffract — KAM/GROD localise it well. That is measurable. |
| Absolute hydrostatic strain without a strain-free reference | Lattice *parameters* are measured; converting to strain needs d0. A d0 error appears as a uniform hydrostatic offset. | For a **cubic, free-standing** polycrystal the free-standing equilibrium ⟨ε_hydro⟩_V = 0 recovers it. Not general. |
| Sub-voxel grain shape | The voxel is the spatial unit; structure below it is in the peak *shape*, not in a spatial map. | pf-odf recovers the sub-voxel orientation/strain *distribution* (a width), not a sub-voxel image. |

## 3b. Unexplained — measured, and not yet placed in a tier

**This section exists because forcing an open problem into one of the tiers above would be
a lie in either direction.** Calling it Intrinsic claims physics forbids it; calling it
Configured promises a next-run fix. Neither is known.

| Question | What is measured | Why it is not tiered |
|---|---|---|
| **Grain SHAPE from the sinogram reconstruction** | Reconstruction residual **0.82–0.84, invariant** across FBP / SIRT / MLEM, ± support, every sinogram variant, and self-fitted vs borrowed reference masks. Some grains return clean compact objects at the right place; the two largest on the reference campaign do not. Eleven mechanisms preregistered and tested; the full simulated stack (geometry + absorption + extinction + detector merging + thresholding) reaches an artifact level of **0.239**, which matches the two grains that reconstruct *well* (0.094, 0.211) and is **2.0× and 3.8× short** of the two that fail (0.476, 0.896). | The modelled physics accounts for the successes and not the failures. **The cause of the failures is unknown.** It is not attenuation (§1), not the scanned field for these grains (occupancy ≤ 0.51), and not any of the eleven. |

**Consequence for a report:** grain **positions** from sinograms are quotable (fit rms
1.3–2.1 µm on clean grains, agreeing with the voxel map to ~1.7 µm). Grain **shapes** are
not. If shapes are the deliverable, absorption tomography of the same sample is the better
instrument — say that rather than shipping the diffraction shapes with a caveat.

**Two metrology constraints that came out of the same work**, and that bind any future
attempt: **dice must not be used** on this problem (it thresholds at the true voxel count,
is blind to sub-amplitude artifacts, and caused four separate requalifications), and
**half-split consistency is the metric that survived** — reconstruct each grain from two
disjoint halves of its rows and correlate. It needs no ground truth, no mask and no chance
floor, and it cleanly separated a self-consistent sinogram (0.82–0.92) from a
noise-dominated one (≈ 0).

## 4. Derived limits

What follows arithmetically from §1–2. A report may quote these directly.

| Quantity | Limit | From |
|---|---|---|
| Per-voxel strain trustworthiness | **magnitude provisional / pattern real** on signal-limited scans | §1 attenuation, §2 rings |
| Orientation / KAM / GROD | robust even when strain is not | orientation needs far less SNR than peak-shape |
| Reflections per voxel | ≈1 usable at high attenuation with ω gaps | §2 rings + §1 gaps |
| Grain **position** from a sinogram | ~1.3–2.1 µm rms on clean single grains; ~1.7 µm against the voxel map | §2 scan step, and the concentration filter (phase 6 §6.5) |
| Grain **shape** | **not quotable** — cause unknown | §3b |
| Sample **boundary** | ~1.5 µm, from two independent grain-free routes agreeing (14.50 and 16.04 µm) | phase 1b. The **tilt** of that boundary is *not* resolved (+2.4° / −8.5° / +4.8° from three methods) |
| **Completeness as evidence** | only **above the measured chance ceiling** — five layers gave none / 0.5333 / 0.6957 / 0.7500 / 0.8333 | phase 7. Below it real and chance overlap. **Not predictable from spot density** (the densest layer came in mid-table) — it must be measured on the layer in hand, and a denser scan does not make the map safer |
| **Grain count from a per-voxel map** | **not a census** — `OneSolPerVox` maps only the largest grains | phase 7 §7.5. One layer gave 284 voxels → 10 distinct orientations (≈42 µm/grain) against a ~0.29 µm primary-particle size |

## 5. Did not versus cannot

Skipped on a given run but perfectly possible. These read identically to hard limits and
mean the opposite.

- **`tx` not refined from grains.** A choice — the powder pass cannot do it (§1), the grain
  pass can.
- **Strain not fit.** If only the grain map was produced, per-voxel strain is *not attempted*,
  not *not achievable*. KAM/GROD were still available from the orientations alone.
- **Cross-modal (tomo) overlay absent.** If the tomo↔lab transform (flip + rotation-centre)
  was never recorded, the overlay was *not done*, not *impossible* — a metadata gap, fixable
  next run with a shared fiducial.
- **Sample boundary not located.** If the vacuum was never masked, the per-voxel statistics
  were averaged over vacuum — *not done*, and cheap to do: the spot-count sinogram needs only
  the peak search (phase 1b).
- **`positions.csv` handedness not measured.** On the reference campaign the translation
  motor readbacks were constant across every file, so the translation was never logged and
  the handedness rests on a convention. That is a **metadata gap fixable next run** by
  logging the scanning motor — not a property of the technique. Until then it must be
  reported as a convention, not a measurement.
- **Concentration filter / occupancy flag not run.** Both are cheap, both are shipped
  (phase 6 §6.5, §6.6), and the position gain is real. Absent from a report means *not
  attempted*.
- **ω-shuffle null not run / chance ceiling not measured.** It is a re-index only — no
  re-prep, no re-peakfit (~95 min at 30 cores for 169 voxels) — and the banked real arm can
  serve as one arm provided it used the same binary and the same `ScanPosTol`. Absent from a
  report means the map's completeness has **not been shown to carry information**, not that
  it cannot be. Phase 7.

---

**Checklist before this file is trusted**

- [x] Every row has a unit or is explicitly dimensionless
- [ ] Every bound in §2 names what limits it — **three rows still blank** (frame rate, stage travel, damage dose)
- [x] Nothing in §1 or §3 is phrased as a suggestion
- [x] An open problem is in §3b, not forced into §1/§2/§3
- [x] `Last checked` is within the current run cycle
