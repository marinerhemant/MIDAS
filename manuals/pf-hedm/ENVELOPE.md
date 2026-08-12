# pf-HEDM — measurement envelope

**Technique:** scanning 3DXRD / pf-HEDM — one detector, one layer, 2-D voxel grid
**Last checked:** 2026-08-12 · **Owner:** MIDAS maintainers

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

## 4. Derived limits

What follows arithmetically from §1–2. A report may quote these directly.

| Quantity | Limit | From |
|---|---|---|
| Per-voxel strain trustworthiness | **magnitude provisional / pattern real** on signal-limited scans | §1 attenuation, §2 rings |
| Orientation / KAM / GROD | robust even when strain is not | orientation needs far less SNR than peak-shape |
| Reflections per voxel | ≈1 usable at high attenuation with ω gaps | §2 rings + §1 gaps |

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

---

**Checklist before this file is trusted**

- [x] Every row has a unit or is explicitly dimensionless
- [ ] Every bound in §2 names what limits it — **three rows still blank** (frame rate, stage travel, damage dose)
- [x] Nothing in §1 or §3 is phrased as a suggestion
- [x] `Last checked` is within the current run cycle
