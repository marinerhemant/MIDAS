# FF-HEDM — measurement envelope

**Instrument:** 1-ID, single monolithic GE panel, one layer
**Last checked:** 2026-08-12 · **Owner:** Hemant Sharma (hsharma@anl.gov)

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
  the grain pass can.
- **Diagnostics sidecar not written.** If `residuals/spot_table` is absent the report is
  descriptive only. That is a pipeline-version question, not a measurement limit — the
  current pipeline writes it.
- **Few rings because of saturation.** Recoverable by re-acquiring with a different exposure
  or attenuation; report as "not acquired", never "not available".

---

**Checklist before this file is trusted**

- [x] Every row has a unit or is explicitly dimensionless
- [ ] Every bound in §2 names what limits it — **three rows still blank** (frame rate, stage travel, damage dose)
- [x] Nothing in §1 or §3 is phrased as a suggestion
- [x] `Last checked` is within the current run cycle
