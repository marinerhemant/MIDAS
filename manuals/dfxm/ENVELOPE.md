# DFXM — measurement envelope

**Instrument:** ESRF ID06-HXM style acquisition, plus the archived ID03 sets
**Last checked:** 2026-08-12 · **Owner:** Hemant Sharma (hsharma@anl.gov)

> Part of the **DFXM doc set**. Spine: [`README.md`](README.md). Contract: `~/opt/beamreport/DOCS_SPEC.md` §6 (separate repo, not under `$MIDAS`).

What this measurement can and cannot determine, and which of those is changeable. Read it
before promising an answer, and before suggesting a different measurement.

> **Not the scope gate.** The spine's halt table says when to stop, including on APS 6-ID-C
> data. This file says whether the measurement can answer the question at all. A scan can be
> squarely in scope and still not support what is being asked of it.

Content is drawn from the halt table in [`README.md`](README.md), [`DIAGNOSIS.md`](DIAGNOSIS.md)
and `LAB_NOTEBOOK.md`; this file gathers it in one place, sorted by whether anything can be
done about it.

---

## 1. Fixed — cannot change this cycle

No suggestions here. State the consequence and the substitute.

| Property | Value | Provenance | What it makes unobtainable | Substitute |
|---|---|---|---|---|
| Objective numerical aperture | fixed per optic | instrument | Sets the angular acceptance, and with it the mosaicity that can be resolved at all. | none within a run |
| Kinematic validity boundary | t/Λ, **per reflection** | README halt table, Notebook §4b, §7g | Beyond it the reduction **converges to a biased answer with no error flag** — the run succeeds and is wrong. | Check the boundary before quoting; there is no post-hoc correction. Λ ∝ 1/\|F\|, so classify each reflection separately — a weak satellite and its strong parent differ by 10²–10³×. Do not use the mosaic width as the coherent block size. |
| Inter-reflection registration | required for the full tensor | Notebook §2 | Without co-registration metadata, per-reflection maps **cannot** be fused into a deformation-gradient tensor. The fused tensor is meaningless, not merely noisy. | Report per-reflection maps separately. Registration, not photon statistics, is the binding systematic. |
| θ-rocking rank ceiling, **given a coplanar reflection set** | **6** of 9 (dimensionless) | measured/derived, Notebook §5f | Each sensitivity row is Q̂ ⊗ v, so once the reflection set is coplanar the recoverable rank is capped at 6 **for any rotation axis** — verified against 500 random axes. Adding rocking axes buys nothing, so **no rocking strategy is a lever here**. | None within a coplanar set. Report rank 6 and which components it supports. *Which* reflections are measured is a §2 decision, not this row. |
| Refraction gauge | present | Notebook §3 | A clean uniform strain offset of order 100s of µε across a whole grain is a **reference offset**, not an intragranular field. | Subtract it as a reference, never interpret it as physics. |

**Consequence worth stating on any report:** two of these fail *silently*. The kinematic
boundary and a missing registration both produce a converged, plausible answer. A report that
does not state which side of them it sits on is not characterizing the measurement.

## 2. Configured — set per run, changeable next time

The only tier where "what could be observed differently" has an answer.

| Parameter | Used | Achievable range | Limited by | What changing it would buy |
|---|---|---|---|---|
| **Background / pedestal subtraction** | per reduction | on or off | choice, not hardware | **The single largest effect in this doc set.** A pedestal carrying ≳95% of ∑I dilutes the centroid by 1/(1−f_ped); on raw ID03 frames f_ped = 0.985 gave a **~67× underestimate** of the orientation amplitude. Not a limit — a required step. |
| Rocking-scan range and step | per run | motor-limited | goniometer travel and time | Points across the rocking curve. Too few flattens the moment for a real reason, independent of the pedestal. |
| **Number *and coplanarity* of reflections** | per run | beamline time and accessible geometry | scheduling, and registration feasibility | The tier where the rank ceiling of §1 is actually decided. One reflection gives a projection of the tensor. The **full** tensor needs **≥3 non-coplanar** reflections co-registered voxel-for-voxel; a coplanar set caps at rank **6** no matter how it is rocked. Choosing an **oblique** geometry with symmetry-equivalent reflections reaches rank 9 and keeps \|F\| identical, which removes the intensity problem (Detlefs 2025). |
| **Reflection / channel choice for a two-phase ratio** | per run | which reflections are accessible | structure factors and systematic extinctions | Whether an intensity ratio is a phase fraction **at all**. Pick a channel **forbidden** in the competing structure; if the channel is shared, the ratio's neutral point is neither 0.5 nor constant and single-phase regions land in "mixed" (Notebook §7i). This is a reflection-list decision, not a post-processing one. |
| Detector exposure | per run | detector-limited | readout and dose | Counting statistics per pixel, which sets the χ²/dof the reduction can legitimately claim — **at the measured gain** (§4). Note the trade saturates in one direction: past the slowest vibration period, more exposure buys counts but no further resolution penalty, and shortening it does not recover resolution unless the vibration power is faster than the frame time (Notebook §7f). |
| Spatial sampling | per run | optic-limited | magnification and pixel size | Measured resolution — but see §4: a fitted feature width at the pipeline's own floor measures the pipeline, not the sample. Sample **each channel** on its own merits: a weak channel acquired 40× coarser in space is structurally blind to physics that lives only there (Notebook §7g). |

**Rows deliberately blank.** Detector maximum frame rate, goniometer travel limits, and the
dose at which a given sample starts to damage are not recorded in this doc set. Until filled
in, a report **will not** propose changing exposure or dwell.

## 3. Intrinsic — the sample or the physics forbids it

No configuration helps.

| Question | Why it is not answerable | Distinguish from |
|---|---|---|
| Full deformation-gradient tensor from **one** reflection | One reflection constrains a projection of the tensor, not the tensor. | With **≥3 non-coplanar** registered reflections it becomes a §2 question. Not intrinsic to DFXM, intrinsic to a single-reflection dataset. A *coplanar* set is a separate, harder wall: rank 6 regardless of registration (§1). |
| Intragranular strain field from a **uniform** offset | A uniform offset carries no spatial information about the field by construction. | The spatially varying part is measurable; it is the constant that is a gauge. |
| A phase fraction from a ratio between **shared** channels | If both candidate structures contribute intensity at one channel's measured **Q**, no calibration recovers a fraction from the ratio — the quantity thresholded is not a two-state contrast. Commensurate periods cause it: a supercell of period 2n·c reproduces every reflection of one at n·c, and twinning permutes which variant feeds the shared position. | **Conditional, and testable — compute \|F\|² at the measured Q for every structure *and* variant.** If exactly one structure contributes at each channel, the ratio *is* a phase fraction and this row does not apply. If a forbidden-in-the-other channel is accessible, it is a §2 reflection-choice question, not an intrinsic limit. Only when no such channel exists is this intrinsic to the structure pair (Notebook §7i). |
| Unbiased strain beyond the kinematic boundary | The forward model does not hold there. | Inside the boundary it is recoverable. Report which side you are on — **per reflection** (§1). |

## 4. Derived limits

What follows arithmetically from §1–2. A report may quote these directly.

| Quantity | Limit | From |
|---|---|---|
| Orientation amplitude from an un-subtracted moment | **diluted by 1/(1−f_ped)** — measure f_ped, do not assume | §2 row 1; Notebook §1a |
| Smallest trustworthy feature width | **the pipeline's own resolution floor** | §2 spatial sampling; a fit at the floor is circular and measures the pipeline |
| Achievable χ²/dof | set by the counting statistics **at the measured gain**, not by model quality | §2 exposure; Notebook §7b — one integrating sCMOS measured `var = 2.23·y + 149`, inflating every absolute χ²/dof ~2.2× and making an adequate model (true 1.08) read as rejected (2.6). A χ²/dof far from 1 everywhere is an error-model problem, and the gain is the first term of that model |
| Points per FWHM available to a per-pixel fit | set by the **per-pixel** width, ~2.6× narrower than an integrated/published one | Notebook §7a; dividing the step into a published FWHM overstates sampling by that factor |
| Tensor availability | requires **≥3 non-coplanar** reflections **and** registration metadata; a coplanar set caps at rank 6 | §1 rows 3–4 |
| Recoverability of a resolution loss by shorter exposures | **undetermined from images alone** — set by where the vibration power sits in frequency | §2 exposure; Notebook §7f — needs a spectrum, not an amplitude |

## 5. Did not versus cannot

Skipped on a given run but perfectly possible. These read identically to hard limits.

- **Background not subtracted.** A processing choice with a 67× consequence, not a
  measurement limit. Never report an orientation amplitude from an un-subtracted moment.
- **Single reflection acquired.** A scheduling choice. Report as "tensor not acquired",
  never "tensor not available".
- **Lineshape invariance not tested.** A check not run, not a property of the data. And when
  it *is* run, the alternative must be **asymmetric** — a symmetric one cannot move a centroid,
  so the test reassures without testing anything (Notebook §5i).
- **Detector gain not measured.** One pass over the frames. Until it is done, every absolute
  χ²/dof and error bar is provisional by an unknown factor — not a limit of the data.
- **Flux monitor not logged.** If no monitor column was recorded, an intensity comparison
  across separately-acquired groups is **not recoverable** from the archive — that one is a
  genuine cannot, and it is a question for the data's authors, not an analysis fix.
- **Vibration spectrum not recorded.** Same shape: images cannot supply it. Report "not
  determinable from this archive", never "vibration is not correctable".

---

**Checklist before this file is trusted**

- [x] Every row has a unit or is explicitly dimensionless
- [ ] Every bound in §2 names what limits it — **three rows still blank** (frame rate, goniometer travel, damage dose)
- [x] Nothing in §1 or §3 is phrased as a suggestion
- [x] `Last checked` is within the current run cycle
