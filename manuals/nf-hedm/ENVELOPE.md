# NF-HEDM — measurement envelope

**Instrument:** 1-ID near-field HEDM (TIFF-per-frame) · **and** 20-ID-D HT-HEDM
(Bluesky/HDF5) — the spine's two scope-table configurations. Rows re-derived at 20-ID are
marked **[20-ID]**; an unmarked bound is still 1-ID's.
**Last checked:** 2026-08-28 · **Owner:** Hemant Sharma (hsharma@anl.gov)

> Part of the **NF-HEDM doc set**. Spine: [`README.md`](README.md). Contract: `~/opt/beamreport/DOCS_SPEC.md` §6 (separate repo, not under `$MIDAS`).

What this measurement can and cannot determine, and which of those is changeable. Read it
before promising an answer, and before suggesting a different measurement.

> **Not the scope gate.** The scope gate says whether these recipes apply to your data.
> This says whether the measurement can answer the question. §1–§4 below were established
> at 1-ID; 20-ID HT-HEDM runs through the same pipeline (spine §3h), and the rows that
> have since been re-derived there are marked **[20-ID]**. An unmarked bound is still
> 1-ID's — inherit it as a starting point, not as a result.

---

## 1. Fixed — cannot change this cycle

No suggestions here. State the consequence and the substitute.

| Property | Value | Provenance | What it makes unobtainable | Substitute |
|---|---|---|---|---|
| Beamstop occlusion | stop radius per distance. Measured **[20-ID]** on `nfdev_jul26`: circular, **R ≈ 1100–1240 px = 600–680 µm** | [`DIAGNOSIS.md`](DIAGNOSIS.md), Lab Notebook §7h | Reflections falling inside the stop are removed. A voxel **cannot reach high `FracOverlap`** no matter how good its orientation is. **The ceiling is geometric, not a fit failure.** | Exclude the occluded rings from the reflection list, or re-measure at a distance where they clear the stop. **Never** raise `MinConfidence` to hide it — that discards real voxels and leaves the ceiling in place. The later `NF_Au_cube_0802` scan is *not* beamstop-limited, which is why it, not the July scan, is the commissioning reference (`RUNBOOK.md` §R2c). |
| Structure-factor extinctions | phase-dependent | [`DIAGNOSIS.md`](DIAGNOSIS.md), Lab Notebook §11 | A multi-atom cell generates reflections that **cannot exist**. They enter the denominator of the overlap fraction and cap it below 1 regardless of the fit. Measured: DHCP has 126 of 736 reflections with \|F\|² = 0, capping FracOverlap at **0.829**; the FCC parent has none, cap 1.000. | Declare the basis with repeated `PhaseAtom` lines so `midas_hkls` computes structure factors and drops the impossible reflections. Never lower `MinConfidence`. |
| Single-crystal calibrant degeneracy | — | README §7b(2), Lab Notebook §7d | `-multiGridPoints` **cannot break the degeneracy on a single-crystal calibrant**: N voxels of one grain give one orientation's worth of constraint. | If the calibrant contains two particles, deliberately draw voxels from both. Otherwise the step reduces to a documented negative. |

**Consequence worth stating on any report:** a `FracOverlap` ceiling below 1 is often
*expected* for the phase being measured. Quoting a confidence distribution without stating
the phase's own cap invites a reader to read a geometric ceiling as a bad reconstruction.

## 2. Configured — set per run, changeable next time

The only tier where "what could be observed differently" has an answer.

| Parameter | Used | Achievable range | Limited by | What changing it would buy |
|---|---|---|---|---|
| **Grid spacing** | per run (µm) | — | acquisition and reconstruction time | Spatial resolution of the orientation map. Two features closer than one grid step are not separable. |
| **Number of detector distances** | per run | stage travel | detector translation range | Constraint on the voxel-to-spot assignment. One distance is weaker than several. |
| ω step and range | per run | — | acquisition time | Angular sampling; how well a voxel's orientation is pinned. |
| Reduction threshold | per run | — | measured from the data | Which spots survive into the fit. Set too high it starves voxels; the beamstop entry above is the failure it is most often mistaken for. |
| Reflection list / rings used | per run | — | beamstop, detector extent, phase | Removes geometrically impossible or occluded reflections from the denominator. |
| `MinConfidence` | per run | — | — | **A reporting threshold, not a lever.** Raising it hides ceilings rather than fixing them. Listed here so it is explicitly ruled out. |

**Rows deliberately blank.** Detector maximum frame rate, stage travel limits, and the dose
at which a given sample starts to damage are not recorded anywhere in this doc set. Until
filled in, a report **will not** propose changing exposure, distance count, or total dwell.

## 3. Intrinsic — the sample or the physics forbids it

No configuration helps.

| Question | Why it is not answerable | Distinguish from |
|---|---|---|
| Elastic strain per voxel | NF recovers **orientation and position**. Strain needs the lattice-parameter precision a far-field geometry provides. | FF-HEDM recovers per-grain strain tensors. Combine the two rather than pushing NF at it. |
| Orientation of a voxel the beam did not illuminate | No signal. | Low confidence from a poor fit looks similar in a map and means something else entirely. Check illumination coverage before interpreting a low-confidence region. |
| Breaking a single-crystal calibrant's orientation degeneracy with more voxels | §1 row 3 — N voxels of one grain carry one orientation's worth of information. | More voxels **do** help when the field contains more than one grain. |
| **The handedness of ANY orientation map, from the data alone** | A global mirror maps a consistent solution to another consistent solution. Confidence, grain statistics, distances and positions are all invariant under it, so nothing downstream flags it. **No measurement inside a reconstruction can supply the ω sign** — at either beamline. | This is **not** low confidence and **not** a fit failure. It is also not a reason to stop: the sign comes from *outside* the data, and at both 1-ID (`.par` field 9) and 20-ID (**[20-ID]** instrument scientist, 2026-08-28) it has been supplied. On a third beamline, get it before you reconstruct. |
| **Absolute `Lsd` on a motor scale** **[20-ID]** | Only ΔD between successive positions was supplied for `NF_Au_cube_0802`; the nominal `nfz` values are unknown. Triangulation recovers the *differences* and the sample-to-detector distances self-consistently, not their offset against the motor readback. | A δ quoted between two campaigns **is** meaningful (−837.9 vs −837.7 µm, 0.2 µm). A δ quoted against a motor scale is not anchored. Ask the beamline for the nominals; this is §5, not §3, the moment they arrive. |
| **Beam energy, from the data alone** **[20-ID]** | The HDF5 carries no energy and there is no `fastsweep_Emon.txt` equivalent. The Bluesky log prints a *static* foil-wheel table and never records which foil was selected. | 63.314 keV is **confirmed** for `nfdev_jul26` and `bt_20id_jul26b` — by asking, not by measuring (§3h). For a new 20-ID beamtime the answer is again unobtainable from the files, so ask early: the wavelength sets every ring radius. |

## 4. Derived limits

What follows arithmetically from §1–2. A report may quote these directly.

| Quantity | Limit | From |
|---|---|---|
| Smallest separable feature | ≈ one grid step | §2 grid spacing |
| Maximum achievable `FracOverlap` | **phase-dependent, often < 1** | §1 rows 1–2; compute from the reflection list, do not assume 1.0 |
| Confidence floor that means "real" | not a fixed number | must be read against the phase's own cap, not a universal threshold |

## 5. Did not versus cannot

Skipped on a given run but perfectly possible.

- **Basis not declared.** A parameter-file omission, not a limit. Declaring `PhaseAtom`
  lines recovers the cap. Report as "not declared", never as "phase limitation".
- **Single distance measured.** A scheduling choice. More distances are available.
- **Occluded rings left in the reflection list.** Fixable in the list, not in the geometry.
- **20-ID map handedness** **[20-ID]** — *was* listed in §3 as unobtainable and **is now
  resolved** (`aero`, negated; instrument scientist, 2026-08-28). Kept here as the worked
  example of the distinction this file exists to draw: it was never physically
  unobtainable, only unobtainable *from the data*. The fix was a question, not a
  measurement. Ask before you assume a bound is intrinsic.
- **20-ID tilts** **[20-ID]** — unconstrained on the campaigns run so far, not
  unconstrainable. Two good SS316L refinements disagree on the sign of `ty`; Au_0802's
  zeros mean "the refinement left them alone". A calibrant with more than one grain, or a
  multi-point refinement drawn from distinct grains, addresses it (§7b(2)).

---

**Checklist before this file is trusted**

- [x] Every row has a unit or is explicitly dimensionless
- [ ] Every bound in §2 names what limits it — **three rows still blank** (frame rate, stage travel, damage dose)
- [x] Nothing in §1 or §3 is phrased as a suggestion
- [x] `Last checked` is within the current run cycle
