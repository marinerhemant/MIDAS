# NF-HEDM diagnosis reference

Symptom → discriminating test → cause → lever, for near-field HEDM. Read by `beamreport`;
each entry attaches to a symptom the generic diagnostics detect.

**Every entry carries a test that can come back the other way.** An entry that cannot
exonerate the cause it names does not belong here — it turns the report into a machine for
confirming whatever its author already believed. Where an entry's alternative is another
entry, it says which.

Four entries. Three is a working start; this grows the day someone works out what a
strange plot meant, written the same day (`beamreport` SPEC §6).

Provenance for every number below is `LAB_NOTEBOOK.md`, cited per entry.

---

## Beamstop occluding the inner rings

symptom: quality.low_fraction

**Test.** Bin the per-voxel confidence deficit by detector distance. A beamstop occludes a
**fixed angular** region, so the reflections it eats are the innermost ones, and the
deficit **grows as `Lsd` shrinks** — at the near distance the strongest low-index
reflections are the only ones landing inside the stop. If the deficit is flat across
distances, the stop is not the cause and this entry does not apply; look at the reduction
threshold (next entry but one) or at the geometry.

Confirm against the reflection list rather than the image: identify which hkl fall inside
the stop radius at each distance and check that those are the missing ones. **A row
profile cannot see this** — it integrates across the very region that is dark.

**Cause.** The beamstop removes reflections that dominate the pattern, so a voxel cannot
reach high `FracOverlap` no matter how good the orientation is. The ceiling is geometric,
not a fit failure.

**Lever.** Exclude the occluded rings from the reflection list rather than chasing the
geometry, or re-measure at a distance where they clear the stop. Do not raise
`MinConfidence` to hide it — that discards real voxels and leaves the ceiling in place.
Lab Notebook §7h.

## Forbidden reflections left in the denominator

symptom: quality.low_fraction

**Test.** Compute `|F|²` for the declared cell and count reflections with `|F|² = 0`. The
predicted `FracOverlap` cap is `1 − N_zero/N_total`. If the observed maximum confidence
sits at that cap, the cell's own extinctions are the limit. If the observed maximum is
**below** the predicted cap, this entry does not explain the gap and something else is
also wrong.

Measured: the DHCP polytype has **126 of 736** reflections with `|F|² = 0`, capping
FracOverlap at **0.829**; the FCC parent has none, cap 1.000. Predicted 0.596 against a
measured 0.5962 after the correction.

**Cause.** Space-group extinction rules do not see **basis-dependent** extinctions, so a
multi-atom cell generates reflections that cannot exist. They enter the denominator of the
overlap fraction and cap it below 1 regardless of the fit.

**Lever.** Declare the basis with repeated `PhaseAtom` lines so `midas_hkls` computes
structure factors and drops the impossible reflections. Never fix it by lowering
`MinConfidence`. §8l, Lab Notebook §11.

## Beam centre wrong, or β borrowed from another beamtime

symptom: systematic.common_offset

**Test.** Compare the **mean** of the per-voxel (Δy, Δz) offsets against their **scatter**.
A mean much larger than the scatter is a common offset — a geometry error shared by every
voxel. A mean near zero with large scatter is genuine per-voxel position spread, which is
not a bug and points at the per-object entry instead.

Then separate BC from β: fit the offset against `DetZ` across distances. A **constant**
offset is `ybc`/`zbc` at the reference distance; one that **trends with `DetZ`** is the
beam tilt β, which is per-beamtime and must not be inherited.

**Cause.** The beam centre used in reconstruction differs from the true one, or β was
carried over. Borrowing β between beamtimes was wrong by **62×** in y; the beam stripe
itself moved **57 px = 31 µm** between two campaigns on the same detector serial.

**Lever.** Re-measure BC from that beamtime's own `DetZBeamPos` scan (§6), emit BC per
distance from the fitted β (§6f), and re-run. Do not widen `BCTol` to absorb it — that
buries a measurable quantity inside a fit that is worse at determining it. Hard rules
11–13, §6d, §6f.

## Refinement parameters resting on their tolerance box

symptom: bound.pileup

**Test.** For each refined parameter, compare its fitted value against its declared
tolerance. Then **widen the box and re-run**: if the parameter moves substantially, it was
bound-limited and the previous value was the box edge, not a measurement. If it stays put,
the bound was not binding and this entry does not apply.

**Cause.** `TiltsTol` defaults to 0.05° in the code, but `ty`/`tz` start at 0 and may
genuinely be ~0.5°, so the tanh box makes the true value unreachable and the optimiser
parks against the wall — at high confidence, because confidence is a plateau (hard rule
14) and does not fall off when a tilt is wrong by a few tenths of a degree.

**Lever.** Set `TiltsTol 1` for calibration, iterate with `NumIterations` **inside one
invocation**, and never re-seed a refinement with its own output — `TiltsTol` is relative
to the current seed, so re-seeding ratchets the tilts ~1°/pass while confidence stays high.
§7c, hard rule 15.
