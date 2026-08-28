# NF-HEDM diagnosis reference

Symptom → discriminating test → cause → lever, for near-field HEDM. Read by `beamreport`;
each entry attaches to a symptom the generic diagnostics detect.

**Every entry carries a test that can come back the other way.** An entry that cannot
exonerate the cause it names does not belong here — it turns the report into a machine for
confirming whatever its author already believed. Where an entry's alternative is another
entry, it says which.

Ten entries; the last five came out of the three 20-ID HT-HEDM campaigns and are marked
**[20-ID]**. This grows the day someone works out what a strange plot meant, written the
same day (`beamreport` SPEC §6).

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

## Grid points masked by the wrong tomography pixel

symptom: systematic.common_offset

**Test.** Re-run `filter_grid_by_tomo` with `legacy_c_parity=False` and count how many grid
points change status. Three independent one-pixel defects sat in `sample_tomo`
(`tomo_filter/filter.py`), all found 2026-08-23 and each documented in its docstring:

1. the row flip was `n - y_pos`, which never reads row 0 and indexes **past the end** at
   `y_pos == 0` (an out-of-bounds heap read in `filterGridfromTomo.c:42`, an `IndexError`
   in the Python transcription);
2. the Python truncated `x/px` and then added `n // 2`, where the C truncates the sum —
   measured on a 200k-point grid over a 1 mm sample at 1.5 µm, **75 % of grid points
   landed on a different pixel than the C**;
3. the Python used integer `n // 2` where the C uses `(double)n / 2`, which puts the origin
   on the *edge* of the centre pixel instead of its centre for odd `n`.

If the two modes agree on your grid, this entry does not apply — every existing test used
integer coordinates, which is the one case where the conventions coincide, and that is why
these survived.

**Cause.** The tomography mask is displaced by up to one pixel relative to the grid, so a
band of grid points at the sample edge is included or excluded wrongly. On a 1.5 µm grid
that is a 1.5 µm boundary error, invisible in the reconstruction and visible only as a
sliver of edge voxels that never reconstruct.

**Lever.** Pass `legacy_c_parity=False` for new work. The default stays `True` so existing
reconstructions reproduce; both modes now drop `y_pos == 0` instead of reading out of
bounds, which is the only behaviour change in parity mode that is not a bug fix.

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

## The background reads as signal, and nearly every voxel indexes **[20-ID]**

symptom: quality.implausible_coverage

**Test.** `np.unique(frame)[:8]` and `frame.max()` on one raw frame. Multiples of 64 with
a maximum near 65472 mean 10-bit data stored ×64; a gap of 2–4 with a maximum near 4092
means 12-bit unscaled. Compare that against the `PixelScale` in the paramfile. **The
reader also tests this itself and warns in both directions** — if it is quiet, the setting
matches the data and this entry does not apply; look at the threshold in σ instead (§8k).

**Cause.** `PixelScale` wrong by 64. The encoding is a property of the **scan**, not of
the detector: `nfdev_jul26` is ×64 while `NF_Au_cube_0802` and the SS316L NF scan are
unscaled *on the same detector serial*, and the SS316L tomography taken the same day is
×64 again. A wrong scale turns the §5d production threshold of "2 counts" into 128 counts,
which sits above the pedestal, so the pedestal itself is admitted as signal.

**Lever.** Set `PixelScale` and let the reader divide — never divide in an analysis
script, or the paramfile and the plots disagree about what a count is. Never inherit the
value from another scan. §3h, §5d, §10f, Lab Notebook §8b.

## A fitted `Lsd` or lattice parameter disagrees wildly with the other distances **[20-ID]**

symptom: geometry.inconsistent

**Test.** Before trusting anything fitted to a radial profile, ask whether rings exist at
all: measure the **illuminated sample width** (equivalent top-hat of the absorption
profile in the beam stripe) against the **innermost ring spacing**. Rings survive only
while width < spacing. On `nf_sampleD` the sample is 247 µm wide against a 111→200 spacing
of 225 px — 2.0× — so there are no rings to fit. If width ≪ spacing, rings are real and
this entry does not apply; suspect BC or β (the entry above).

A second control, when two layers were measured at one distance: their ring radii must be
identical. If they are not, you are fitting sample structure, not a powder pattern.

**Cause.** An NF spot lands at *grain position* + `Lsd·tan(2θ)·d̂`. The first term is why
NF resolves grains at all, and it is also why NF data is not a powder pattern — every ring
is convolved with the illuminated width. On a coarse-grained sample the "rings" are an
artefact of that convolution and any `Lsd` fitted to them is meaningless.

**Lever.** Get `Lsd` from **spot triangulation** (§6i), not from a radial fit. This was
the root of three successive wrong SS316L answers (δ = −937, then −1550, then "27.6 %
off"), each one a new hypothesis layered on the same unexamined assumption. §5e,
Lab Notebook §8c.

## `fit_axis(...).is_reliable` returns False, or the axis is off by ~100 px **[20-ID]**

symptom: geometry.axis_unreliable

**Test.** Re-run the shadow tracker with `band_frac=0.70` and compare the recovered axis
against the default `band_frac=0.30`. If the two agree, the default was fine and this
entry does not apply. At 20-ID they do not: 0.30 puts the axis **+100 to +130 px** off and
returns a clipped amplitude, while 0.70 reproduces the known Au axis to **0.41 px**.

**Cause.** The default band is a fraction of the beam stripe, and 0.30 lets the tracker
wander into the beam's dim wings where the transmission profile is not the sample's
shadow. The 20-ID stripe is wider relative to the sample than 1-ID's, so the default
under-constrains it.

**Lever.** `band_frac=0.70` at 20-ID, and **branch on `is_reliable` rather than
overriding it** — it is correctly False in the failing configuration. If it stays False at
0.70, the specimen may simply be extended, in which case the moving-shadow method does not
apply at all: an irregular specimen's deepest-dip centre does not trace a rigid sinusoid
(shadow width swung 56→886 px with ω on `nf_sampleD`) and no setting rescues it. §6e,
Lab Notebook §8e.

## Triangulated distances split between y and z **[20-ID]**

symptom: geometry.inconsistent

**Test.** Solve the distance separately from the y and the z spot coordinates and compare.
A y-z split of a few tens of µm is the healthy case (25/58/54 µm on `NF_Au_cube_0802`). A
split of ~140 µm against ~60 µm means the point-source assumption is straining. Estimate
the perturbation directly as (sample half-width)/(typical spot radius): **3 %** for a
70 µm cube, **11 %** for a 247 µm specimen. If the sample is compact and the split is
still large, this entry does not apply — check BC first.

**Cause.** `triangulate` models the sample as a point at BC. On a wide sample each spot is
displaced by its grain's own position, which the point model absorbs into the distance.

**Lever.** Treat the triangulated `Lsd` as a **seed, not the answer**, and refine it. On
`nf_sampleD` triangulation was **211 µm** off; after refinement δ landed **6.8 µm** from
the previous campaign's value. Triangulate → refine → then quote, and check the nulls
(ω-shuffled and position-scrambled) rather than the residual. §6i, §6i-ter,
Lab Notebook §8d, §8h.

## Every voxel comes back as its own grain, or grain radii are absurd **[20-ID]**

symptom: grains.oversegmented

**Test.** Compare `EdgeLength` against `GridSize` in the paramfile. `mic2grains
-doNeighborSearch 1` merges voxels closer than `2·TriEdgeSize`, while neighbours on the
grid sit `GridSize/2` apart — so whenever `EdgeLength ≪ GridSize` the merge threshold is
far below the voxel spacing and nothing can ever merge. If `EdgeLength == GridSize` and
the map is still oversegmented, this entry does not apply; the orientation field itself
may be noise (run the neighbour-vs-random test, `RUNBOOK.md` §R2b).

**Cause.** `EdgeLength` is the **probe triangle** edge and is an independent, supported
knob — small probes on a coarse grid are intentional (`hex_grid/grid.py:97-153`). What
breaks is only the *grain segmentation* built on top, and the radii it reports describe
the probe triangle rather than the cell. A measured 500× discrepancy.

**Lever.** Report grain **counts**, not `mic2grains` radii, unless `EdgeLength ==
GridSize`. Segment by neighbour **misorientation** instead —
`midas_stress.misorientation`, which is the maintained implementation; do not write a new
one. Note also that `GridSize` is the triangle **edge**, so voxel pitch is `GridSize/√3` —
treating it as the pitch overstates grain diameters by 1.73×. §8a, §10e.
