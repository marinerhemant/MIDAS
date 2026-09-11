# Phase 1 — geometry, before any cell

**A cell can be no better than the geometry it was fitted on** — but the two kinds of geometry
error do completely different damage, and knowing which is which saves days.

## 0. From raw frames to g-vectors — the loop, broken on purpose

A Friedel beam centre needs spots; spots need a background model; a background model needs a
geometry. Break the loop in this order, and re-run once:

1. **Seed** the centre from the header with `poni_file_to_row_col(poni)` — not `poni_file_to_bc`
   (README trap 3) — flipped if your reader disagrees with the calibration's row order. It is only a
   seed: a tilted PONI names the point of normal incidence, not the beam centre (~15 px apart on the
   La3Ni2O7 Pilatus, Rot ~ 0.006/0.003 rad).
2. **Ingest** on the seed with the defect doc set's calling contract
   (`manuals/defect/phase-1-ingest.md`): `live_frames`, `detector_angle_maps`, mask, background,
   `find_blobs_3d`, fractional ω mapped through the live indices, `qlab_to_qsample`, powder flag.
3. **Measure** the centre from Friedel pairs — unseeded, or seeded with a WIDE search (below).
4. **Re-ingest** on the measured centre. Everything downstream uses those spots.

On the two delivered La3Ni2O7 positions step 3 recovered the recorded centre unseeded: 2604 (row
810.24, col 737.38) from 71 pairs, S5 (810.47, 737.16) from 51.

## The one distinction that matters

| parameter | effect on d | can it change an axial ratio (c/a)? |
|---|---|---|
| distance (Lsd) | scales every d equally | **NO** — exactly common-mode |
| wavelength λ | scales every d equally | **NO** — exactly common-mode |
| beam centre | azimuth-dependent | **YES** |
| detector tilts | azimuth-dependent | **YES** |

So: if `c/a` is wrong, refining the distance will never fix it. If `a` and `c` are both off by
the same factor, suspect the scale and go to a calibrant. Measured on one dataset,
`c/a = 5.3254 vs 5.3255` under a ∓0.5 % change in Lsd·λ — six significant figures of nothing.

Measured sensitivities on that same dataset, for scale: `d(c/a)/d(bc_col) = +0.00074 /px`,
`d(c/a)/d(bc_row) = +0.00041 /px`. A documented 3 px beam-centre ambiguity moved the inferred
pressure by **0.1 GPa**. Do not hand-wave this — propagate it.

## Get the beam centre from the data, not from the header

```python
from midas_calibrate_v2.friedel import beam_centre_from_pairs
```

Friedel pairs give a centre that is testable against a null: **130 antipodal matches against a
null median of 10** is evidence; **110 against a null median of 63 (1.75×)** is not, and that
second case was a sample whose geometry never did settle. Always report the null.

**A low pair count with p = 0 is a spurious optimum, not a measurement.** The search is local
around its seed. Seeded more than 100 px off with the default `search_px=30`, it returned **7 pairs** on
2604 — sitting exactly on the search boundary, 30.18 px from the seed — against **71** unseeded, and
**10** against **51** on S5; **both spurious answers reported `p_value = 0.0`.** The p-value tests the
optimum it found, not whether it found the right one. Run unseeded, or widen `search_px` until the pair
count stops rising; distrust an optimum on the search edge; compare pair counts, not p.

**A first look that needs no fit.** Friedel-equivalent reflections must come out at the SAME
d-spacing. Two spots that should be a Friedel pair but give different d mean the centre is wrong. That
is how the La3Ni2O7 row flip was caught by eye: 8.623 Å and 7.692 Å from a pair that had to be equal,
which put the working centre 57 px off in row.

## The trap that declares itself nowhere

**A delivered `.poni` may be ROW-FLIPPED relative to the image as your reader returns it.**
Measured: a Friedel centre at as-read row 810.345 against the PONI's `getFit2D` row 867.784 —
**+57.44 px wrong as-read, 0.13 px when flipped**.

**Nothing in the metadata records this.** In particular `Detector_config: {"orientation": 3}`
does NOT declare it: orientation 3 (BottomRight) is pyFAI's DEFAULT and is geometrically
identical to 0 — both give p1 increasing down rows, i.e. **no flip**. TIFFs usually carry no
Orientation tag (274) either. It is knowable only from the data.

**The rule that keeps it fixed:** flip the image **once, on load**, and work in one row order
for the rest of the run. Convert as-read coordinates once (`row_flipped = N_ROWS - 1 - row`).
A per-call conversion is a per-call chance to forget. Verify from the data, not the header:
the correct convention minimises the azimuthal modulation of a ring's 2θ (measured, 0.079°
flipped against 0.700° as-read).

**Use a statistic that can fail.** This doc set's own test harness first judged the flip by sector
centroids inside a ±0.25° window, which can never show more than 0.5° of modulation — so two centres
120 px wrong both scored ~0.19° and it "chose" one. A bounded estimator giving similar values for every
candidate is a rail, not a verdict. Judge by the distance to an unseeded Friedel centre.

That check is packaged: `midas_calibrate_v2.poni_check.check_poni_against_friedel` takes the Friedel spot
list and the PONI point, finds unseeded and wide-seeded Friedel centres, and decides each axis as given,
flipped, swapped or **undecidable**, with a tolerance widened by the PONI's own tilt offset. It deliberately
does not use ring sharpness: on the two delivered La3Ni2O7 positions ring sharpness picked a reading 169.6 px
(2604) and 130.8 px (S5) from the Friedel centre, while Friedel distance picked the row-flipped reading at
15.6 / 23.7 px — the expected point-of-normal-incidence offset. A column flip is undecidable when the centre
sits near the middle column (737 of 1475 here), and it says so.

## Detector rate, before you trust any width or intensity

Check the peak count rate before quantifying anything from peak shapes or strong-reflection
intensities. Measured on one sample: cores at **6.8–7.2 Mcounts/s/pixel, 1256 px above
2 Mcps** — 3–7× past validated Pilatus/CdTe rate correction. Widths inflated with brightness
(Spearman +0.599, p = 0.04, vanishing below 1e5 counts).

## The a/b floor lives on the SAMPLE's frames, not the calibrant's

Any cell-SHAPE result — an a/b splitting, a γ shear, δ — is bounded by how much the geometry shears on
its own, and that is the cos 2η of a ring that should be circular.
`midas_calibrate_v2.ring_anisotropy.ring_harmonics(eta_deg, q)` returns `A2/q0`, which IS the equivalent
δ, against an azimuth-permutation null (`A1/q0` is a beam-centre error).

**Measure it on a ring on the same frames as your reflections.** The calibration's distortion model
absorbs the calibrant's own cos 2η, so a separate calibrant exposure bounds the calibration, not the
sample. On La3Ni2O7 in a DAC the separate CeO2 exposure gave a δ floor of 0.0218 %; the **on-frame
diamond-anvil rings gave A2/q0 = 0.142 %** — about **7× larger**, the size of the 0.108–0.149 % splitting
being reported, which was refuted as a material property on that evidence. In a DAC the anvils ARE the
on-frame calibrant. One raster position may hold too few anvil spots, or too clustered in azimuth, to
bound it (`determined=False`); pool the raster.

A geometric systematic is coherent across every subset of the data, so a sign test or a
disjoint-subset test cannot catch it.

## tx, and what powder cannot give you

`tx` (rotation about the beam) is an **exact gauge freedom of single-panel powder
calibration** — swept 0–20° at tz = 14°, 2θ is unchanged. So a powder calibrant cannot
determine it, and in MIDAS v2 no pipeline refines it (`compat/from_v1.py:49` hardcodes
`refined=False`). Two consequences:

* if you need tx, get it from **Friedel ω-splitting**, not from powder:
  `midas_calibrate_v2.friedel` (validated against planted tx, 0–4°, to < 0.25°);
* if a grain-refined tx is composed into a `Parameters.txt`, the azimuths φₖ must be rotated
  with it — worth 6.5–20.4 µε if skipped.

## Tilts from the crystals, when these frames carry no calibrant

In a DAC the rings on the sample frames make a poor calibrant (next section), and a separate calibrant
exposure bounds the calibration rather than the sample. The crystals carry two constraints no external
standard improves on: **α = β = 90° exactly** for a layered crystal (c ⟂ ab in the parent and the
orthorhombic child), with γ left FREE because γ is the observable; and **one detector, several domains** —
one shared geometry, independent orientations and cells.

```python
from midas_defect.selfcal import CrystalSpots, selfcalibrate_from_crystals
res = selfcalibrate_from_crystals([CrystalSpots(row, col, omega_deg, hkl, U, a, b, c), ...], geom)
```

It refines the shared centre, distance, `ty` and `tz`, and per domain U, a, b and γ with c HELD — c sets the
length scale, which is otherwise degenerate with the distance; `tx` is left out (next-but-one section). The
hkl assignment is fixed per call: re-match on the refined geometry and repeat until it stops changing.
**The check is the residual, never the angles** — α and β are pinned by construction. On 2604, two domains:
total tilt **0.402°** from the crystals alone against the PONI's independently calibrated **0.412°**; rms
|ΔG| **0.00194 → 0.00105 Å⁻¹**; radial systematic on a/b-BLIND families **0.344 % → 0.129 %**; the (21L)
anomaly that had been read as a 0.79 % splitting **+0.218 % → −0.055 %**. The family systematic left
(0.129 %) was still the size of the splitting being sought, so no splitting was quoted from 2604.

## The powder lever on a DAC pattern — pulled, and it does not work

Recorded so it is not tried again without a reason. Fitting the on-frame gasket and anvil rings, a free
distance with a free 2θ per ring is **exactly degenerate** (Lsd ran to 1e63 mm at zero residual); an
intensity-weighted ring centroid is dragged by crystal reflections sitting on the rings (a 0.872° tilt where
PONI and crystals agree on ~0.41°); and done properly the ring residual still could not tell four candidate
geometries apart (0.0451–0.0478° rms against a 0.048° ring floor). The rings did constrain the centre, but
centre and tilt are nearly degenerate for rings and the column moved 1.9 px with the tilt assumption.
Numbers: `manuals/calibrate-integrate/ENVELOPE.md`. Tilts from the crystals; centre from Friedel pairs.

## tx and wedge over a short rotation — force the fit, do not free it

"Does freeing tx improve the fit?" is a weak test over a short rotation: across S5's 12° of ω about 80 % of
any tx is absorbed by a compensating change of orientation (to a **0.021°** residual), so a free fit hides tx
rather than revealing it. tx is also exactly the rotation-axis tilt about the beam — the two models differ by
one fixed rotation the orientation absorbs (residual 1.4e-16 Å⁻¹) — so they are one parameter, not two.

Invert the question. ASSERT the hand index, choose targets by |G| only (which tx cannot move: 4e-5 Å⁻¹ at
1°), and at each value of the scanned parameter find the orientation that explains the most targets; run the
same search with the parameter pinned at 0. On S5 that force-fit was **flat — 9 of 16 targets at every tx
from −3° to +3°, and at every wedge from −4° to +4°**; wedge is not absorbable, so that scan had real power.
The other 7 sat a median 3.84° from the nearest prediction, far beyond anything geometry produces: **a
different grain, not a mis-modelled one.** Measure tx from Friedel ω-splitting (above), where the orientation
cannot absorb it. (Nickelate project `step36_force_fit.py`, `step38_force_wedge.py`, `RUNNING_LOG.md`.)

## Before you leave this phase

* the beam centre has a null attached, and it passed;
* the row convention is settled **from the data** and applied once;
* the scale is checked on a calibrant (a good one: 13/13 rings at certified d, rms 0.0116 %);
* the count rate is inside the detector's validated range, or the limitation is written down;
* **no solution file predates a geometry edit.** Measured: `geom.py` was edited 67 s after a
  solution was written and only 8 of its 12 reflections still passed. Re-run, do not reuse.
