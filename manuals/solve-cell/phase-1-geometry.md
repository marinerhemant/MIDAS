# Phase 1 — geometry, before any cell

**A cell can be no better than the geometry it was fitted on** — but the two kinds of geometry
error do completely different damage, and knowing which is which saves days.

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

## Detector rate, before you trust any width or intensity

Check the peak count rate before quantifying anything from peak shapes or strong-reflection
intensities. Measured on one sample: cores at **6.8–7.2 Mcounts/s/pixel, 1256 px above
2 Mcps** — 3–7× past validated Pilatus/CdTe rate correction. Widths inflated with brightness
(Spearman +0.599, p = 0.04, vanishing below 1e5 counts).

## tx, and what powder cannot give you

`tx` (rotation about the beam) is an **exact gauge freedom of single-panel powder
calibration** — swept 0–20° at tz = 14°, 2θ is unchanged. So a powder calibrant cannot
determine it, and in MIDAS v2 no pipeline refines it (`compat/from_v1.py:49` hardcodes
`refined=False`). Two consequences:

* if you need tx, get it from **Friedel ω-splitting**, not from powder:
  `midas_calibrate_v2.friedel` (validated against planted tx, 0–4°, to < 0.25°);
* if a grain-refined tx is composed into a `Parameters.txt`, the azimuths φₖ must be rotated
  with it — worth 6.5–20.4 µε if skipped.

## Before you leave this phase

* the beam centre has a null attached, and it passed;
* the row convention is settled **from the data** and applied once;
* the scale is checked on a calibrant (a good one: 13/13 rings at certified d, rms 0.0116 %);
* the count rate is inside the detector's validated range, or the limitation is written down;
* **no solution file predates a geometry edit.** Measured: `geom.py` was edited 67 s after a
  solution was written and only 8 of its 12 reflections still passed. Re-run, do not reuse.
