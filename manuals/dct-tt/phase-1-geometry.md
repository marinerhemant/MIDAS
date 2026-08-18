# Phase 1 — geometry from the data

**Goal:** effective pixel size, rotation-axis column, detector distance, lattice type and
`λ/2a`, and the ω sign — all derived from the frames, then checked against something you did
not fit.

**Why this phase exists.** On the DCT scan taken end-to-end here, *none* of this was in the
file. The only geometric number the header carried was the sensor pixel, and the true imaging
pixel was a factor **6.65** away. A different, TT, dataset from the same instrument had a
metadata pixel size a factor **2** from the camera actually in use. Neither error announces
itself: they produce complete, self-consistent, wrong reconstructions.

## 1.1 Effective pixel size — from a known length, not from the header

The header pixel is the **sensor** pixel. With any magnifying optic in the path, the imaging
pixel is different and the header does not know it.

Find a physical length in the field of view and measure it in pixels. A slit box is ideal,
because its gaps are set and recorded in millimetres:

```
slit box measured   134.0 x 420.6 px      (50% crossings of the frame-median image)
slit gaps set        0.2199 x 0.700 mm
=> pixel            1.641 / 1.664 um      -> adopt 1.653 um
```

**The check that makes this trustworthy is that the two axes agree independently** — here to
**1.4 %**. One axis alone is a number; two axes agreeing is a measurement. If they disagree by
more than a few per cent, something is anisotropic (binning, a tilted detector) and you should
stop rather than average them.

Alternative known lengths: a fiducial of known size, a translation stage moved a known
distance, or the sample itself if independently measured.

## 1.2 Rotation-axis column

The rotation axis projects to a column on the detector. Find it by the criterion that makes
the physics sharpest rather than by eye:

> The column that makes **Friedel-pair ring radii sharpest.**

For a Friedel pair `(y, z)` and `(y', z')` at ω and ω+180°, the grain-position blur cancels
exactly in `(y+y')/2 − c` when `c` is the true axis column. Scan `c`, compute the ring-radius
distribution, take the minimum width. Here that gave **1016.53** — and the diagnostic that it
worked is that **paired** ring radii came out sharp (widths 0.9–2.9 px) where **unpaired**
radii were broad. That is the algebra behaving as predicted, which is a stronger check than
the minimum itself.

## 1.3 Lattice type and `λ/2a` together

You cannot get λ and the lattice parameter separately from a ring pattern — only their ratio.
Fit them jointly:

```
fit: ring radii -> {lattice type, s = lambda/2a, detector distance}
result: fcc, s = 0.037257, Lsd = 6.775 mm   (6.725-6.826 at delta-chi2 < 9)
        5 rings, 2 free parameters, 0.91 px rms
```

Report **rings used** and **free parameters** every time. The reason is a retraction: an
earlier fit reported "hcp, c/a = 1.856, L = 123 mm" — a small-angle-limit local minimum where
only the *product* `s·L` is determined, so the individual values were meaningless. A
two-parameter fit over five rings at sub-pixel rms is a different kind of claim from a
four-parameter fit over three.

Equally, earlier hcp and simple-cubic "wins" were **overfitting artefacts of
nearest-neighbour scoring** — with enough free lattice parameters, some ring lands near every
observed radius. Score against *all* rings including the ones your candidate predicts and you
do not see.

**Watch for rings you cannot explain.** One ring at R = 1046 px, flat in η with ~1000 net
pairs, sat between (311) and (222) and is unexplained by fcc. A second phase is not excluded.
An unexplained ring is a standing caveat, not a rounding error.

## 1.4 What `λ/2a` does and does not buy you

| Wanted | Needs |
|---|---|
| indexing, orientations, grain map | lattice **type** and `λ/a` only — ✅ available |
| absolute d-spacing, any strain | λ and `a` **separately** — ❌ not available from the pattern |

So: **you can build a complete, validated grain map without knowing the material.** You cannot
report a strain, or any absolute length in the crystal, without naming it. If the material is
unknown, say which quantities in your report are therefore ratios.

Candidate materials can be narrowed by other evidence (attenuation with ω, for example) but
"the strong ω-dependent attenuation argues against a light alloy" is an argument, **not a
measurement**, until it is made quantitative.

## 1.5 ω sign

Two conventions exist and the wrong one mirrors the reconstruction without raising the
residual. `midas_dct_tt.conventions` carries them as named constants rather than bare
literals, precisely because a sign error here is undetectable downstream:

```python
from midas_dct_tt import DCT_OMEGA_SIGN_CCW, DCT_OMEGA_SIGN_AERO
# CCW  = +1  a stage genuinely right-handed about +z
# AERO = -1  a clockwise stage: every recorded omega must be negated
```

Establish it from the stage's documented sense if you can. If you cannot, note that only
`y_sign × ω_sign` is determined by the diffraction (phase 0.4) and carry the ambiguity
forward.

## 1.6 The check that closes the phase

Confirm the geometry against something that was **not** in the fit. Two that worked here:

* **Interplanar angles.** Recovering the fcc {111} angles **70.53°/109.47°** at 5σ from the
  indexed data uses the lattice type but not the ring radii you fitted. That pinned
  `y_sign × ω_sign = +1`.
* **A cross-technique prediction.** For TT, predicting the goniometer tilts of scans you did
  not fit: median residual **0.043°/0.050°** across 74 independent real settings, against a
  random-grain null of 25–40°. Discrimination **985× / 526×**, with **0 of 200** null draws
  beating the truth. That validates the whole chain — orientation convention, reciprocal
  basis, instrument offsets, alignment solution — in one number.

## 1.7 Exit criteria

- [ ] effective pixel from a **known length**, with two independent axes agreeing
- [ ] rotation-axis column, with paired-vs-unpaired ring width as the diagnostic
- [ ] lattice type, `λ/2a`, distance — with **rings used and free parameters quoted**
- [ ] any unexplained ring recorded as a standing caveat
- [ ] ω sign established, or the ambiguity explicitly carried forward
- [ ] one confirmation against a quantity that was not fitted

**Halt** if the pixel size cannot be tied to a known length. Everything downstream — grain
sizes, positions, field magnitudes — is linear in it, and a wrong value is invisible.
