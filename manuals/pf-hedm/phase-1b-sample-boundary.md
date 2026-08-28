# Phase 1b — The sample boundary: find it from the spot-count sinogram

> Part of the **pf-HEDM doc set**. Spine: [`README.md`](README.md).
> Numbered with phase 1 because it is a **geometry** question. It is *run* later — it needs
> the peak search, so it becomes available after `transforms`, and before indexing.

When the rotation axis sits near an edge of the sample — deliberately, or because the
sample is smaller than the scanned field — part of the voxel grid is **vacuum**. You need
to know which part, because every per-voxel statistic you report is otherwise averaged over
material that is not there.

**This phase exists because the two obvious ways to find the boundary are both wrong**, and
both produce a confident, plausible answer.

## 1b.1 Trap 1 — the completeness map cannot find the edge

**Vacuum voxels are not empty.** A vacuum voxel shares beam lines with material further
along the same ray, **inherits that grain's orientation**, and scores a completeness of
about **0.92**.

Measured on the reference campaign: the completeness floor over the whole grid was
**0.445**, with **nothing below 0.40** — which reads as "material everywhere" and was
initially read exactly that way. After masking the vacuum by the boundary found in §1b.3:

| region | median completeness |
|---|---|
| material | **1.0000** |
| vacuum | 0.9219 |

The vacuum is not a hole in the completeness map. It is a slightly dimmer region of a map
that looks full.

## 1b.2 Trap 2 — the disc in the completeness map is scan geometry, not the sample

There *is* a circular falloff in the completeness map, and it is not the sample. A voxel at
radius `r` from the rotation axis is illuminated only where `|r·cos(ω − α)| ≤ S` for a
half-scan-span `S`, so the illuminated fraction is

```
f(r) = 1 − (2/π)·arccos(S/r)      for r > S
```

That null explains **R² = 0.92** of the observed radial profile (with a 0.63 µm centring
offset). Run it before interpreting any radial structure.

**It also gives you the discriminating test:** the scan-geometry null predicts a **circle**;
a real straight sample edge predicts a **chord**. If your falloff is circular and centred
on the rotation axis, you are looking at the scan, not the sample.

## 1b.3 The method that works — the support of the spot-count sinogram

At rotation ω the beam at scan position `s` crosses the **whole sample** — millimetres,
tens of grains — so every `(s, ω)` cell yields many spots **unless `s` falls outside the
silhouette, where it yields exactly zero.**

So the **support of the `(s, ω)` spot-count histogram is the sample silhouette.** It needs
no indexing, no grains and no reconstruction: it comes straight from
`InputAllExtraInfoFittingAll*.csv` after peak search.

```python
# Read the column names from the file's own '%' header line — do not hard-code indices.
import numpy as np, pandas as pd
ia = pd.read_csv(spot_csv, sep=r"\s+", comment="%", header=None)
# omega is column 3 on the reference campaign; CONFIRM it against the header line.
counts = np.histogram2d(scan_index, ia[3], bins=(n_scans, n_omega_bins))[0]
```

A **straight edge appears as a wedge** with its apex at `(φ, d)` — the edge normal
direction and its distance from the rotation axis.

### The control that makes it trustworthy

The wedge appears **twice, 180° apart, with the sign of `s` flipped**, because
`û(ω+180°) = −û(ω)`. **Check for the partner.** On the reference campaign it was found at
**180.6°** separation. That pairing is the strongest single control available here.

Three more that were run and passed:

| control | what it rules out |
|---|---|
| the deficit is confined to the outermost `\|s\|` | a dropped frame or a bad ω range — either would empty all scan positions, not just the outer ones |
| the wedge opening implies a sane sample length `L` | a fit that happens to be a wedge but is not geometric |
| **it vanishes in the neighbouring scan set** after the sample translated 15 µm | anything fixed in the instrument frame. The feature moves with the sample |

## 1b.4 What this resolved, and what it did not

On the reference campaign, two **grain-free** routes agreed to **1.54 µm**:

| route | edge distance from rotation axis |
|---|---|
| spot-count sinogram wedge, `y(0)` | **+14.50 µm** |
| completeness-null residual, `y(0)` | **+16.04 µm** (rms 2.51 µm over 51 rows) |

and a sign test settled which side the vacuum is on:

| region | mean completeness-null residual | completeness |
|---|---|---|
| `y > +14.5` | **−0.0560** | 0.808 |
| `\|y\| ≤ 14.5` | +0.0117 | 0.988 |
| `y < −14.5` | +0.0251 | 0.889 |

**Vacuum is at `y > ~+15 µm`.** Consequence: 560 of 2601 voxels (**21.5 %**) are vacuum,
and the dominant grain shrinks 1255 → 1012 voxels (40.0 → 35.9 µm) while the other four are
untouched — inheritance comes from the *largest* grain, which is exactly what beam-line
sharing predicts.

### ⚠ Quote the distance. Never quote the tilt.

Three methods give **+2.36°** (sinogram), **−8.48°** (completeness) and **+4.80°** (tomo).
The tilt is **unresolved**. The distance is the reportable quantity.

*(A correction that is part of the record: the figures for this campaign were first drawn
with the tilt at −2.36°. Under the validated convention `s = +x sinφ + y cosφ` it is
+2.36°. Only the sign as drawn was wrong; the distance never moved.)*

## 1b.5 RETRACTED — do not use the tomo intensity map as a material map

A mid-thread claim that the edge was "12 µm off" was **wrong**, and it is worth knowing how.
It used the tomographic **max normalised grain intensity** map as ground truth.

That map answers **"did one of the listed grains reconstruct here"**, *not* "is there
material here". A band containing real material but no listed grain reads as dark. **Never
locate a sample boundary with it.**

Downstream numbers computed against `y > 14.5` (the vacuum count, the completeness split,
the 1255 → 1012 shrink) are **approximately right, not invalid** — they would shift
slightly if remasked at the 16 µm estimate.

## 1b.6 OPEN — the two `s(ω)` conventions were never reconciled

This is the campaign's live bug and it must not be silently closed by a later session:

| where | convention used |
|---|---|
| the completeness test | `s = −x sin φ + y cos φ` |
| the sinogram forward-model validation | `s = +x sin φ + y cos φ`, bin 0 ↔ `s = −25 µm` |
| the **spot-count sinogram the edge was fitted from** | indexed in `positions.csv` **file order**, i.e. index 0 = +25 µm — the **opposite** sense to the validated projector |

The reconstruction code's own convention is settled and tested — `s = x sin ω + y cos ω`,
`midas_pipeline/recon/fbp.py:176` (phase 6 §6.2). What is open is whether the spot-count
histogram above is being read in the same sense. The edge *distance* is unaffected (it is a
`|s|` quantity, and the 180° pairing control is sign-symmetric); the **tilt sign and the
handedness are not**, which is part of why §1b.4 refuses to quote a tilt.

An empirical pin exists but is weak: of the four candidate forms, the two `y cos ω` forms
beat the two `x cos ω` forms (+0.074 / +0.068 against −0.007 / −0.000). That separates
`y cos` from `x cos`. It does **not** separate the sign.

> Related: ImageD11 writes the same relation as `dty = y0 − sx·sin ω − sy·cos ω`. Useful
> for cross-checking against an S3DXRD result, not as an authority for MIDAS's sign.

## 1b.7 A separate hedge — `positions.csv` handedness was never measured here

On the reference campaign the translation motor readbacks (`samX`, `WM/AeroX`,
`WM/AeroXDail`) were **constant across all 51 files in both scan sets** — the translation
was never logged. The map's handedness therefore rests entirely on the descending
`+25 → −25` convention (phase 1.2), which is a **convention, not a measurement.**

Say so wherever the handedness matters. This is a phase-1 halt condition that this campaign
could not clear, and reporting it as cleared would be wrong.

## 1b.8 What to hand forward

```
boundary found?      yes / no / no edge in the scanned field
method               spot-count sinogram support
180 deg partner      found at <...> deg   [REQUIRED control]
edge distance        <...> um   (from <n> independent routes, agreeing to <...> um)
edge tilt            NOT QUOTED  (or: <...> deg, with the spread across methods)
vacuum voxels        <n>/<N> (<%>)
handedness           measured / CONVENTION ONLY
```

Then back to [`phase-2-configure.md`](phase-2-configure.md), or on to
[`phase-5-read-report.md`](phase-5-read-report.md) if the map already exists and this was a
retrospective masking pass.
