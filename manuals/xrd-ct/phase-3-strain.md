# Phase 3 — strain: the tractable deliverable

> Part of the **XRD-CT doc set**. Spine: [`README.md`](README.md).

**Goal:** per-azimuth and per-voxel strain from ring centroids.

**Run this whether or not texture is the goal.** It is independent of phase 4, it survives
contrasts where texture cannot, and on the one real dataset taken through both it was the
deliverable the data supported.

---

## 3.1 Why this works where texture does not

The centroid is a **ratio** — `Σ I·r / Σ I` — so a slowly varying background largely cancels.
The area is a **difference**, and inherits the background model's error in full. Measured at
2 % contrast with no planted azimuthal structure: **centroid scatter 0.85 %, area scatter
36 %**.

That is the whole reason this phase exists separately.

## 3.2 Centroid → d-spacing → strain

```python
eps = strain_from_centroid(centroid, r_axis, two_theta_axis, wavelength_a)
```

* Uses the integrator's **R↔2θ map** (phase 1.2), which carries the tilts and distortion. An
  idealised `arctan(r/L)` does not.
* **Relative by default**: with `reference_d` unset, the reference is the median over the
  azimuths supplied. That makes it a deviatoric-like strain and absorbs any distance error.
* Supply `reference_d` **only** when you have a `d₀` you can defend, and when the distance has
  been confirmed against the data (phase 1.1). On one 11-ID-C scan the metadata distance and
  the beamline calibration were both wrong by 2 % and 3.3 % — which is 20,000–33,000 µε of
  apparent strain, several times a real DAC signal.

## 3.3 Apply the per-azimuth mask FIRST

```python
live = ext.live_mask(min_snr=3.0)
cen = np.where(live, ext.centroid, np.nan)
eps = strain_from_centroid(cen, r_axis, tt_axis, lam)
```

**This is the step whose omission produced 107,410 µε and 638,282 µε** (11 % and 64 % strain).
Gating on the ring's *median* SNR lets dead azimuths through, and any peak-to-peak spread over
η is a max-minus-min that a handful of them dominate.

## 3.4 Report spread robustly, and report how much is live

Never a peak-to-peak. Never a mean alone.

```python
q5, q95 = np.nanpercentile(eps_eta, [5, 95])
mad = 1.4826 * np.nanmedian(np.abs(eps_eta - np.nanmedian(eps_eta)))
n_live = int(np.isfinite(eps_eta).sum())
```

Real numbers from the DAC Ti scan, after the per-bin mask:

| | value |
|---|---|
| inter-percentile range (5–95 %) | **3352–7324 µε** across six reflections |
| MAD | 1228–3354 µε |
| interpretation | 0.3–0.7 % azimuthal deviatoric strain |

**What makes those credible is that six independent reflections agree.** That is the right
reason, and it is still not verification — the number is labelled **provisional** in
`LAB_NOTEBOOK.md` §6 and must stay labelled in any text that leaves the session.

## 3.5 Cross-reflection agreement is the test that can fail

The same physical strain state must be seen by every vetted reflection. If per-ring strain maps
disagree (pairwise correlation below ~0.3), what is being fitted is not one strain state —
suspect a multiplet that slipped the vetting, a ring assignment error, or a phase mixture.

For a DAC specifically, `Q(hkl)` is **expected** to differ between reflections (elastic
anisotropy) — so compare the *azimuthal pattern* and the *spatial map*, not the magnitude.

## 3.6 The full tensor, when the coverage supports it

`midas_dt.tensor_strain` solves a **deviatoric** strain tensor per voxel by direct inversion,
replacing a free peak centre with physics: every measurement of every ring constrains the same
five numbers.

**Five components, not six, and this is exact rather than a conditioning nicety.** `q̂` is a
unit vector, so `q_x² + q_y² + q_z² ≡ 1` and `tr(ε)` multiplies the same constant as the
reference lattice parameter. The two are **exactly degenerate** — measured condition number
5.4e14 and answers of order 1e12 µε when six components plus `d₀` are fitted. So the model
carries the deviatoric part plus a per-voxel *apparent* d-spacing, and the apparent spacing is
never decomposed into `d₀` and a dilatation.

**The projection is of PATTERNS, not of peak positions.** A ray's observed peak sits near the
intensity-weighted mean of the voxel spacings along it, and a weighted mean **does not add**.
Back-projecting fitted positions is measured at correlation 0.03 between branches on real data.
So each voxel renders its own pattern, the patterns are summed along the ray by the same sparse
Radon operator, and the residual is taken in projection space where the measurement lives.

## 3.7 Before reporting a per-voxel strain map

Run the separability check, same as for texture:

```python
r2 = explained_by_polynomial(strain_map_values, xy, order=3)
```

Above ~0.5 the map is a smooth field — absorption, illumination, a centring error — not
per-voxel physics. **This has retracted a result in this project three times.** Report the `r²`
next to the map either way; both outcomes are informative.

## 3.8 What to hand forward

```
Per ring:   azimuthal strain profile, 5-95% range, MAD, live azimuth count
Cross-ring: pairwise agreement of the azimuthal patterns and the spatial maps
Per voxel:  strain map (or deviatoric tensor), with polynomial r^2
Scale:      RELATIVE (median-referenced) or ABSOLUTE (distance confirmed how?)
Label:      provisional unless it has been through /verify
```

Then `phase-4-texture.md` if the contrast allows, otherwise straight to
`phase-5-report.md`.
