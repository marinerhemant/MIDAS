# Phase 1 — geometry: calibration, the distance, and the R↔2θ map

> Part of the **XRD-CT doc set**. Spine: [`README.md`](README.md).

**Goal:** get from radial bin to 2θ correctly, and confirm the sample-to-detector distance
against the data rather than against a stored value.

---

## 1.1 Confirm the distance against the data. Do not trust metadata.

**This is a halt condition.** On an 11-ID-C CeO₂ scan:

| Source | Value |
|---|---|
| Scan metadata | 1600 mm |
| Beamline calibration file | 1579.5 mm |
| **Refined from the data** | **1632 mm** |

Both stored values were wrong for the data as collected — by 2 % and 3.3 %. A 2 % distance
error is a ~2 % error in every `d`-spacing, i.e. **20,000 µε** of apparent strain, which is
three times the entire real signal on a loaded DAC.

**Test.** Refine the distance against ring positions across as many rings as possible. Many
rings over-determine one distance, so the residual pattern tells you whether the geometry is
consistent or whether a tilt is absorbing the error. Use `midas_calibrate_v2`; do not
hand-roll this.

**Consequence if you cannot confirm it:** report **relative** strain only, referenced to the
median over azimuths. That is `strain_from_centroid`'s default for exactly this reason, and it
absorbs any distance error into the reference. Say in the report that the strain is relative.

## 1.2 Use the integrator's R↔2θ map, not `arctan(r/L)`

The integrator writes a per-radius 2θ table (`.REtaAreaMap.csv` for the MIDAS DT integrator).
**Use it.** It carries:

* the detector **tilts**,
* the **distortion** coefficients (p0–p3),
* the actual bin centres, which are not evenly spaced in 2θ.

An idealised `arctan(r/L)` throws all three away. The resulting error is smooth in R, which
means it looks like a small strain offset per ring rather than like a mistake.

```python
r_axis, tt_axis = load_r_and_two_theta(...)     # from the map file
# then everywhere downstream:
strain = strain_from_centroid(centroid, r_axis, tt_axis, wavelength_a)
```

**Note the bin size.** Integrated DT cakes are commonly finer than one pixel per bin — 0.25
px/bin is typical. Every window width in phase 2 must therefore be specified in **pixels and
converted**, which is what `ring_windows` does. A width written in bins is silently ~4× too
narrow.

## 1.3 Confirm the ω sign and the ω→frame mapping

Established in phase 0; restated because it is a halt condition and cannot be checked later.

```
omega(frame) = SIGN * (startOme + frame * omeStep)
```

At 1-ID with the aero stage, `SIGN = -1` — **every ω negated**. A flipped ω mirrors the
reconstruction; nothing downstream reports a problem, and the mirrored map is internally
consistent and plausible.

Also confirm the **throwaway first frame**. 1-ID discards one frame every acquisition. Check
whether `HeadSize` already skips it: double-skipping loses a real frame, and not skipping
shifts every ω by one step.

## 1.4 Wavelength — trust the number, not the comment

From a real U₃O₈ parameter file:

```
Wavelength 0.136994 # 55.618 keV
```

The comment is **wrong**: 0.136994 Å is 90.5 keV. Trust the number.

Sanity check any λ you are handed: `E[keV] = 12.39842 / λ[Å]`. It takes one line and it has
caught a live error.

## 1.5 The rotation-axis convention that phase 4 depends on

For the standard DT geometry — beam along lab **x**, rotation about lab **z**, translation
along lab **y**:

```
q̂_lab = (−sin θ,  cos θ cos η,  cos θ sin η)
n̂_s   = R_z(ω)ᵀ q̂_lab        ⇒        n̂_s · ẑ = cos θ sin η
```

**That last expression has no ω dependence**, and it is the foundation of the uniaxial texture
model in phase 4 — and of a withdrawn inference. A fibre about the rotation axis produces an
azimuthal pattern that is *static in ω*, so "static in ω therefore instrumental" is **wrong**.
See `DIAGNOSIS.md`.

If your geometry differs (different rotation axis, different beam direction), `fibre_cos_theta`
does not apply as written and phase 4 needs re-deriving. Check, do not assume.

## 1.6 What to hand forward

```
Distance:        value, and HOW confirmed (refined from data / metadata / calibration file)
                 -> if not refined: strain is RELATIVE only, say so in the report
R -> 2theta:     the map file used, and its bin size in px/bin
Wavelength:      value, cross-checked against the stated energy
Omega:           SIGN, startOme, omeStep, first-frame handling
Rotation axis:   confirmed as lab z, or the derivation redone
```

Then go to `phase-2-extract.md`.
