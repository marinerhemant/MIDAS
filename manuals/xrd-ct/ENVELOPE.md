# XRD-CT — measurement envelope

**Technique:** powder-like diffraction tomography — translations × ω, azimuthally integrated
**Last checked:** 2026-08-18 · **Owner:** MIDAS maintainers

> Part of the **XRD-CT doc set**. Spine: [`README.md`](README.md).

What this measurement can and cannot determine, and which of those is changeable. Read it
before promising an answer, and before suggesting a different measurement.

> **Not the scope gate.** The scope gate in the spine says whether these *recipes* apply to
> your data. This file says whether the *measurement* can answer the question. A dataset can
> be squarely in scope and still unable to support what is being asked of it.

---

## 0. TWO independent gates — contrast, and an ω-locked floor

**Contrast decides whether the *peak* is measurable. A separate question decides whether the
*azimuthal pattern* means anything: is there structure that more frames will not remove?** Passing
the first gate does not get you the second.

The measured counter-example is unambiguous: **CeO₂ at 11-ID-C has peak/background 22–137** —
excellent by any standard — **and still carries a 3.7 % random + 0.9 % ω-locked azimuthal floor on
every ring**, so its per-voxel "texture" is not interpretable. **The leading candidate for the
floor is interpolation error in the background estimator itself** — not the sample
(`LAB_NOTEBOOK.md` §5b-ter). Grain counting was proposed and refuted by three lenses.

### 0a. Gate two: is there structure a texture fit could mistake for texture?

Two cheap measurements, both worth making before any texture claim:

**Poisson excess.** Compare the measured per-frame azimuthal RMS against `sqrt(Σ raw counts in
window) / area`. A ratio near 1 means the "structure" is shot noise and there is nothing to
explain. CeO₂ measured **3.5×** (range 2.4–8.1), so its structure is real — and still not texture.

**Does it average down?** Sweep N_ω and fit

```
RMS(N)² = systematic² + random²/N
```

`random` is the part more frames will remove; **`systematic` is a floor that no amount of ω can
touch**, and it is the component a texture fit can genuinely mistake for texture. **Do NOT try to identify the floor's cause from how its amplitude scales with the illuminated
chord.** That was attempted and refuted: the exponent is dominated by an unfitted capillary
radius (±11 % moves it by ±0.35) and, worse, is manufactured by subtracting an across-position
mean. Shot noise and grain counting predict the *same* exponent anyway.

**Two things to get right in that fit**, both learned the hard way:
* **Weight it.** `curve_fit` with no `sigma=` on points spanning 6 % → 0.6 % pins `random` to the
  large low-N values and leaves `systematic` to residual scatter — understating its uncertainty by
  ~4×. Bootstrap the error; do not quote the covariance.
* **Simulate the sampling before believing a floor.** Overlapping random draws at large N *could*
  manufacture one. On the CeO₂ scheme they do not — a pure-random null gives at most 0.35 %
  against an observed 0.9 % — but that had to be checked, not assumed.

### 0b. Gate one: peak-to-background

Measure it in phase 2, per ring, before promising anything.

| peak/background | Strain (centroid) | Texture (area) |
|---|---|---|
| **> 0.3** | quantitative | attemptable — still needs the positive control and the polynomial `r²` |
| **0.05 – 0.3** | quantitative | **global (sample-average) only.** A planted texture is detected at 12–31 % improvement here, but per-voxel `\|corr\|` never exceeds 0.35 |
| **< 0.05** | still good to ~1 % | **not answerable.** At 2 % contrast area scatter is 36 % against 0.85 % for the centroid |

**The sensitivity floor is contrast-dependent, not a flat percentage.** Measured with a
realistically estimated background, a planted ~25 %-amplitude fibre produces:

| peak/bg | 0.50 | 0.20 | 0.10 | 0.05 | **0.02** |
|---|---|---|---|---|---|
| global improvement over null | 30.8 % | 30.6 % | 24.0 % | 12.1 % | **2.6 %** |

At peak/bg = 0.02, **genuine texture yields less than a 5 % improvement** — so a refute line
fixed at 5 % would reject real texture there. Set the refute line from a control run at *your*
contrast, not from a constant. This is the single most useful thing the control tells you, and
it is why `ENVELOPE.md` cannot give you one number.

Real anchor: the DAC Ti scan sat at **0.005–0.17** and delivered a credible strain result and a
texture bound near zero. That is a successful outcome for that data, not a failure.

**Why this is not a signal-averaging problem you can beat with more frames.** At ~220 counts
of background a bin carries ~15 counts of Poisson noise, while a peak at `peak/bg = 0.02` has
an amplitude of ~4 counts — *below* the per-bin noise. Integrating the window recovers signal,
but the dominant error is then the **background model**, which is systematic per frame and
does not average down like Poisson noise. This is the softener that must travel with any
bound derived from the positive control, which carries Poisson noise only.

## 1. Fixed — cannot change this run or beamline cycle

No suggestions here. State the consequence and the substitute.

| Property | Value | Provenance | What it makes unobtainable | Substitute |
|---|---|---|---|---|
| **Only even harmonic orders are measurable** | odd `l` annihilated | `Y_l^n(-h) = (-1)^l Y_l^n(h)` + Friedel | The odd-`l` "ghost" part of the ODF, **for every scan design**. No extra ω, translations or rings recover it | A **positivity constraint** escapes it (Matthies) — a modelling change, not a measurement one. Or report the even part and quote `SymGSH.ghost_dimension()` |
| Symmetry recoverable | the **Laue** group, not the point group | Friedel | The distinction between a non-centrosymmetric group and its Laue class. 73 space groups differ | None from diffraction intensity alone |
| Peak-to-background per ring | set by sample, anvils/furnace, energy | acquisition | **Decides whether texture is answerable at all** (§0) | Strain, which survives where area does not |
| Fibre axis identifiability | one axis, if it is the rotation axis | `n_s·ẑ = cos θ_B sin η` has no ω dependence | A fibre about the rotation axis is **static in ω**. So an axial fibre and an ω-independent instrumental artefact are *not* separable by their ω behaviour | Separate them by the ladder (global vs per-voxel) and the polynomial `r²`, not by ω |
| Ring multiplets | fixed by the phase and the geometry | phase + λ + distance | An overlapped line cannot give an area or a centroid. hcp Ti α(101) is a doublet | Exclude it. `count_maxima` reports it; do not fit it |
| Absolute strain scale | needs an independently confirmed distance | calibration | Absolute `d₀`. On one 11-ID-C scan both the stored metadata (1600 mm) and the beamline calibration (1579.5 mm) were wrong; the data required 1632 mm | **Relative / deviatoric** strain, referenced to the median over azimuths. This is the default in `strain_from_centroid` for exactly this reason |
| Trace of the strain tensor | degenerate with `d₀` | `q̂` is a unit vector, so `tr(ε)` multiplies the same constant as the reference lattice parameter | The hydrostatic part. **Exactly** degenerate, not weakly — fitting six components plus `d₀` gives condition number 5.4e14 | Deviatoric part + a per-voxel apparent `d`-spacing, never decomposed. This is what `tensor_strain` does |

**Consequence worth stating on any report:** at low contrast the *texture* result is set by
the background model, not by the fit. A report that presents a low-contrast per-voxel texture
map as a measurement is wrong; one that "improves" it by adding harmonic orders is making it
worse.

## 2. Configured — set per run, changeable next time

The only tier where "what could be observed differently" has an answer.

| Parameter | Typical | Achievable | Limited by | What changing it would buy |
|---|---|---|---|---|
| **Anvil / furnace scattering** | dominates at low 2θ in a DAC | collimation, energy, anvil material | station + cell design | **The single largest texture improvement.** Raising peak/background from 0.02 to 0.3 moves texture from unanswerable to attemptable |
| Ring count and selection | per run | rings visible, unsaturated, **singlet** | detector extent, energy, phase | Independent azimuthal channels. The cross-ring agreement test needs ≥3 vetted rings to mean anything |
| ω step and range | 0.25°, 360° | acquisition time | — | Spatial (Radon) sampling. Note η bins are **not** spatial views: they sample orientation space, and only ω samples the Radon variable |
| Translation step / beam size | 50 µm | optics + stage | in-plane voxel size | Spatial resolution. Also the unknowns/rows ratio in §3 |
| Energy | per run | source + optics | ring accessibility, penetration | Which rings are reachable, and the anvil background |
| Azimuthal binning | 50–64 bins | integrator | counts per bin vs angular resolution | Directly trades pole-figure resolution against per-bin SNR. Muerer use 64 |
| Exposure / attenuation | one setting | HDR or graded attenuation | station configuration | Un-saturating bright rings while lifting weak ones |

## 3. Intrinsic (identifiability) — count before you fit

**Unknowns grow roughly as `L³`; the row count does not.** So a harmonic model goes
underdetermined at high `L`, and an underdetermined least-squares fit drives its residual
toward zero *for free*.

Measured, 156 voxels and 24192 rows — unknowns/rows:

| L | 6 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|
| ratio | 0.30× | 0.52× | 0.79× | **1.43×** | 1.81× |

**Report this ratio with every residual.** "The residual reached the noise floor at L ≥ 12" is
not evidence of anything when the system is underdetermined there.

**Required `L` rises with crystallites per voxel:** L=6 at ~1000, L=8 at ~3000, L=10 at ~10⁴,
measured in the overdetermined regime. So a finer voxel (fewer crystallites) needs a *lower*
`L`, and there is a window — too coarse and the texture is a sample average, too fine and the
rings go spotty and the technique changes (see the spine's scope gate).

**A favourable count is necessary, never sufficient.** Conflating "overdetermined" with "full
rank" is the specific error this section exists to prevent.

**Coefficient counting, done right.** A per-voxel ODF has **no sample symmetry to spend**, so
the `m` index runs over all `2l+1` values and the count is `Σ_l M(l)(2l+1)` — an order of
magnitude larger than the `Σ_l M(l) = 23` figure (Bunge's cubic-*orthorhombic* rolling-texture
number) that a handoff plan used. `SymGSH.n_coef` is the number to quote.

## 4. What a model choice buys and costs

| Model | Params/voxel | Non-negativity | Use when |
|---|---|---|---|
| **Uniaxial squared-modulus** (`odf_uniaxial`) | 4 | **free** (squaring) | Default at any contrast below ~0.3, or when a single fibre axis is physically justified. Only even orders arise, so the ghost problem never appears |
| General symmetry-adapted GSH (`gsh`) | 23–61 at L=6–10 | not enforced | High contrast, many independent rings, and an identifiability count that clears §3 |
| Non-negative radial-basis | grid-dependent | pointwise | **Untested here.** Our comparison against GSH was INVALID (dof mismatch). `texture_kernel` is a simulator, not an inversion basis |

The uniaxial model's own ceiling: **Hermans `S` saturates near 0.61** with only `a₂` free, and
then decreases. A fit stalling near `S ~ 0.6` may be at that ceiling rather than at the data's
limit — `a₄` and `a₆` are what a sharper texture needs.

## 5. Questions this measurement cannot answer

State these plainly rather than attempting them:

* **Absolute `d₀` / hydrostatic strain** without an independently confirmed distance (§1).
* **The odd-`l` part of any ODF**, without a positivity constraint (§1).
* **Whether an ω-static azimuthal pattern is texture or an instrument**, from its ω behaviour
  alone (§1) — an axial fibre is necessarily ω-static.
* **A per-voxel texture map from a low-contrast scan** (§0). A *global* one may still be
  reachable, and the ladder in `fit_uniaxial_ladder` tells you which.
* **Anything about a spotty-ringed sample.** That is scanning-3DXRD; use `pf-hedm`.
* **A texture claim that has not been through the positive control** at the measured contrast
  — not because the answer would be wrong, but because it would be uninterpretable.
