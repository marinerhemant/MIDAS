# Phase 4 — Analysis past orientation, and the validity boundary

> Part of the **DFXM doc set**; spine is [`README.md`](README.md). Everything past a plain
> orientation map lives here, together with the two checks that keep a strain claim honest:
> the **refraction gauge** (rule 2) and the **~0.3 Λ** kinematic validity boundary (rule 3).
>
> **Honesty marker.** The capability inverses in this phase (typing, defect model, the
> dynamical inverse) are **simulation-grounded** in `midas_dfxm` as of writing — verified
> against closed forms and cross-model, but not yet on a real DFXM capability measurement.
> Say so in the report (traps table). The refraction and boundary results below are verified
> against the dynamical forward.

---

## 4a. The refraction gauge — do not correct it as a field

Before reporting any strain, settle whether a uniform offset is refraction. The mean
refraction books a **constant** apparent strain $\varepsilon_{\mathrm{ref}} =
\chi_{0r}/(2\sin^2\theta_B)$ (≈ 144 µε for Cu 002 at 0.71 Å; §1b). On a **relative**
intragranular map it is a **gauge** — absorbed into the lattice reference, recovered to
0.03 µε — **not** a per-pixel bias (Notebook §3, DIAGNOSIS: uniform_strain_offset).

- **Relative strain map:** leave it. Subtracting it removes a real reference and leaves the
  gradient part in.
- **Absolute / cross-reflection strain:** apply it (the Hart-1988 refractive-index
  correction).
- **Where refraction varies** (thickness / perfection gradient): the varying part aliases
  into a real spurious strain (transfer slope ~0.43); its magnitude needs the Takagi–Taupin
  forward and it is maskable in the near-perfect matrix.

Extinction is a *symmetric* rocking-curve reshaping, so the orientation centroid is invariant
to it (< 1e-4 mdeg) — it does not bias orientation even when it varies spatially.

## 4b. The kinematic validity boundary — is the strain claim in the safe regime?

From the t/Λ classified in §1b:

- **Orientation** — exact by symmetry at any thickness in symmetric Laue (§4a of the
  Notebook). No boundary.
- **Strain / defect** — the kinematic inverse is at the noise floor only for t ≲ 0.15 Λ,
  biases past ~0.3 Λ (+38 % amplitude by 1.1 Λ). Past 0.3 Λ, use the dynamical forward:

```python
from midas_dfxm.takagi_taupin import solve_tt_laue, solve_tt_bragg   # dynamical forward
```

The boundary is verified **cross-model** (dynamical data, kinematic inverse recovers 0.4 %
thin, breaks at 0.3 Λ), not by inverse crime (`cross_model_test.py`, Notebook §4b). The
geometric full-F inverse is exact on clean thin data (round-trip 4.67e-20), so past the
boundary the fix is the forward model, never the linear algebra.

In **asymmetric/oblique** Bragg geometry, absorption skews the Darwin curve and shifts the
orientation centroid by ~0.16 mdeg — the leading correction for oblique full-tensor DFXM.

## 4c. Dislocation typing (simulation-grounded)

From multi-reflection weak-beam images, recover the dislocation type, Burgers vector
**including sign**, character and core position:

```python
from midas_dfxm import (stroh_dislocation, g_dot_b, classify_character,
                        recover_burgers, visibility_series)
from midas_dfxm.detect import identify_dislocation, weak_beam_stack
```

The sign is recoverable because $\mathbf b \to -\mathbf b$ point-inverts the displacement
field about the core (it is not degenerate with core position) — **under an assumed absolute
lattice reference that fixes the sign of $\Delta\mathbf Q$**. That reference plus
voxel-for-voxel registration is exactly the hard experimental part (§3). Report typing as a
simulation capability with that assumption stated.

## 4d. Defect model and physics-constrained inference (simulation-grounded)

The physics-regularised defect model fits a parametric dislocation (core + amplitude) rather
than a free per-voxel field, and beats a fair free-F baseline by a median ~57× under matched
noise — an **upper bound under a matched model**, quoted as a median with a 10–90 band, not a
single draw. The core-position objective is non-convex around the 1/r singularity but has a
finite basin (~4 px); the argmax initialiser sits inside it. GND density / wall spacing:
`fit_gnd_density`, `fit_wall_spacing`. Crystal-plasticity inference through the JAX-CPFEM
adjoint is a proof-of-concept that puts plasticity and the DFXM observable on one graph — it
fits a strain field, not the raw observables (Notebook §5d).

## 4d′. Segmenting two structures from an intensity ratio — check the channel first

A common analysis thresholds the ratio of two reflections' integrated intensities to get a
phase fraction. **That is only a phase fraction if each channel is forbidden in the competing
structure.** Compute |F|² at the **measured Q** for every candidate structure *and* every
twin/domain variant before thresholding. Commensurate periods are the thing to catch: a
supercell of period 2n·c reproduces every reflection of one at n·c, so a reflection assigned to
the coarser structure is a **shared channel**. Twinning compounds it — a symmetry rotation
permutes which modulation arm each variant contributes to the same lab-frame **Q**, and a
systematic extinction can forbid one variant outright at the shared position.

Where the channel is shared, the ratio produced by a **100 % single-phase** region is neither
0.5 nor constant: it depends on local variant populations (13.9× spread across variants in one
computed case), and a single-phase region can land inside the "mixed/undecidable" band or be
actively mislabelled as the other phase (Notebook §7i). Prefer a forbidden-in-the-other channel;
where none exists, replace the hard deadband with a per-pixel class probability plus an explicit
**"undecidable at this dose"** class. Do not reach for a better feature extractor first — a
better extractor on a mis-specified contrast returns a mis-specified answer.

## 4d″. Statistics on maps — pixels are not independent samples

Any significance computed over map pixels needs the field's autocorrelation length. The optical
PSF and the microstructure both correlate neighbours, so an iid bootstrap understates the
variance: a "−3.2 σ" over 6,812 pixels in a field autocorrelated to 0.90 at 48 px became
**−0.73, p = 0.47** under a block bootstrap and a phase-randomised surrogate, at n_eff ≈ 172
(rule 19, Notebook §5h). Report n_eff and the correlation length. Two companions: match a null
on every nuisance variable that matters (matching intensity alone flipped a sign once |∇I| was
matched too), and check a statistic's admissible range before believing it — a "dip" of 0.74
cannot be Hartigan's dip, which is bounded by 0.25.

## 4e. Experiment design (Fisher)

`field_inverse.fisher_information` and the A-optimal reflection selection choose reflection
sets that minimise the Cramér–Rao trace; the same calculus fixes the optimal crystal
thickness (numerically ~0.56 Λ, a trade between weak thin contrast and thick saturation).
Use it to *design* the next scan, not to validate the last one.
