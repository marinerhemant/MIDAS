# Phase 1 — Configure the material, reflection and geometry

> Part of the **DFXM doc set**; spine is [`README.md`](README.md). This phase fixes θ_B, Λ
> and χ from the material and energy — never a template (rule, traps table). It also decides
> whether you are in the thin (kinematic-safe) or thick (dynamical) regime, which governs
> what §4 is allowed to claim.

---

## 1a. Reflection, Bragg angle, and the reference

> **HALT if the deposit carries no energy or wavelength.** Everything below —
> `wavelength_A`, `energy_keV`, and therefore θ_B, Λ, χ and ε_ref — takes energy as a
> literal argument, and this phase never says where to get it when the file does not
> have it. Checked 2026-08-12: **neither `darling.assets.mosaicity_scan()` nor
> `rocking_scan()` carries any energy field anywhere in its HDF5 tree**, and
> `darling.metadata.ID03`'s motor map has none either.
>
> Assuming one is not a shortcut, it is the documented failure. Lab Notebook §5k
> ("Darwin ladder", RETRACTED, four independent errors) records that the retraction on
> the `fatigue_test` sample turned partly on exactly this: *"the file carries no element
> or energy field, and λ was itself assumed."* The lesson was archived and never turned
> into a stop, so a fresh reader can pattern-match `rocking_scan()` to
> `mosaicity_scan()`'s "Al 1050" docstring and repeat it.
>
> Get the energy from the beamline record or the proposal, or **stop and ask**. An
> orientation-only (mosaicity) reduction is exempt — hard rule 3 — so §2 can proceed
> without it. Nothing in §1 or §4 may.

From the material and the reflection **read in §0** (not the filename), compute the Bragg
geometry with the package, not by hand:

```python
from midas_dfxm.scan import bragg_two_theta_deg, reference_q_nom
from midas_dfxm.takagi_taupin import bragg_angle_deg

theta_B = bragg_angle_deg(crystal, hkl, wavelength_A)   # or bragg_two_theta_deg(...)
q_nom   = reference_q_nom(crystal, hkl)                  # the nominal scattering vector
```

`µm, degrees, Å` throughout; Å only for wavelength and lattice parameters (rule 6).

## 1b. Susceptibility and the extinction length Λ — the regime gate

The extinction length sets the kinematic-vs-dynamical boundary (§4). Compute it from the
susceptibility, per reflection and energy:

```python
from midas_dfxm.takagi_taupin import susceptibility_fourier, extinction_length

chi0, chih, chihbar = susceptibility_fourier(crystal, hkl, energy_keV)  # complex χ_0, χ_h
Lambda_um = extinction_length(crystal, hkl, energy_keV, theta_B)        # Λ in µm
```

Then classify the crystal:

| t / Λ | regime | what §4 may claim |
|---|---|---|
| ≲ 0.15 | thin | kinematic strain/defect inverse is at the noise floor — safe |
| 0.15–0.3 | marginal | kinematic residual leaving the floor — flag any strain claim |
| ≳ 0.3 | thick / near-perfect | kinematic strain inverse **biases**; use the dynamical forward (§4) |

**Orientation is exempt** — the centroid read is exact by symmetry at any thickness in
symmetric Laue (§4a). The boundary is a *strain/defect* boundary.

**Classify per reflection, never once per sample.** Λ ∝ 1/|F|, so a weak superlattice or
satellite reflection sits orders of magnitude further from the boundary than its parent: one
archived case gave **Λ ≈ 24 µm** for the strong parent but **7,000–10,000 µm** for satellites
whose |F| was ~400× smaller. The same crystal was safely kinematical in one channel
(t/Λ ~ 1e-4) and potentially marginal in the other (Notebook §7g). Two traps here:

- Do **not** use the *mosaic* width as the coherent block size in t/Λ. Mosaic spread and
  coherent block size are independent quantities, and substituting one for the other is
  precisely the retracted reasoning in Notebook §5k.
- Do **not** use rocking-width / Darwin-width as a dynamical-relevance criterion. The
  criterion is **t_coherent/Λ**. On real public ID03 Al the rocking curve was 62× the Darwin
  width with the step *equal* to the Darwin width, so the dynamical feature was not resolved
  by even one step — a wide rocking curve bounds what you can *diagnose*, not how large the
  dynamical effect is (Notebook §7h). Test the dynamical forward on near-perfect crystals
  (Ge, Si, HgCdTe, oxide films) at step ≲ 0.2 mdeg.

`χ_0` also fixes the **refraction gauge** you must not mistake for strain (§4):
$\varepsilon_{\mathrm{ref}} = \chi_{0r}/(2\sin^2\theta_B)$ (≈ 144 µε for Cu 002 at 0.71 Å).

## 1c. Resolution — anisotropic, from Poulsen

DFXM's instrument resolution is **anisotropic** (reciprocal-space resolution function of the
objective + beam). Do not assume an isotropic PSF (traps table):

```python
from midas_dfxm import poulsen_resolution_widths, aligned_resolution, detector_model
widths = poulsen_resolution_widths(...)   # the anisotropic reciprocal-space widths
```

This kernel is what you deconvolve mosaicity against in §4 — the measured spread is the
intrinsic mosaicity convolved with it.

**Known package simplification (Notebook §10).** `poulsen_resolution_widths` returns three
distinct widths — `sigma_par` (longitudinal, objective NA), `sigma_rock` (in-scattering-plane
transverse, condenser NA) and `sigma_roll` (out-of-plane transverse, objective NA) — which are
genuinely different (Carlsen/Poulsen: σ_rock ≈ 1e-3 ≪ σ_roll ≈ 9e-3). But `aligned_resolution`
currently takes only `sigma_par` + one `sigma_perp`, so the built `ResolutionFunction` is
**transverse-isotropic** (both transverse directions get the same σ). For a fully anisotropic
resolution, pass the tighter transverse width and note the roll direction is under-resolved, or
extend `aligned_resolution` to two transverse widths.

## 1d. Write the configuration into `SURVEY.md`

Record θ_B, 2θ, Λ, `Im χ_h` (absorption), the t/Λ regime, ε_ref, and the resolution widths,
each with the call that produced it. Every downstream number in §2–§5 is conditional on
these, and the report (§5) must cite them.
