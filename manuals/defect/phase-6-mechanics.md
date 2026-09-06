# Phase 6 — from the diffraction observables to deformation mechanics

**This is where most of the package lives**, and it is the half that turns diffuse-scattering
observables into plasticity: dislocation content, resolved shear stress, stored energy, work
hardening, and a handoff to crystal plasticity. Everything here consumes `phase-5`'s asterism
and line-profile widths plus the per-grain orientations and strains.

Two standing cautions before any of it:

* **Nothing here is more reliable than the grains it runs over.** If the indexer fragmented
  the microstructure (`DIAGNOSIS.md` 3), every per-grain distribution below has an `n` that
  belongs to the indexer.
* **The whole chain inherits the phase.** Getting the phase wrong changes the Burgers vector,
  which changes ρ by the square. On the reference sample an early analysis used b = 4.29 Å
  from the wrong phase; the correct FCC b = 2.571 Å moved ρ by more than an order of
  magnitude. `LAB_NOTEBOOK.md` R6.

## Decompose the asterism before interpreting it

```python
from midas_defect.asterism.second_moment import per_grain_asterism_tensor
from midas_defect.asterism.local_decomposition import per_grain_asterism_local
from midas_defect.asterism.eigenvalue_spectrum import asterism_anisotropy_per_grain
from midas_defect.asterism.direction import edge_fraction_per_grain
```

`per_grain_asterism_local` splits the broadening into **lattice rotation** and **strain**.
On the reference sample that came out ~**72 % rotation / 28 % strain** — worth doing before
attributing all of it to either.

## Line profile → dislocation density

```python
from midas_defect.line_profile.per_grain_reflections import collect_per_grain_reflections
from midas_defect.line_profile.modified_wh import modified_wh_per_grain
from midas_defect.line_profile.warren_alpha import warren_alpha_per_grain
from midas_defect.line_profile.warren_beta import warren_beta_proxy_per_grain
```

Reference-sample outcome: **ρ ≈ 1.3 × 10¹³ m⁻²** median (per-grain 10¹²–6.5 × 10¹³, 191 of
248 grains fitting), coherent domain ≈ 155 Å, microstrain ≈ 0.023 %.

**Three caveats that must travel with the number** (`ENVELOPE.md` §8, §10):

* **No instrumental-broadening subtraction ⇒ ρ is an UPPER bound.** Opposite direction to the
  coherence-length bound in §2 — do not confuse them.
* The WH prefactor (`k ≈ 16` in `ρ = kε²/b²`) is **convention-dependent**.
* The defensible statement was the order of magnitude, `ρ ~ 10¹²–10¹³ m⁻²`, not the median.

**Warren α is a fault probability and it is easy to get wrong.** On the reference sample an
α = 0.16 was self-retracted as a **Hendricks–Teller local-minimum artifact**; the corrected
value is ~0.005 typical and ~0.022 in the faulted third. `warren_beta_proxy_per_grain` is
explicitly a **proxy** — a rod/Bragg intensity ratio, not a rigorous Warren α from {111}/{200}
peak shifts — and must be labelled as one.

## How much of the dislocation content can WH even see?

```python
from midas_defect.thermodynamics.taylor_inversion import (taylor_implied_total_rho,
                                                          wh_visible_fraction)
```

`wh_visible_fraction` is the honesty check on the whole section: Taylor-inverting the flow
stress gives the total ρ the mechanics *requires*, and the ratio to the WH-visible ρ says what
fraction the diffraction can account for. Quote it. A dislocation density presented without it
implies a completeness the measurement does not have.

## GND, SSD

```python
from midas_defect.gnd.scalar_gnd import scalar_gnd_from_inter_grain_misorientation
from midas_defect.gnd.nye_tensor import per_grain_nye_tensor
from midas_defect.gnd.ssd_decomposition import ssd_gnd_decomposition
```

Scalar GND from inter-grain misorientation gradients; the Nye tensor where the orientation
field supports it; the GND/SSD split. The scalar route depends on the **grain adjacency** you
feed it — over-fragmented "grains" manufacture misorientation gradients that are really
intra-grain mosaic.

## Stress, strain, Schmid

```python
from midas_defect.stress.cubic_anisotropic import per_grain_stress_cubic
from midas_defect.stress.invariants import von_mises, hydrostatic, max_shear, lode_parameter, triaxiality
from midas_defect.stress.resolved_shear import max_resolved_shear_per_grain
from midas_defect.strain.von_mises import von_mises_strain, deviatoric_hydrostatic_decomposition
from midas_defect.strain.eigenvalue_spectrum import per_grain_eigenvalues
from midas_defect.schmid.per_grain import schmid_factor_per_grain
from midas_defect.schmid.per_system import schmid_factor_per_system, spatial_active_system_agreement
from midas_defect.schmid.stratification import stratify_pairs_by_schmid_max
from midas_defect.phases.fcc import FCC_SLIP_111_110, FCC_TWIN_111_112
# ...and midas_defect.phases.bcc / .hcp for the other lattices. hcp carries the
# Bravais<->Miller conversions and tensile/compressive twin systems; the package is
# phase-agnostic here, but only FCC has a real-data anchor (ENVELOPE.md 9).
```

**The trap here is elastic anisotropy.** On the reference sample the twin population showed
higher stress and stored energy than the parent — and the honest reading was that this is an
**elastic-anisotropy projection of ~equal strain**, not a real mechanical asymmetry. The
asterism rotation and strain widths were identical parent ≈ twin to 1–3 %. A stress difference
computed through an anisotropic stiffness from two differently-oriented populations will
appear even when the strain is the same. `LAB_NOTEBOOK.md` E8.

## Variants and Σ3 pairs

```python
from midas_defect.variants.matched_pairs import find_sigma3_partners
from midas_defect.variants.common_reference import assign_variants_common_reference, build_sigma3_pair
from midas_defect.variants.kmeans_fz import assign_variants_kmeans
```

Misorientation operators are side-sensitive: cross-check any variant assignment against
`midas_stress.misorientation` on a pair believed nearly coincident (`Uaᵀ Ub S`, **S on the
right**). And note the MIDAS-wide trap: `midas_stress` misorientation is in **radians**.

## Distributions, and testing them

```python
from midas_defect.distributions.mackenzie import mackenzie_pdf
from midas_defect.distributions.divergence import kl_divergence, jensen_shannon_divergence
from midas_defect.distributions.friedel import friedel_pair_asymmetry
```

Mackenzie is the random-texture reference for a misorientation distribution — the null a real
one has to beat, and the diagnostic for over-fragmentation (`DIAGNOSIS.md` 3).

## Energy and work hardening

```python
from midas_defect.energy.per_grain import elastic_energy_density_cubic
from midas_defect.energy.balance_closure import twin_boundary_energy_density, energy_balance_closure
from midas_defect.energy.volume_weighted import volume_weighted_energy_per_variant
from midas_defect.thermodynamics.mecking_kocks import variant_specific_k2, mk_evolve
```

Same warning as the stress section: an energy difference between two populations that differ
in *orientation* is an anisotropy projection until shown otherwise.

## Spatial

```python
from midas_defect.spatial.autocorrelation import epsilon_autocorrelation
from midas_defect.spatial.hall_petch import hall_petch_slope
from midas_defect.spatial.stress_gradient import stress_spatial_gradient_per_grain
```

**Check the layers are distinct data before claiming spatial uniformity.** On the reference
sample a pipeline bug (the FF zip-convert stage reusing the first layer's converted data for
every subsequent layer in a multi-layer batch) would have made ten layers look perfectly uniform
because they were the same layer. It was caught and fixed, and the depth profile was re-run
with a verified unique md5 per layer. Uniformity across depth is exactly the claim that a
data-reuse bug fabricates. `LAB_NOTEBOOK.md` R7.

Reference-sample result after the fix: grain count 252 ± 19 (CV 7 %), volumetric strain
0.148 ± 0.023 %, {111} texture MRD 4.9 ± 0.4, median misorientation 14.9 ± 1.2° — genuinely
uniform over ~10 layers.

## Uncertainty on any of it

```python
from midas_defect.bootstrap.samplers import ...
from midas_defect.bootstrap.aggregators import ...
from midas_defect.bootstrap.decorators import ...
from midas_defect.honesty import systematic_uq
from midas_defect.stress.voigt import stress_tensor_to_voigt, voigt_to_stress_tensor
```

Bootstrap resampling for any per-grain aggregate. **But resampling alone is not UQ here**:
resampling grains that all carry the same systematic reports the artifact's *consistency* as
*precision* — it once returned "P = 1.00 / 11.2 σ" for a geometry artifact. `systematic_uq`
perturbs the systematics (relabel, re-choose the axis, reseed) instead, and that is the number
to quote.

## Handoff to crystal plasticity

```python
from midas_defect.cpfem.damask_io import write_damask_initial_microstructure, read_damask_grain_output
from midas_defect.cpfem.fepx_io import write_fepx_initial_microstructure
from midas_defect.cpfem.prisms_io import write_prisms_initial_microstructure
from midas_defect.cpfem.elastic_sc import kroner_self_consistent, hill_average_isotropic
```

Writers for DAMASK, FePX and PRISMS, and self-consistent elasticity (Kröner, Hill, Voigt,
Reuss) for the per-grain stiffness the stress section needs. A CPFEM comparison is only as
good as the initial microstructure it was handed — see the over-fragmentation caution at the
top of this file.
