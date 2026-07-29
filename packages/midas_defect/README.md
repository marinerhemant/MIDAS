# midas-defect

Differentiable **diffuse-scattering defect metrology** layered on top of standard
far-field HEDM.

The standard MIDAS chain (`midas_index → midas_fit_grain → midas_process_grains`,
with `midas_transforms` for the detector model and `midas_stress` for
orientations) operates on **detected Bragg peaks** and returns **grains**. It
discards everything between and around the peaks. For a deformed material that
diffuse field is not noise — it is the defect signal:

- **Asterism** — extended, anisotropic intensity hugging each Bragg core,
  encoding the dislocation strain field and orientation gradient. Quantifying it
  gives a per-grain **dislocation density** (Williamson–Hall on radial breadth).
- **Rods** in q-space — 1-D streaks threading several Bragg shells along
  low-index directions, the reciprocal-space signature of planar defects
  (stacking faults, twin walls). For FCC, faults on {111} ⇒ ⟨111⟩ rods.
- **The full intensity budget** — classifying *every* above-threshold voxel
  against the predicted reciprocal lattice of the indexed grains lets the
  scattered intensity be **decomposed and closed at 100 %** (Bragg / near-Bragg
  asterism / inter-Bragg diffuse / low-q halo). Closure is what licenses
  *quantitative* statements ("31 % dislocation asterism, 4 % fault rods")
  instead of qualitative ones.
- **Selection-rule / forbidden-reflection tests** and **3D-ΔPDF** — phase and
  defect-symmetry checks (no second phase, no anti-phase boundaries).

This package is the productized, tested form of that whole-field, post-indexing
diffuse layer.

**Phase-agnostic by design.** Every analysis module is driven by a
`midas_hkls.Crystal` + a `Geometry`; nothing is hard-wired to FCC. The demk
default is `lattice.fcc_cu_crystal()`, but `lattice.cual2_crystal()` (tetragonal
θ-Al₂Cu) is retained as a first-class phase — a future CuAl₂ sample that genuinely
shows superstructure rods is precisely the case the rod / forward-sim / ΔPDF
machinery was built for, and re-points by swapping the crystal cell.

**Shippable notebooks.** [`midas_defect/examples/`](midas_defect/examples/) holds
end-to-end notebooks parameterized by `(Crystal, Geometry, data)`. The flagship
reproduces the demk FCC defect inventory (index → budget → ρ → rods → forbidden)
and is runnable by external collaborators; the same notebook re-points to a
CuAl₂ dataset by editing one cell.

See [`implementation_plan.md`](implementation_plan.md) for the analysis pipelines,
the MIDAS reuse map, and the test/CI architecture.

## Status

Re-scoped 2026-05-21 after the demk FCC re-analysis. The package's original v0.1
framing — "heavily-deformed *single crystal*, treat the pattern as a continuous
field because the standard pipeline cannot index it" — is **retired**: on the
driving dataset the standard pipeline indexes cleanly (~250 grains/layer), the
material is an ordinary FCC polycrystal/sub-grain mosaic, and the diffuse field
is dominated by dislocation asterism (~31 %) over rods (~4 %). The corrected
scope is **diffuse defect metrology on top of, not instead of, standard
FF-HEDM**.

Pre-alpha. v0.1 target: shared infrastructure (`geometry`, `lattice`,
`bragg_diffuse`) + the three capabilities that close the budget
(`intensity_budget`, `williamson_hall`, `defect_tests`), each validated to
reproduce the published demk numbers. Rod/asterism/sub-grain/ΔPDF/forward-sim
modules exist and pass synthetic tests; they promote once anchored to real-data
regressions.

## Module status

The **validated core** (v0.1) is anchored to the published demk full-res numbers;
the rest is synthetic-tested and promotes once anchored to real data.

| module | status | notes |
|---|---|---|
| `geometry` | **core** | MIDAS-canonical detector model (reuses `apply_tilt_distortion`), ω about Z |
| `lattice` | **core** | FCC + tetragonal CuAl₂, phase-agnostic shells |
| `bragg_diffuse` | **core** | full-field classifier + geometry QC (96.4 % on-lattice) |
| `intensity_budget` | **core** | 4-bin partition, closes to 100 % |
| `williamson_hall` | **core** | per-grain dislocation density (radial-breadth, FCC b) + modified WH (contrast-corrected) |
| `contrast_factor` | **core** | anisotropic dislocation contrast factors C̄_hkl (Stroh/sextic, ANIZC); cubic; validated to the silver C=0.3843 worked example; symmetry-general single-dislocation core |
| `contrast_factor_hex` | **core** | hexagonal contrast factors (Dragomir & Ungár 2002): 11 sub-slip-systems, C̄=C̄_{hk.0}(1+q₁x+q₂x²); validated to Table 2 (Ti+Zr, all systems ≤~3%) |
| `burgers_population` | **core** | ⟨a⟩/⟨c⟩/⟨c+a⟩ Burgers-vector-type fractions from measured (q₁,q₂); reproduces deformed-Ti 75/20/5 % |
| `defect_tests` | **core** | forbidden-reflection, fault-rod (explicit), fault-α |
| `examples` | **core** | end-to-end inventory driver + notebook |
| `rod_detect`, `asterism_fit`, `subgrain`, `delta_pdf`, `forward_sim`, `seed_index` | *experimental* | synthetic-tested; the deformation-physics layer / future genuine-CuAl₂ rod data |

`defect_tests.rod_family_enrichment` is a **screening** metric only — confounded by
reciprocal-lattice geometry; use `fault_rod_alignment` (explicit per-grain) for the
authoritative ⟨111⟩ fault-rod test.

## Driving dataset

Sep-2025 1-ID-E beamtime — 10 Y-layers × 1440 ω-frames @ 0.25°/frame on Pilatus3
CdTe 2M, 71.676 keV (λ = 0.172979 Å). **Phase: FCC Cu(-rich solid solution),
a = 3.6356 Å** (space group 225). θ-Al₂Cu (CuAl₂, I4/mcm, a ≈ 6.066 Å) was the
a-priori candidate given the Cu–Al provenance and was **tested and eliminated**
from the powder line-out (the 1/d² ring sequence is the exact FCC fingerprint;
CuAl₂'s strongest (110) line at d = 4.29 Å is absent). The validated
gold-calibrant detector geometry (Lsd 652.7 mm, tilts + distortion, correct ω
sign) is the default in `geometry.demk_default_geometry()`.

Published analysis (the worked validation case for this package):
`~/Desktop/analysis/demk/fcc_reanalysis/` (FINDINGS.md + figures + scripts).

## Engineering contract

All modules satisfy ALL four:

1. **Differentiable**. Every physics / scoring / fitting routine is torch;
   numpy / scipy live only in `_discrete` helpers off the gradient path.
2. **Device portable**. CPU / CUDA / MPS via `midas_transforms.device.resolve_device`.
3. **Thoroughly tested**. Per module: synthetic + autograd + device + **real-data
   regression** (reproduces a published demk number) + benchmark.
4. **Reuses upstream MIDAS**. No re-ported orientation, lattice, transform, or
   device-resolution code. See the reuse map in `implementation_plan.md`.
