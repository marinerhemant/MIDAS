# midas-pdf examples

One short, runnable script per capability. Each prints a clear result; a few
save a figure into `examples/figures/`. Run any of them with:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python examples/01_composition_and_form_factors.py
```

| # | Script | Capability |
|---|--------|------------|
| 01 | `01_composition_and_form_factors.py` | `Composition` → ⟨f⟩, ⟨f²⟩, Laue term |
| 02 | `02_normalize_to_SQ.py` | Faber-Ziman `faber_ziman_S`: I(Q) → S(Q) + σ |
| 03 | `03_iq_to_gr_with_uncertainty.py` | `i_of_q_to_Gr`: I(Q) → G(r) with a propagated 1σ band |
| 04 | `04_pixels_to_gr.py` | `image_to_Gr`: detector image → G(r) (slow: polygon binning) |
| 05 | `05_refine_normalization.py` | differentiable scale/background refinement |
| 06 | `06_correlation_functions.py` | F(Q), g(r), T(r), RDF family |
| 07 | `07_delta_pdf.py` | difference-PDF + n-σ significance (time-resolved) |
| 08 | `08_debye_validation.py` | Debye scattering equation → known-answer G(r) |
| 09 | `09_compton_hubbell.py` | Hubbell incoherent scattering + Breit-Dirac |
| 10 | `10_detector_efficiency_and_absorption.py` | Q-dependent detector efficiency, self-absorption |
| 11 | `11_fluorescence_diagnostic.py` | which sample elements fluoresce at an energy |
| 12 | `12_multiple_scattering.py` | Tier-1 lumped background; cross-section; MC + analytic single (Tier 2/3) |

All computations are torch and differentiable unless noted (the Monte-Carlo MS
reference in 12 is the one non-differentiable piece, by design).
