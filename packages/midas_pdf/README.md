# midas-pdf

**Differentiable, error-propagating total-scattering / pair-distribution-function
(PDF, *G(r)*) pipeline.**

`midas-pdf` is a deliberately thin layer. Almost everything it needs already
exists elsewhere in MIDAS and is reused rather than reimplemented:

| Stage | Provided by | Status |
|-------|-------------|--------|
| detector geometry + wavelength (+ covariance) | `midas-calibrate-v2` | existing |
| pixels → I(Q) with σ (polygon-exact, pol/solid-angle/dark) | `midas-integrate-v2` | existing |
| atomic form factors f(Q), anomalous f′,f″ (differentiable) | `midas-hkls` | existing |
| Compton / incoherent subtraction | `midas-integrate-v2.corrections.compton` | existing |
| S(Q) → G(r) sine FT **with σ propagation** | `midas-integrate-v2.pdf` | existing |
| **polyatomic Faber-Ziman normalization** ⟨f²⟩, ⟨f⟩² | **`midas-pdf` (new)** | this package |
| **Δ-PDF (difference PDF) for time-resolved/operando** | **`midas-pdf` (new)** | this package |

The single piece that did not exist anywhere was the **composition layer**: the
existing `midas_integrate_v2.pdf.normalize_to_S` is *monoatomic* (it divides by a
single ⟨f²⟩). Real total scattering of a polyatomic sample needs the Faber-Ziman
form, which requires both ⟨f²⟩(Q) and ⟨f⟩²(Q) built from the sample composition.
That bridge — and the Δ-PDF helper — is all `midas-pdf` adds.

## What is novel

Every arrow in the chain is a torch operation carrying a 1σ uncertainty, so the
pipeline is **end-to-end differentiable** *and* **end-to-end error-propagating** —
a combination no production total-scattering tool (PDFgetX3 / PDFgetN / GudrunX)
offers. Concretely this enables:

* gradient-based **normalization refinement** (`refine.py`): `scale`/`offset`/ρ₀
  are fit by L-BFGS against model-free physics (⟨S⟩→1 at high Q, G(r)=−4πρ₀r at
  low r) — the "ad hoc scale twiddling" of PDF analysis becomes an optimization;
* an analytic 1σ band on every G(r) point, **validated** against a Monte-Carlo
  bootstrap to <1% (`dev/demo_sigma_validation.py`) — error propagation is the
  one thing existing software tends to drop;
* statistically-meaningful **Δ-PDF** difference maps for time-resolved studies:
  σ²(ΔG) = σ²(G₁) + σ²(G₂), so a feature change can be tested against noise;
* differentiability in atomic positions (`validate.py`, Debye equation), so the
  same code is a forward model for structure refinement against G(r).

## Modules

| Module | Contents |
|--------|----------|
| `composition.py` | `Composition` → ⟨f⟩(Q), ⟨f²⟩(Q), Laue term, Compton |
| `compton.py` | Hubbell tabulated incoherent scattering + Breit-Dirac recoil |
| `corrections.py` | Q-dependent detector efficiency, flat-plate self-absorption (MAC-backed) |
| `fluorescence.py` | `expected_fluorescence`: which elements fluoresce at a given energy |
| `multiple_scattering.py` | `lumped_background`: Tier-1 smooth MS/fluorescence/air background |
| `cross_section.py` | `differential_cross_section`: per-atom dσ/dΩ(Q) (MS engine) |
| `ms.py` | first-principles MS: analytic single + double scattering, Monte-Carlo references (slab + cylinder) |
| `ms_transport.py` | all-orders MS by differentiable discrete-ordinates radiative transfer (slab) |
| `structure.py` | differentiable small-box PDF (PDFfit-style) forward model + error-aware refinement |
| `normalize.py` | `faber_ziman_S`: I(Q) → S(Q) with σ (and lumped `background`) |

Runnable, one-per-capability demonstrations live in [`examples/`](examples/).
| `gr.py` | re-export of the reused sine FT (S(Q) → G(r) with σ) |
| `pipeline.py` | `i_of_q_to_Gr`: I(Q) → G(r) end to end |
| `frontend.py` | `image_to_iq`, `image_to_Gr`: detector pixels → G(r) |
| `conventions.py` | `structure_function_F`, `pair_distribution_g`, `total_correlation_T`, `radial_distribution_R` |
| `refine.py` | `refine_normalization`: differentiable scale/offset/ρ₀ fit |
| `deltapdf.py` | `delta_pdf`, `significant_mask`: difference PDF + n-σ test |
| `validate.py` | `debye_scattering_intensity`, `synthetic_powder_image`: model-free references |

## Quick start

```python
import torch
from midas_pdf import Composition, i_of_q_to_Gr

comp = Composition({"Si": 1, "O": 2})          # SiO2, number fractions
q = torch.linspace(0.5, 25.0, 2000, dtype=torch.float64)
r = torch.linspace(0.0, 10.0, 1000, dtype=torch.float64)

# I_q, sigma_I come from midas-integrate-v2 (pixels -> I(Q) with sigma)
G, sigma_G, S = i_of_q_to_Gr(
    q, I_q, comp, r,
    wavelength_A=0.1665, sigma_intensity=sigma_I,
    compton=True, q_max=22.0,
)
```

## Conventions

Default is the **Faber-Ziman** total structure factor (X-ray, neutral-atom
form factors), matching the PDFgetX3 default. `S(Q) → 1` as `Q → ∞`;
`G(r) = (2/π) ∫ Q[S(Q)-1] sin(Qr) W(Q) dQ`. Window defaults to Lorch.
Convention choice (FZ vs Keen; which of S/F/G/g/D/T to report) is intended to be
settled with the experimental collaborators — see `dev/PLAN.md`.

See `dev/PLAN.md` for the phased build plan and open items.
