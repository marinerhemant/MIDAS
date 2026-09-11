# Phase 5 — I(Q) → S(Q) → G(r)

> Part of the **pdf doc set**. Spine: [`README.md`](README.md).
> The spine keeps the gates; this keeps the recipe. The code is lifted from
> `$ANALYSIS/scripts/04_sq_gr.py`, which ran on the reference beamtime, with names made generic.

Inputs from §4: a cake `mean2d[eta, R]`, its `sigma2d`, the pixel counts `npix2d`, the Q of each
R bin, the net ion-chamber counts for every frame, and the stated sample facts.

## §5.1 One profile per frame, on a uniform grid

```python
import numpy as np

def one_d(mean2d, sigma2d, npix2d, Q_axis, keep_eta_rows=None):
    """Pixel-count-weighted eta mean. keep_eta_rows (bool per eta row) cuts a wedge or a sliver set."""
    w = npix2d.astype(float)
    if keep_eta_rows is not None:
        w = np.where(keep_eta_rows[:, None], w, 0.0)
    ok = np.isfinite(mean2d) & np.isfinite(sigma2d) & np.isfinite(w) & (w > 0)
    w = np.where(ok, w, 0.0)
    W = w.sum(0)
    I = (w * np.where(ok, mean2d, 0.0)).sum(0) / np.where(W > 0, W, np.nan)
    s = np.sqrt((w * w * np.where(ok, sigma2d, 0.0) ** 2).sum(0)) / np.where(W > 0, W, np.nan)
    return Q_axis, I, s

q = np.arange(Q_MIN, Q_MAX + DQ / 2, DQ)          # UNIFORM: the transform assumes it (hard rule 3)
def to_grid(Q, y):
    ok = np.isfinite(y)
    return np.interp(q, Q[ok], y[ok], left=np.nan, right=np.nan)

I_s = to_grid(Qs, Is) / ic_ratio_sample           # net C/IC1 of the frame ÷ that of the air frame
s_s = to_grid(Qs, ss) / ic_ratio_sample
I_e = to_grid(Qe, Ie) / ic_ratio_empty             # the empty container (or air, for the container)
s_e = to_grid(Qe, se) / ic_ratio_empty
```

Reference settings: Q 1.0–21.0 Å⁻¹, dQ 0.01. If either profile does not cover the grid, **stop**.
This happens to a narrow η wedge on an edge-centred beam.

## §5.2 Container subtraction and absorption — Paalman-Pings, cylinder in cylinder

```python
from midas_pdf.corrections import linear_attenuation_um, paalman_pings_cylinder_in_cylinder

mu_s = linear_attenuation_um(MASSES_SAMPLE, lam, density_g_cm3=DENSITY * PACKING)   # 1/µm
mu_c = linear_attenuation_um(MASSES_CONTAINER, lam, density_g_cm3=DENSITY_CONTAINER)
pp = paalman_pings_cylinder_in_cylinder(q, wavelength_A=lam, mu_sample_um=mu_s, mu_container_um=mu_c,
                                        R_sample_um=R_S, R_container_um=R_S + WALL, n_grid=48)
ratio = (pp["A_c_sc"] / pp["A_c_c"]).numpy()
A = pp["A_s_sc"].numpy()
I_c = (I_s - ratio * I_e) / A
s_c = np.sqrt(s_s ** 2 + (ratio * s_e) ** 2) / A
```

`MASSES_*` are `{element: count × atomic mass}`. For the empty container itself, subtract air with
no absorption term.

**Check** that `(I_c < 0).sum()` is 0. A negative after subtraction means the flux ratio or the
container geometry is wrong. Reference values: package μ/ρ matched xraylib `CS_Total` to four
digits; Ni at bulk density with R 500 µm gave μR 0.50 and A_s,sc flat at 0.436–0.441 over Q 1–21.

## §5.3 Multiple scattering

```python
from midas_pdf import Composition, cylinder_effective_tau, slab_transport_ms, ms_background_on_grid
from midas_pdf.ms import slab_optical_params

comp = Composition(COUNTS)                         # {element: count}
mu_um, _tau, albedo = slab_optical_params(comp, wavelength_A=lam, thickness_um=2 * R_S,
                                          number_density_A3=RHO0, packing_fraction=PACKING)
tau = cylinder_effective_tau(mu_um, R_S)
ms = slab_transport_ms(comp, wavelength_A=lam, tau=tau, albedo=albedo, q_max=float(q.max()))
beta = ms_background_on_grid(q, np.ones_like(q), ms).numpy()
I_m, s_m = I_c * (1.0 - beta), s_c * (1.0 - beta)
```

`refine_normalization` has no `background=` input, so the multiple-scattering fraction comes out
of the intensity before normalisation. Reference β medians: Ni 0.047, CeO2 0.026, IPA 0.0085,
Kapton 0.0009.

## §5.4 Normalisation — anchor on the tail, keep the low-r line as a test

```python
import torch
from midas_pdf import refine_normalization, i_of_q_to_Gr, faber_ziman_S

r_out = torch.arange(0.01, 30.0 + 1e-9, 0.01, dtype=torch.float64)
r_fit = torch.arange(0.01, 20.0 + 1e-9, 0.01, dtype=torch.float64)
q_t = torch.as_tensor(q)
_f, f2 = comp.form_factor_averages(q_t)
cmp = comp.compton(q_t, wavelength_A=lam)          # Hubbell + Breit-Dirac, k = 2 (package default)
hiq = q >= 18.0
init = float(((f2 + cmp)[torch.as_tensor(hiq)]).mean() / np.mean(I_m[hiq]))
rn = refine_normalization(q, I_m, comp, r_fit, wavelength_A=lam, number_density=RHO0,
                          sigma_intensity=s_m, compton=True, q_max=Q_MAX, window="lorch",
                          r_min_phys=LOWR_HI, q_asymptote_frac=(Q_MAX - 18.0) / (Q_MAX - Q_MIN),
                          init_scale=init, fit_background=False, w_lowr=0.0, steps=100)
G, sG, S = i_of_q_to_Gr(q, I_m, comp, r_out, wavelength_A=lam, scale=rn["scale"], compton=True,
                        sigma_intensity=s_m, q_max=Q_MAX, window="lorch")
_S, sS = faber_ziman_S(I_m, q, comp, wavelength_A=lam, scale=rn["scale"], compton=True, sigma_intensity=s_m)
```

- **`w_lowr=0.0` with the anchor on [18, Q_max].** The package default (`w_lowr=1.0`,
  `packages/midas_pdf/midas_pdf/refine.py:64`) makes the −4πρ₀r line a fit target, so §6's
  low-r check can no longer fail.
- **Window.** `"lorch"`, or `"rect"` / `"none"` for none
  (`packages/midas_integrate_v2/midas_integrate_v2/pdf.py:172-177`). Any other string raises.
- **To vary Q_max for the uncertainty,** truncate the transform only and keep the scale from
  the full anchor: `i_of_q_to_Gr(q[q <= 18], I_m[q <= 18], ..., q_max=18.0)`. An [18, 18] anchor
  is empty.
- **Compton.** Pass a tensor from `comp.compton(q_t, wavelength_A=lam, k=3)` to change the
  Breit-Dirac exponent, or `method="it94"` for the other table. On the reference data k = 3 made
  both liquids worse, and it94 moved G(r) by a small fraction of the first peak.
- **Uncertainty bands.** `sG` inherits the per-pixel error model and assumes independent Q bins.
  Until the sliver test passes, label it uncalibrated (hard rule 6).

## §5.5 The other correlation functions, and what the frame emits

```python
from midas_pdf import expected_fluorescence
from midas_pdf.conventions import (structure_function_F, pair_distribution_g,
                                   total_correlation_T, radial_distribution_R)
F = structure_function_F(q, S)[0]
g = pair_distribution_g(r_out, G, number_density=RHO0)[0]
T = total_correlation_T(r_out, G, number_density=RHO0)[0]
Rr = radial_distribution_R(r_out, G, number_density=RHO0)[0]
fluor = expected_fluorescence(list(COUNTS), wavelength_A=lam, min_yield=0.05)   # diagnostic only
```

Fluorescence is reported, not subtracted. Ce K (edge 40.44 keV, Kα₁ 34.72 keV) is excited at
67 keV, and a flat additive background in I(Q) becomes a *rising* S(Q), because ⟨f²⟩ falls with Q.

## §5.6 Arms to run beside the headline, every time

| arm | how | what it discriminates |
|---|---|---|
| uncorrected | the same chain on `mean2d_uncorrected` | whether the intensity corrections reached the profile (rule 1) |
| package-default normalisation | `w_lowr=1.0, fit_background=True, bg_order=0` | how much the low-r constraint is hiding |
| no window | `window="none"` | window broadening against the model (rule 8) |
| FT Q_max 18 | the truncation above | termination |
| detector efficiency (report-only) | `I / detector_efficiency(q, wavelength_A=lam, material=..., thickness_um=T, density_g_cm3=...)` | the obliquity term — at the data-sheet thickness only (rule 14) |

Reference outputs: `$ANALYSIS/out/pdf_sep11_none_map_q21_lorch*/summary.json`.
