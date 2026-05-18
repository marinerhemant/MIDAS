# `midas_calibrate_v2` Notebooks

Hands-on, end-to-end workable examples for the
[`midas_calibrate_v2`](../README.md) package. Every notebook
loads a real synchrotron calibrant image (or builds a clean
synthetic ground-truth experiment) and runs the actual production
code paths — they are not toy examples.

## Prerequisites

Activate the `midas_env` conda environment and confirm the test
data is mounted:

```bash
source /Users/hsharma/miniconda3/bin/activate midas_env
ls $V2_TEST_BASE/refined_MIDAS_params_Ceria_63keV_900mm_*.txt
```

If `V2_TEST_BASE` is unset, the notebooks default to
`/tmp/midas_v2_test`.

Optional dependencies:
- `pip install pyro-ppl` — for **NB 11** (NUTS posterior sampling)
- `pip install pyFAI` — for **NB 10** to run pyFAI live (otherwise
  it falls back to paper-validated comparison numbers)

## The 20 notebooks

### Tier 0 — the one-shot entry point

| # | Notebook | Wall time | What you'll learn |
|---|---|---|---|
| **00** | [One-Shot Fully-Automated Calibration](00_one_shot_calibration.ipynb) | ~5 min | The `calibrate(image, wavelength, pxY, ...)` function: seed → refine → residual map → binary in one call. **Start here if you just want a calibration.** |

### Tier 1 — core capabilities

| # | Notebook | Wall time | What you'll learn |
|---|---|---|---|
| **01** | [Getting Started](01_getting_started.ipynb) | ~30 s | Load image → `autocalibrate_pv` → MAP → Laplace σ → reliability gates |
| **02** | [Bayesian Uncertainty](02_bayesian_uncertainty.ipynb) | ~45 s | Full Laplace covariance; per-parameter σ in physical units; σ(Q) propagation for downstream PDF/Rietveld |
| **03** | [Multi-Panel Pilatus](03_multi_panel_pilatus.ipynb) | ~45 s | Σ=0 gauge on 48-module Pilatus3 2M-CdTe; per-panel σ collapse by 10 orders of magnitude |
| **04** | [Refining Pixel Size & Wavelength](04_refining_pixel_size_and_wavelength.ipynb) | ~2 min | The (L_sd, p_x) gauge null and the S5 protocol that breaks it with a Gaussian prior |
| **05** | [Reliability Gates and F2](05_reliability_gates_and_F2.ipynb) | ~3 min | All three gates run live; reproduce the paper's "+95% CV-gate finding" on Varex CeO₂; per-ring `δr_k` (F2) fix reduces strain 26–39% |

### Tier 2 — specialty workflows

| # | Notebook | Wall time | What you'll learn |
|---|---|---|---|
| **06** | [First-time Calibration](06_first_time_calibration.ipynb) | ~2 min | Auto-seed from ONLY material+wavelength+image — no paramstest, no manual peak picking, no L_sd guess |
| **07** | [Multi-distance Lattice Precision](07_multi_distance.ipynb) | ~5 s | Analytical Fisher: σ(a) collapses 6 → 2.8 ppm with two distances. Why a tight L_sd prior alone *doesn't* help. |
| **08** | [Doublet Calibrants](08_doublet_calibrants.ipynb) | ~10 s | Cu Kα₁/Kα₂ within-ring doublet bias quantified; `fit_doublet_pairs` co-fit removes 250 µε of fake strain |
| **09** | [Basis Extension + BIC](09_basis_extension_and_BIC.ipynb) | ~30 s | The 15-coef basis explained as `HarmonicTerm` tuples; how BIC decides whether extra η-folds help |
| **10** | [pyFAI Head-to-Head](10_pyfai_head_to_head.ipynb) | ~30 s | Same image, two pipelines; v2 strain 18.5 µε vs pyFAI 1105 µε on Varex CeO₂ |

### Tier 3 — advanced / paper-companion

| # | Notebook | Wall time | What you'll learn |
|---|---|---|---|
| **11** | [NUTS vs Laplace](11_nuts_vs_laplace.ipynb) | ~5 min | When the Laplace approximation breaks; reproduce the paper's "Laplace under-counts σ by 1.5–17× under priors" finding |
| **12** | [Cone-aware Seed for Tilted Detectors](12_cone_aware_seed_for_tilts.ipynb) | ~5 s | Synthetic POC: ellipse fit + (2θ→0) extrapolation recovers BC sub-pixel at every tilt up to 15° |
| **13** | [Henke vs Parallax Disentangling](13_henke_disentangling.ipynb) | ~5 s | Multi-energy Fisher analysis showing d_eff and δ_henke ARE separable at single energy; the v2 limit is parametric |
| **14** | [σ(Q) for PDF / Rietveld](14_sigma_q_for_pdf.ipynb) | ~45 s | Per-Q-bin σ propagation; calibration σ vs PDF bin width; when calibration σ matters |

### Tier 4 — feature deep-dives + cross-package handoff

| # | Notebook | Wall time | What you'll learn |
|---|---|---|---|
| **15** | [Empirical Residual-Correction Map](15_residual_correction_map.ipynb) | ~3 min | What ΔR(Y, Z) is, how `build_residual_corr_map` fits an RBF, before/after strain on a real CeO₂ image, save/load v1-compat binary |
| **16** | [Calibrate → Integrate Handoff](16_calibrate_to_integrate_handoff.ipynb) | ~5 min | `calibrate()` → `IntegrationSpec` → `midas_integrate_v2`. Same `residual_corr.bin` consumed by all three packages + the C tool. Forward-model parity check at a single pixel. |
| **17** | [Hex-Pixel Detectors (PIXIRAD)](17_hex_lattice_pixirad.ipynb) | ~30 s | `lattice='hex_offset_y'` with refinable Apothem + LatticeOrientation; synthetic round-trip on a hex grid |
| **18** | [Non-Cubic Calibrants](18_non_cubic_calibrant.ipynb) | ~10 s | The `CALIBRANTS` dict, trigonal d-spacing branch for Al₂O₃, custom-material dict form, adding a calibrant to the global database |
| **19** | [Conv-NN Residual Augmenter](19_nn_residual_augmenter.ipynb) | ~3 min | `autocalibrate_nn` two-stage training, the per-harmonic drift report (anti-cheating diagnostic), when **not** to use this on top of NB 15 |
| **20** | [Joint Forward-Cake Engine](20_joint_cake_engine.ipynb) | ~5 min | `autocalibrate_joint` — geometry + per-(ring, η) pseudo-Voigt shape DOFs refined jointly on raw radial windows. No centroid extraction; no centroid bias. The escalation path when alternating + residual map still leaves >100 µε |

## Running them

Open in Jupyter or VS Code:

```bash
cd /Users/hsharma/opt/MIDAS/packages/midas_calibrate_v2/notebooks
jupyter lab 01_getting_started.ipynb
```

Or batch-execute (the smoke test for "does v0.2.0 still work end-to-end"):

```bash
for nb in 0*.ipynb 1[0-4]*.ipynb; do
    jupyter nbconvert --to notebook --execute --inplace "$nb" \
        --ExecutePreprocessor.timeout=1800
done
```

Total wall time: ~25 min for all 14 (the multi-LM and NUTS notebooks
dominate).

## How the notebook source is organised

The `.ipynb` files are **build artefacts**, not source.
[`_build.py`](_build.py) is the source of truth — each notebook is
a list of `(kind, source)` cells maintained as Python strings,
which makes them diffable, reviewable, and refactorable.

To rebuild after editing `_build.py`:

```bash
python _build.py                           # rebuild everything
python _build.py 06_first_time_calibration  # rebuild one
```

## Recommended reading order

If you're new to the framework, do the Tier 1 notebooks in order
(01 → 02 → 03 → 04 → 05).  These cover the framework's core
capabilities and the paper's main findings.

For specific use cases, jump directly:

| Your situation | Start with |
|---|---|
| **Just give me a calibration** | **00** |
| First time on a new beamline | **06** + **00** |
| Need absolute lattice constants | **07** |
| Lab Cu source or LaB₆ at small L_sd | **08** |
| CV gate fired and you want to know what to do | **09** + **05** |
| Comparing v2 to your existing pyFAI workflow | **10** |
| Need defensible σ values for a paper | **11** |
| Tilted detector (>5°) | **12** |
| Working with PDF / total scattering | **14** |
| Want the last 50–100 µε that harmonic distortion misses | **15** |
| Apply v2 calibration to a sample image | **16** |
| Have a hex-pixel detector (PIXIRAD etc.) | **17** |
| Non-cubic calibrant (Al₂O₃ etc.) or custom material | **18** |
| Pathological per-pixel residuals harmonics can't capture | **19** (after **15**) |
| Centroid bias from asymmetric peaks / doublets | **20** (after **15**) |

## Common gotchas

- **Image flip convention** — Most synchrotron TIFFs need
  `image[::-1, :].copy()` to match the MIDAS coordinate system.
  The notebooks include this; double-check on your beamline.

- **Refining without a prior** — On a single image at single
  energy, refining `pxY`/`pxZ` or `Wavelength` creates a gauge
  null. You'll see σ saturate at the regularisation ridge floor.
  Notebook **04** walks through the fix.

- **`reuse_fits=True` vs `False`** — Recommended `True` for clean
  calibrant images. `False` re-extracts the cake every outer iter
  and is slower; only useful if your seed is so bad the initial
  fits are nonsense.

- **`auto_max_ring=True` vs explicit** — The auto-detector usually
  picks a sensible ring set, but if you're benchmarking
  reproducibly across runs, set `spec.max_ring_number` explicitly
  and pass `auto_max_ring=False`.

- **Laplace σ under priors** — Under heavy Gaussian priors on
  parameters, the Laplace approximation under-counts σ by 1.5–17×
  on the v0.2.0 LM closure (notebook **11** explains why).  Use
  NUTS in that regime until the prior-row σ_r rescaling fix lands.

## Contributing

Notebooks should be **executable end-to-end** against the test
data. The `nbconvert --execute --inplace` step (which we run as
part of CI) catches accidental regressions.  If a notebook fails:

1. Check `$V2_TEST_BASE` is set correctly and the calibrant image +
   paramstest exist there.
2. Edit the relevant `NB_xx` list in `_build.py` (NOT the .ipynb
   directly — your changes will be wiped on next rebuild).
3. Rebuild with `python _build.py <name>`.
4. Re-execute and verify.

To add a new notebook: add an `NB_15` (etc.) cell list to
`_build.py`, register it in the `NOTEBOOKS` dict at the bottom,
rebuild, and update this README.
