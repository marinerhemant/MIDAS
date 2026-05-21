# `midas-integrate-v2` notebooks

Hands-on tutorials for someone new to detector calibration / radial
integration in HEDM. **Run them in order**; each builds on the
previous.

| # | Notebook | Time | What you'll learn |
|---|----------|------|-------------------|
| 01 | [Your First Diffraction Pattern](01_your_first_diffraction_pattern.ipynb) | 15 min | Load a real Pilatus CeO₂ frame; build an `IntegrationSpec` from `parameters.txt`; integrate; identify CeO₂ rings via Bragg's law. |
| 02 | [Geometry — How Each Parameter Shapes the Pattern](02_geometry_what_each_parameter_does.ipynb) | 20 min | Visualise what `BC`, `Lsd`, `λ`, tilts, and distortion each do to a synthetic ring pattern. The intuition behind calibration. |
| 03 | [Refining Calibration with Joint Optimization](03_refining_calibration_with_joint_optimization.ipynb) | 25 min | The headline v2 workflow: perturb `BC_y` by 1 px, refine back via `EtaUniformityLoss` + Adam. Then multi-parameter (`BC_y, BC_z, Lsd`) joint refinement on real Pilatus data. |
| 04 | [Multi-Distance Calibration](04_multi_distance_calibration.ipynb) | 15 min | Synthesise 3 frames at 3 Lsds, share a single refinable `BC_y` tensor across them, refine jointly via `MultiImageLoss`. |
| 05 | [End-to-End: `midas-calibrate-v2` to `midas-integrate-v2`](05_end_to_end_calibrate_v2_to_integrate_v2.ipynb) | 15 min | The production handoff: bridge a v2 calibration result into an integration pipeline via `to_integrate_params`. |
| 06 | [Writing Your Own Loss Function](06_writing_your_own_loss_function.ipynb) | 20 min | The three rules of a custom loss; build a `PeakSharpnessLoss`; combine with `EtaUniformityLoss` and a `GaussianPriorLoss`; write a robust Huber variant. |
| 07 | [Bayesian Calibration with Laplace Uncertainty](07_bayesian_calibration_with_laplace_uq.ipynb) | 25 min | Refine to MLE; compute the Hessian via PyTorch autograd; invert for the parameter covariance; visualise the joint BC_y/BC_z 2σ ellipse; bootstrap-validate the error bars. |
| 08 | [From Integrated Profile to PDF Analysis](08_from_integrated_profile_to_pdf.ipynb) | 25 min | The downstream pipeline: R(px) → Q(Å⁻¹), background subtraction, Fourier-transform S(Q) → G(r), peak-match against CeO₂ atom-atom distances. |
| 09 | [Production Workflow: Mask + Variance + XYE Export](09_production_workflow.ipynb) | 25 min | The deployable pipeline: auto-build a bad-pixel mask, integrate with variance propagation (per-bin σ), write XYE/DAT/2D-CSV outputs with embedded provenance metadata for reproducibility months later. |
| 10 | [Differentiable Mask: Auto-detect Bad Pixels](10_differentiable_mask_auto_bad_pixel.ipynb) | 25 min | The MIDAS differentiator. Make the per-pixel mask itself a learnable parameter; train it jointly with η-uniformity loss + sparsity prior; the optimiser zeros out planted hot pixels while keeping good pixels at weight 1. No equivalent in pyFAI. |
| 11 | [Sweep-mode Batch Processing](11_sweep_mode_batch_processing.ipynb) | 25 min | Production throughput: per-pixel temporal cosmic-ray rejection, per-frame normalisation (monitor / exposure / transmission), out-of-core HDF5 streaming, single-HDF5 output with provenance. |
| 12 | [APS PDF + Operando Demo](12_aps_pdf_operando_demo.ipynb) | — | Live PDF reconstruction during operando scans |
| 13 | [Parasitic Spots on Multi-Panel](13_parasitic_spot_detection_multi_panel.ipynb) | — | Detect single-crystal parasitic spots on a Pilatus 2M cake |
| 14 | [Real-Data Parasitic-Spot Detection](14_real_data_calibrate_then_clip.ipynb) | — | Calibrate first, then clip; Pilatus 2M @ APS 1-ID-E |
| 15 | [σ-aware XRD-CT Reconstruction](15_xrdct_sigma_aware_reconstruction.ipynb) | — | XRD-CT with variance propagation through the reconstruction |
| 16 | [`LearnableGain` — auto-recover spatial gain drift](16_learnable_gain_recovery.ipynb) | — | Drift-track gain across a long beamtime |
| 17 | [σ Chain: Pixel → MIDAS → TOPAS Rietveld](17_sigma_to_rietveld.ipynb) | — | Honest σ propagation for refinable weights |
| 18 | [`EmptySubtraction` with refinable scale](18_empty_subtraction_refinable_scale.ipynb) | — | Empty-cell background subtraction, no trial-and-error |
| 19 | [`fit_drift_trajectory` — operando Lsd / BC drift](19_fit_drift_trajectory_operando.ipynb) | — | Bayesian σ on time-varying calibration |

### Tier A — core capability coverage (recipes for everyday workflows)

| # | Notebook | Coverage |
|---|---|---|
| 20 | [Binning Method Comparison](20_binning_method_comparison.ipynb) | `binning.{hard,soft,subpixel,polygon}` — accuracy/speed/grad-through tradeoffs |
| 21 | [pyFAI Handoff](21_pyfai_handoff.ipynb) | `compat.pyfai.{bc_to_poni, poni_to_bc, make_pyfai_integrator}` |
| 22 | [Physics Corrections — Polarization, Solid Angle, Compton](22_physics_corrections_pol_sa_compton.ipynb) | `corrections.{PolarizationCorrection, SolidAngleCorrection, ComptonSubtraction}` |
| 23 | [Q-Uniform Binning](23_q_uniform_binning.ipynb) | `corrections.build_q_bin_edges_in_R` — PDF/Rietveld-ready bin edges |
| 24 | [First-Time Auto-Bootstrap](24_first_time_auto_bootstrap.ipynb) | `bootstrap.{estimate_BC_from_image, estimate_initial_spec}` + `ring_detect.{detect_rings, suggest_material}` |
| 25 | [`integrate_with_corrections` One-Call Recipe](25_integrate_with_corrections_one_call.ipynb) | `corrections.integrate_with_corrections` + variance-aware variants |

### Tier B — domain-specific capabilities

| # | Notebook | Coverage |
|---|---|---|
| 26 | [Cylindrical Absorption (capillary / in-situ)](26_cylindrical_absorption_capillary.ipynb) | `corrections.CylindricalAbsorption` |
| 27 | [DAC Gasket Masking](27_dac_gasket_masking.ipynb) | `dac.{build_gasket_mask, eta_coverage_per_ring}` |
| 28 | [Streaming Outlier Rejection](28_streaming_outlier_rejection.ipynb) | `streaming.outlier.{reject_cosmic_rays, reject_spatial_spikes, azimuthal_sigma_clip*}` |
| 29 | [Pole Figure Construction](29_pole_figure_construction.ipynb) | `texture.{cake_to_pole_figure, write_popla_pol}` |
| 30 | [GISAXS + Inelastic Regrid](30_gisaxs_and_inelastic_regrid.ipynb) | `grazing.gisaxs.*` + `inelastic.regroup_eta_R_E_to_Q_E` |
| 31 | [Multi-Detector Streaming](31_multi_detector_streaming.ipynb) | `streaming.multi_detector.integrate_multi_detector` |
| 32 | [Energy-Sweep Pipeline](32_energy_sweep_pipeline.ipynb) | `pipelines.energy_sweep.run_energy_sweep` |

### Migration

| # | Notebook | Coverage |
|---|---|---|
| 33 | [Migrating from v1 to v2](33_v1_to_v2_migration.ipynb) | `compat.spec_from_v1_paramstest` / `spec_from_v1_params` / `v1_params_from_spec`; reusing `Map.bin` + `residual_corr.bin`; drop-in production recipe; when to stay on v1 |
| 34 | [HYDRA Multi-Detector (real data)](34_hydra_multi_detector_real_data.ipynb) | `io.MILKMultiGeometryAdapter` — four azimuthal GE panels (shared L_sd, per-panel BC + `tx`) merged onto one 2θ axis; per-panel overlay; needs `$V2_HYDRA_BASE` + calibrate-v2 NB 23 |
| 35 | [Azimuthal Peak Fitting (per-η)](35_azimuthal_peak_fitting_per_eta.ipynb) | `build_geometry` + `integrate` → 2-D cake `(n_r, n_eta)`; auto-fit the ring in every η-bin (Gaussian+bg); peak 2θ/intensity/FWHM vs η → strain & texture |

## Prerequisites

- Some Python familiarity (loops, dataclasses, plotting). No prior MIDAS
  knowledge assumed.
- The `midas` Python environment with these packages installed:
  - `midas-integrate` ≥ 0.4.0
  - `midas-calibrate-v2` ≥ 0.1.0
  - `midas-integrate-v2` ≥ 0.1.0
  - `tifffile`, `matplotlib`, `numpy`, `torch`
- The example dataset at
  `FF_HEDM/Example/Calibration/CeO2_Pil_100x100_..._001956.tif` (ships
  with MIDAS).

Activate the env then launch JupyterLab:

```bash
source /Users/hsharma/miniconda3/bin/activate midas_env
cd packages/midas_integrate_v2/notebooks
jupyter lab
```

## How to read these

Each notebook is structured the same:

1. **What you'll learn** — concrete deliverables.
2. **Background** — the physics / math behind the operation.
3. **Step-by-step** — code cells with explanations of every line.
4. **Try it yourself** — exercises to reinforce the concept.
5. **Common pitfalls** — what goes wrong in practice.
6. **Next** — link to the next notebook in the series.

If a cell fails, check that `KMP_DUPLICATE_LIB_OK=TRUE` is in your
environment (a torch/numba OpenMP interaction we work around).

## Regenerating the notebooks

The `.ipynb` files are generated from `_build_NN.py` scripts so the
content lives in version-controlled Python:

```bash
cd notebooks
for n in 00_tour 01 02 03 04 05 06 07 08 09 10 11 12 13 14 16 17; do
    python _build_${n}.py
done
```

(Notebooks 18 and 19 are currently hand-authored and have no `_build` script.)

Then re-execute via `jupyter nbconvert --to notebook --execute
--inplace <notebook>.ipynb` to refresh the embedded outputs.

## More

- The package README (`../README.md`) has the full API reference.
- v0.x notebooks for the calibration-side companion package live at
  `../midas_calibrate_v2/notebooks/` (Bayesian calibration, Stage-4
  spline, NUTS sampling, etc.).
