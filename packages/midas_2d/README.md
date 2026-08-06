# midas-2d

Differentiable diffraction for **2D / few-layer & ultrafast** experiments,
built on `midas-hkls` (structure factors, form factors, lattice math, HKL
enumeration) and `midas-pink` (energy spectrum). All forward paths are
torch-differentiable and run on CPU / CUDA / MPS.

Scope: ultrafast nanoplatelet diffraction, Bragg coherent diffraction imaging,
ML-assisted inversion, and MD-coupled forward modelling.

## Status: Phases 1-4 implemented (38 tests, MPS/CPU verified)

### Phase 1 -- finite-size forward core
- **`shape_factor`** -- Laue interference `|S(q)|^2 = prod sin^2(N_i pi x_i)/
  sin^2(pi x_i)` (thickness fringes / Laue oscillations), its `N -> inf`
  crystal-truncation-rod limit, `nanoplatelet_rod`. Differentiable in a
  real-valued layer count `N`, with finite/correct gradients at the Bragg peak.
- **`forward`** -- continuous-q structure factor `F(q)` (reuses `midas-hkls`
  form-factor / metric / DWF physics, no integer rounding), assembled into
  `I(q) = |F|^2 . |S|^2 . Lp`.
- **`energy`** -- swappable mono/pink/white via `midas_pink.ParameterisedSpectrum`.
- **`io`** -- zinc-blende CdSe builder.

### Phase 3 -- MD-coupled engine (`debye`)  *(the novel core)*
Diffraction straight from atomic coordinates, **differentiable w.r.t. the atom
positions**:
- `coherent_amplitude` / `coherent_intensity` -- oriented `|sum f e^{iQ.r}|^2`
  (reproduces the Phase-1 analytic fringes to 2e-3 -- cross-checked).
- `debye_intensity` -- orientationally-averaged Debye equation (colloidal case).
- `ensemble_intensity` -- average over MD frames; **the spread of coordinates
  *is* the disorder**, so anisotropic Debye-Waller falloff emerges with no DWF
  assumed.
- `io.cdse_supercell`, `io.load_xyz_frames` -- explicit structures / trajectories.

### Phase 2 -- transient anisotropic disorder, rocking, inversion, UQ
- **`disorder`** -- `AnisotropicMSD` / `TransientMSD` (fittable, time-resolved
  out-of-plane vs in-plane MSD) and `msd_tensor_from_frames` (read the same
  tensor off an MD trajectory -- closes the loop with `debye`).
- **`rocking`** -- `rocking_curve`, `fwhm`, `thickness_from_fwhm` (FWHM ~ 0.886/N),
  `reciprocal_space_map`; rod models from either the analytic or MD forward.
- **`inverse`** -- `fit` (Adam/L-BFGS), `cosine_loss` (smooth scale-invariant
  shape loss), `laplace_uncertainty` (Hessian-at-optimum error bars).

### Phase 4 -- coherent / BCDI (`coherent`, `bcdi`)
- `coherent_speckle`, `bcdi_forward` (`|FFT(psi)|^2`), and `phase_retrieval`
  (autograd phase retrieval inside a support -- the differentiable ER/HIO
  alternative and a slot for learned priors).  `loss=` selects the residual;
  the default `"amplitude"` matters, because an intensity-domain L2 over the
  many decades of a coherent pattern is dominated by the brightest voxels and
  stalls (`"intensity"`, `"poisson"` also available).
- `bcdi` -- **where the FFT lands on the detector.** The measured array is
  indexed by (detector column, detector row, rocking step), which spans a
  *sheared* parallelepiped in q, not a box: `q_basis` builds `B`,
  `conjugate_real_basis` gives the real-space grid the FFT actually uses
  (`B^T C = 2 pi diag(1/N)`), `oversampling` / `shear_angles_deg` diagnose the
  sampling, `detector_distance_for_oversampling` and
  `rocking_step_for_oversampling` size a scan, and `sheared_to_lab` removes the
  shear from the *reconstructed object* at the end -- never from the measured
  intensity before phasing.
- **Forward chain, differentiable end to end** (`bcdi`): `q_grid`,
  `object_to_amplitude` (`A = FFT(psi)`), `detector_signal` (|F_hkl|^2,
  polarisation/solid angle, partial coherence in the autocorrelation domain,
  flux) and `sample_counts`.  `detector_signal` returns the *expected rate* and
  is differentiable; `sample_counts` draws Poisson counts and is not.  To fit
  real data, hold the counts fixed and differentiate `poisson_nll` of the rate.
- **From atoms** (`bcdi`): `rotation_to_bragg` orients a crystal so a reflection
  satisfies Bragg in the lab frame, then either
  * `speckle_from_atoms` -- exact `sum_i f_i exp(iQ.r_i)`, no envelope or
    small-strain approximation, differentiable w.r.t. every coordinate.  Cost is
    `O(N_atoms * N_q)`, so it caps out near 10 nm (`atom_sum_cost` tells you); or
  * `atoms_to_object` -- bins the same coordinates in `O(N_atoms)` into
    `psi = occupancy * exp(-i G.u)`, which scales to real grains.  6.7 M atoms
    (60 nm) in ~1.6 s, against ~1.75e12 terms for the direct sum.  The two agree
    to `corr = 0.9997` on a 6 nm test crystal, with inverted (0.45) and
    shape-only (0.77) controls.
- **Reading someone else's data** (`bcdi_io`): `load_bcdi` /
  `BCDIData` / `list_datasets` for `.npy .npz .h5 .cxi .nxs .mat .tif .bin`
  (container deps are optional and lazily imported).  `kind=` declares whether
  the array is a real-space `object`, a far-field `amplitude`, or `intensity` --
  a complex array is genuinely ambiguous and the loader refuses to guess, since
  guessing applies or skips a Fourier transform and still looks plausible.
  `permute` / `recenter` fix axis order and centring.

  Worked examples: `examples/tutorial_bcdi_forward` (simulate a strained
  nanocrystal end to end, with six self-checks against closed form) and
  `examples/tutorial_bcdi_from_data` (`--from-file`, `--from-md`,
  `--cross-check`, `--grad-demo`).

### Diffraction-as-a-loss-on-dynamics (`dynamics`)  *(novel closure)*
- `thermal_ensemble` -- differentiable thermal cloud from anisotropic spring
  constants (reparameterisation trick), `stiffness_from_msd` (equipartition),
  `recover_stiffness` -- fit `(k_par, k_perp)` straight from diffraction, so a
  transient drop in `k_perp` (lattice softening) is read off the patterns.

### Coherent phonons (`phonon`)
- `strain_wave`, `bragg_timeseries`, `fit_coherent_phonon` -- a damped
  out-of-plane breathing mode modulates a Bragg reflection; recover its
  frequency, damping and amplitude from the time series.

### Amortised ML inference (`ml`)
- `make_dataset` (forward model = data generator), `ParameterMLP`,
  `train_surrogate` -- one network pass maps a pattern to {N3, u_perp}
  (held-out N3 MAE ~0.07 cell).

### Instrument realism + real data (`instrument`, `realdata`)
- `project_to_detector` (Ewald-correct), `solid_angle_polarization`,
  `poisson_nll` / `add_poisson_noise`, `resolution_convolve`;
  `load_profile` and `debye_reference_numpy` (independent NumPy Debye oracle,
  agrees with the torch path to 1e-14).

### Multi-reflection (`rocking.thickness_loss_scan`)
- Joint thickness loss over several rods -- breaks the single-rod thickness
  multimodality (one basin per integer N3) to a unique minimum.

### Depth-resolved strain + unified thermal (`strain_profile`)
- A per-atom out-of-plane displacement `u_z(z)` makes the Bragg peak asymmetric
  (the d-spacing-vs-depth signal); `recover_depth_strain` inverts the asymmetric
  peak back to the depth profile.  `thermal_rod` drives BOTH the shift (thermal
  expansion) and the amplitude (local Debye-Waller) from one `T(z)` field.
  Builders: `linear_strain`, `exponential_strain`, `acoustic_pulse` (Thomsen).

### Diffraction -> transport/coupling coefficients (`thermal_transport`)
- `two_temperature_model` + `fit_electron_phonon_coupling` -- recover the
  **electron-phonon coupling g** from the Bragg-amplitude transient.
- `heat_diffusion_1d` + `fit_thermal_diffusivity` -- recover the **thermal
  diffusivity kappa** from the depth-resolved strain front.

### Differentiable MD -> learn the potential (`md_integrator`)
- `velocity_verlet` (autograd through every step), `bragg_from_trajectory`,
  `coherent_mode_kick`, `recover_potential_from_movie` -- recover the
  **interatomic spring constant** by differentiating an MD trajectory to match a
  Bragg-intensity oscillation.  (Uses a non-uniform standing-wave mode: a
  uniform kick is a rigid translation that |A|^2 cannot see; the intensity rings
  at 2*omega.)

### Frontier tier
- **`multimodal`** -- joint X-ray + optical (transient-absorption) inversion.
  `fit_multimodal` recovers the **deformation potential Xi** and thermal
  coefficient; the optical channel pins the carrier dynamics. (Honest finding:
  when electronic/thermal timescales are well separated, the X-ray strain shape
  alone already localizes Xi -- see `xray_only_degeneracy`.)
- **`latent_dynamics`** -- `discover_eom` recovers the equation of motion
  (e.g. a damped phonon `v_dot = -omega^2 x - gamma v`) from a structural
  trajectory by sparse regression (SINDy/STLSQ), *without assuming the form*;
  the spurious cubic term is thresholded to zero.
- **`ensemble`** -- `recover_thickness_distribution`: deconvolve the thickness
  *distribution* of a polydisperse sample from the smeared fringes.
- **`active_learning`** -- `fisher_information` / `next_best_measurement`: rank
  candidate delays/reflections by how much they constrain a target parameter
  (autonomous-beamline experiment design).

### Showcase demos (write to `dev/paper/figures/`)
- `tutorial_npl_fringes` -- 3/4/5-monolayer Laue oscillations (fringe count = N3-1).
- `tutorial_md_transient_disorder` -- **anisotropic transient disordering from
  the atoms + differentiable recovery of u_perp(t); planted = MD-derived =
  recovered.**
- `tutorial_coherent_rsm` -- coherent reciprocal-space map + phase retrieval.
- `tutorial_stiffness_and_phonon` -- transient lattice softening k_perp(t) +
  coherent-phonon frequency/damping, both recovered from diffraction.
- `tutorial_ml_and_detector` -- amortised-inference parity plot + a coherent
  pattern Ewald-projected onto a detector with Poisson noise.
- `tutorial_depth_strain` -- depth-resolved lattice-displacement reconstruction
  from an asymmetric Bragg peak (the d-spacing-vs-depth signal).
- `tutorial_transport_and_md` -- recover electron-phonon coupling g, thermal
  diffusivity kappa, and interatomic stiffness k, all from diffraction.
- `tutorial_frontier` -- deformation potential (multi-modal), equation-of-motion
  discovery, and ensemble thickness distribution.
- `tutorial_bcdi_forward` -- **Bragg CDI end to end: a strained nanocrystal to
  Poisson counts, with the non-orthogonal q-basis made explicit.** Six
  self-checks against closed form (conjugate-basis identity, analytic Laue
  transform, autocorrelation support, Ewald curvature, oversampling, shear
  correction). Writes its own figure; run with `--dislocation` for an
  anisotropic Stroh field if `midas-dfxm` is installed.
- `tutorial_bcdi_from_data` -- **BCDI from data you already have**: read an
  external array (`--from-file`, any of npy/npz/h5/cxi/mat/tif/bin) or compute
  the signal from atomic coordinates (`--from-md`), by the exact atomic sum or
  the O(N_atoms) binned envelope. `--cross-check` validates one against the
  other with controls; `--grad-demo` shows the detector-level loss
  backpropagating to every atom. Runs self-contained with no arguments.

## Quick start

```python
import torch
from midas_2d import cdse_supercell, coherent_intensity

# Diffraction straight from atoms, differentiable in the coordinates:
coords, elements, cell = cdse_supercell((8, 8, 4))     # few-layer CdSe platelet
coords.requires_grad_(True)
q = (2 * torch.pi / 6.077) * torch.tensor([[1., 1., 1.0]])
I = coherent_intensity(coords, elements, q)            # I.backward() -> per-atom grads
```

```bash
python -m midas_2d.examples.tutorial_md_transient_disorder   # the headline figure
pytest                                                       # 38 tests (set KMP_DUPLICATE_LIB_OK=TRUE on macOS)
```

## Not yet implemented

- Phase 5: ML inversion / amortized-inference surrogates (uses the
  differentiable forwards as data generators + consistency layer).
- Multi-reflection / multi-Bragg joint BCDI; absolute-scale spectral weighting.
