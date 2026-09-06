# midas-saxs

Differentiable small-angle X-ray scattering for MIDAS. Two source terms, one
detector:

- **Density contrast** — voids, gas bubbles, precipitates. Sphere / ellipsoid /
  cylinder form factors, lognormal polydispersity, Percus-Yevick `S(Q)`, Guinier
  and Porod analysis. These came from `midas_pdf.saxs`, which now re-exports them
  from here, so MIDAS has one definition of "SAXS form factor".
- **Strain contrast** — dislocation loops, driven by a `midas_ddd` network read
  from ExaDiS. Optional extra: `pip install 'midas-saxs[dislocations]'`.

```python
from midas_saxs import SAXSGeometry, SpherePopulation, simulate_frame

geom = SAXSGeometry(lsd_um=2.0e6, bcy_px=512, bcz_px=512, px_um=75.0,
                    wavelength_A=0.7293, n_pix_y=1024, n_pix_z=1024,
                    beamstop_radius_px=30)
voids = SpherePopulation(radius_A=50.0, number_density_per_A3=1e-8,
                         delta_rho_e_per_A3=-2.45, label="voids")
frame = simulate_frame(geom, particles=[voids], sample_volume_A3=1e15)
```

`dev/paper/loop_vs_void_demo.py` renders the five-panel figure that motivates the
whole package.

## Read this before promising anyone a loop measurement

Three numbers, all reproducible from the demo script.

**1. Loops are faint.** A dislocation loop of radius R displaces only its
relaxation volume `pi R^2 b`; a void of the same radius displaces
`4 pi R^3 / 3`. For R = 5 nm in Cu that is 26x in volume and **680x in forward
intensity**. The loop scatters like a *sphere of radius 1.7 nm*.

**2. What separates a loop from a void is shape, not size.** The loop's
small-angle amplitude is direction-dependent — `kappa dV` in its own plane,
`dV` along its normal, with `kappa = lambda/(lambda + 2 mu)`. Intensity is
therefore modulated by `1 - kappa^2` (0.84 for `lambda = 100, mu = 75`). A void
is isotropic. **Radially averaging a frame destroys exactly this**, which is why
this package renders 2-D frames and offers `azimuthal_profile` alongside
`radial_average`.

**3. Real voids bury the signature.** Measured on the demo: 24 aligned loops
alone give an azimuthal contrast of **6.30x** (the closed form predicts
`1/kappa^2 = 6.25`). Add equal-radius voids at the same number density — the
physically realistic irradiated case — and the contrast collapses to **1.002x**,
with the voids outshining the loops by **1089x** integrated over the frame.

So: loops are detectable at small angle in a clean matrix, and essentially
undetectable alongside a void or bubble population of comparable size. If the
question is defect-type discrimination in an irradiated material, near-Bragg
diffuse (Huang) scattering is stronger than the small-angle signal by
`(G/q)^2` ~ 1e4 to 1e7 and is the better measurement.

## Identifiability — what a frame determines, and what it only appears to

**Loop number density is not recoverable from a single SAXS frame.** A study in
this package claimed 2.6 % on loop number density and 0.6 % on loop radius, at
1e6 photons/q-point with voids at volume fraction 5.2e-5, from a loop-block
condition number of 639. It was **REFUTED on 2026-09-04** by adversarial review.
Every number in it reproduces, several bit-identically — the arithmetic was
never the problem. The numbers describe a different parameter vector, a
different void loading and a different sample mounting than the claim stated.

Six ways it fails, each measured by a refuter rather than read from the study's
own output:

| | effect |
|---|---|
| Signed `dV` restored to the parameter vector (it had been dropped *because* it made the Fisher matrix rank-deficient — that rank deficiency was the finding) | cond **2.77e16**; the data fix only the combination `n*dV^2`; SE[log n_loop] **unbounded** |
| Void loading raised to this package's own "realistic irradiation case" (equal radius, equal number density) — the study ran voids 100x **rarer** than loops | SE[n_loop] **252 %** |
| A true Watson habit distribution instead of `watson_normals`, which is a delta cone at a single polar angle and cannot reach the isotropic control | **4.7 %** at the planted S = 0.75, **699 %** at uniform |
| Sample rotated: the q-grid had put the habit axis in the detector plane, the single most favourable choice | in-plane 2.57 %, 10 deg off the beam 52 %, **along the beam 1.4e4-2.7e4 %**; about **7 %** of habit-axis orientations fail outright |
| Absolute intensity scale freed (it was hard-fixed, i.e. flux x volume x transmission x efficiency known exactly) | an exact null vector appears; only the **ratio** `n_loop/n_void` is identifiable. With a 1 % calibration prior 2.77 %, with 10 % 10.3 % |
| Mild model error: log-normal loop sizes at sigma_ln = 0.2, or a true Watson habit at the same S | bias **-28 %** and **-24 %** on `n_loop`, at 9-11 sigma, with chi2/dof of 0.61 and 0.02 — i.e. entirely inside "the fit looks fine" |

At the claim's own operating point voids already supply **97.9 %** of the
photons and the whole result rides on a **3.54 %** azimuthal modulation. Over
99 % of the claimed information is azimuthal anisotropy, not scattering-curve
shape.

**A condition number cannot be a confirm criterion here.** A Fisher matrix
scales linearly with photon budget, so its condition number is *exactly*
scale-invariant: measured at 639.5832 to seven digits across 1e0, 1e3, 1e6, 1e9
and 1e12 photons/q-point while SE[log n_loop] ran 2580 % to 0.0026 %. Across
four decades of void loading it moved 639.6 to 640.2 while SE went 2.58 % to
101 %, straight through the refute threshold with the criterion passing
untouched.

**What survived.** Loop **radius** at 0.6 % is robust — it held under every
lens, including the ones that destroyed the number density, and is unaffected
by freeing the absolute scale. State it with its condition: `q_max * R >~ 3`.

**What this means for the code.** Nothing here is wrong. Three independent
lenses confirmed the Fourier kernel, the Laue correction, the Fisher assembly
and the Schur profiling. The forward model is sound; the claim was wrong about
what it measured. If you build a Fisher or CRB analysis on this package,
include the degenerate directions (signed `dV`, a loop-vs-plate shape
parameter), use a habit distribution that can actually reach isotropy, put the
void population at its realistic loading, and report the sample mounting --
because the answer depends on all four.

## Scope

**Closed dislocation loops only.** A cut surface exists only for a closed
circuit. Open lines — the *deformation* population — enclose no area, carry no
relaxation volume, and contribute nothing as `q -> 0`; their small-angle
signature is a weak transverse streak that is not modelled. `simulate_frame`
reports how much line length it ignored rather than returning a number that
looks complete.

## Units

| Quantity | Unit |
|---|---|
| distances, geometry | micrometers |
| wavelength, particle radii, electron density | angstroms |
| `q` from `midas_saxs.geometry` | **inverse angstroms** |
| `q` into the `midas_ddd` kernel | **inverse micrometers** |

The 1e4 between the last two is easy to lose, so `inv_A_to_inv_um` /
`inv_um_to_inv_A` exist and `strain_source` uses them rather than a literal.

## Detector geometry

`pixel_to_q` goes through `midas_transforms.apply_tilt_distortion` — the same
tilt (`R_z R_y R_x`) and 15-coefficient distortion model the HEDM side uses.
That is deliberate: a SAXS geometry and an FF geometry calibrated from the same
detector must agree about where q sits. Transmission SAXS has no omega, so lab q
is sample q.

## Migration note

`form_factors`, `model`, `wide_band` and `core_shell` moved here from
`midas_pdf.saxs`. Both the package-level path (`from midas_pdf.saxs import
SAXSModel`) and the deep path (`from midas_pdf.saxs.form_factors import ...`)
still resolve, to the same objects — asserted by
`test_midas_pdf_reexports_the_same_objects`. `midas_pdf.saxs` keeps the genuinely
PDF-coupled part: joint SAXS + PDF refinement and its Bayesian variants.

One convention worth knowing: `sphere_form_factor_squared` returns `V^2 |F|^2`,
**not** the normalised `|F|^2`. `SpherePopulation.intensity` multiplies it by
`n * delta_rho^2` and nothing else, which is only correct because the volume is
already in there.
