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
