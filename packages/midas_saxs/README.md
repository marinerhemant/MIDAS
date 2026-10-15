# midas-saxs

Differentiable small-angle X-ray scattering for MIDAS. Three source terms, one
detector:

- **Density contrast** — voids, gas bubbles, precipitates. Sphere / ellipsoid /
  cylinder form factors, lognormal polydispersity, Percus-Yevick `S(Q)`, Guinier
  and Porod analysis. These came from `midas_pdf.saxs`, which now re-exports them
  from here, so MIDAS has one definition of "SAXS form factor".
- **Strain contrast** — dislocation loops, driven by a `midas_ddd` network read
  from ExaDiS. Optional extra: `pip install 'midas-saxs[dislocations]'`.
- **Atomistic** (provisional) — a periodic MD box read from a LAMMPS dump; scattering
  from the atomic positions themselves. See *Scattering from MD atomic positions* below
  for what has and has not been checked.

```python
from midas_saxs import SAXSGeometry, SpherePopulation, simulate_frame

geom = SAXSGeometry(lsd_um=2.0e6, bcy_px=512, bcz_px=512, px_um=75.0,
                    wavelength_A=0.7293, n_pix_y=1024, n_pix_z=1024,
                    beamstop_radius_px=30)
voids = SpherePopulation(radius_A=50.0, number_density_per_A3=1e-8,
                         delta_rho_e_per_A3=-2.45, label="voids")
frame = simulate_frame(geom, particles=[voids], sample_volume_A3=1e15)
```

## Read this before promising anyone a loop measurement

Three numbers, measured with a development demo script that is not part of the
package.

**1. Loops are faint.** A dislocation loop of radius R has relaxation volume
`pi R^2 b`; a void of the same radius displaces `4 pi R^3 / 3`. For R = 5 nm in
Cu that is 26x in volume, so **680x in intensity from volume alone**. The loop's
relaxation volume is that of a sphere of radius 1.69 nm, but it does not scatter
like that sphere: in its own plane its intensity carries a further
`(1 - kappa)^2` (0.36 for `lambda = 100, mu = 75`, making **1890x**), and along
its normal it is zero (point 2).

**2. Aligned loops differ from voids in shape.** A loop's small-angle amplitude
is the distortion term *plus* the Laue term, the scattering from the loop's own
extra or missing atoms (Ehrhart, Trinkaus & Larson, Phys. Rev. B 25 (1982) 834,
Eq. 8a/8b). As q -> 0 it is `dV (1 - kappa) sin^2(theta)`, with theta measured
from the loop normal and `kappa = lambda/(lambda + 2 mu)`: **zero along the
normal**, largest in the loop plane. On the demo (24 loops, normals along
detector z) the azimuthal profile follows `cos^4(azimuth)`, and its minimum is
set by the bin width (I_min/I_max = 3.8e-5 with 48 bins, 2.1e-14 with 144). A
void is isotropic. **Radially averaging a frame destroys the null**, which is
why this package renders 2-D frames and offers `azimuthal_profile` alongside
`radial_average`. Two limits: the minimum reaches zero only when the loop
normal lies in the detector plane and flattens as the normal tilts toward the
beam, and loops with isotropically distributed normals scatter isotropically,
like voids (see Identifiability below).

An earlier version of this README quoted `kappa dV` in the loop plane and `dV`
along the normal, a `1/kappa^2 = 6.25` contrast, and 6.30x from the demo. That
is the distortion term alone, and at small angle it is backwards.

**2b. The null is a property of the linear-elastic model, not (yet) of relaxed atoms.** A first atomistic check
(2026-10-07, details in *Scattering from MD atomic positions* below) relaxed a vacancy Frank loop in Cu with an EAM
potential and compared its frame with the continuum amplitude. Away from the normal the two agree to tens of percent; along
the normal the continuum is exactly zero while the relaxed atoms keep a small residual that is 40-200 times below the
un-relaxed disc but dominates the ring at q R above about 3.5. The registered test for the null returned REFUTED. One
potential, one radius, 0 K, vacancy loops only; treat the "exact null" as a statement about the continuum model.

**3. Real voids bury the signature.** Add voids of the same radius at the same
number density — the physically realistic irradiated case — and the azimuthal
contrast collapses to **1.001x**, with the voids outshining the loops by
**5212x** integrated over the frame.

So: loops are detectable at small angle in a clean matrix, and essentially
undetectable alongside a void or bubble population of comparable size. If the
question is defect-type discrimination in an irradiated single crystal,
near-Bragg diffuse (Huang) scattering is usually the better measurement. For a
single prismatic loop (isotropic matrix, Cu 111) the median gain over q
directions is 1.3e6, 1.3e4 and 1.3e2 at `|q|` = 1e-3, 1e-2 and 1e-1 1/A, about
7x below the naive `(G/q)^2`. The spread covers more than three decades, and
small-angle is the stronger signal in 9.8 % of directions at 1e-1 1/A. Method
and the full table are in the `midas_defect.huang` module docstring.

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

## Scattering from MD atomic positions (provisional)

A worked example that runs without LAMMPS (a synthetic Cu box with a void, written to a dump and read back, checked against an analytic curve and against the exact atom sum) is `notebooks/02_saxs_image_from_md_atoms.ipynb`.

```python
from midas_saxs import SAXSGeometry, simulate_frame, read_lammps_trajectory

frames = read_lammps_trajectory("irradiated.dump", type_map={1: "Cu"}, frames=slice(0, 40, 5))
geom = SAXSGeometry(lsd_um=5.0e5, bcy_px=256, bcz_px=256, px_um=75.0, wavelength_A=0.7293,
                    n_pix_y=512, n_pix_z=512, beamstop_radius_px=10)
fr = simulate_frame(geom, atoms=frames, sample_volume_A3=1e12)
fr.components["atoms"], fr.info["atoms"], fr.warnings
```

**What it computes.** For one periodic box the amplitude `A(h) = sum_i f_s(i)(|G|) exp(2 pi i h . s_i)` at the
box's own reciprocal-lattice points `G`, with Cromer-Mann form factors, summed per species. The box defines
intensity only there (Paddison, arXiv:1809.07088; the same statement the dislocation code makes). Pixels
between lattice points get a Gaussian window average at a stated resolution, in the un-normalised (Parseval)
form, so the same structure stored in a supercell gives the same intensity per volume. The result is
electrons^2 per A^3 and is multiplied by `sample_volume_A3`, exactly like the particle terms.

**Inputs that matter.**
- LAMMPS types are integers. You must pass `type_map={1: "Cu", 2: "He"}` (or have an `element` column);
  there is no default element. Units must be metal or real (Angstrom). The reader accepts orthogonal and
  triclinic boxes, plain or `.gz`, and refuses a trajectory whose cell or species order changes.
- **Orientation.** Transmission SAXS has no omega, so the sample orientation is yours to set:
  `atoms_options={"lab_from_sample": R}` with `r_lab = R r_sample`. The default puts the cell's x axis along the beam.
- **Frames** are averaged as intensities (incoherently); the spread across them is the thermal disorder.
- `atoms_options` also takes `resolution_fwhm_inv_A` (default two reciprocal-lattice spacings) and
  `method` (`"auto"`, `"direct"`, `"fft"`).

**The q floor.** Below about three reciprocal-lattice spacings (`3 * 2 pi / L`) the excluded `G = 0` term and
the box's periodicity dominate; `Frame.warnings` reports the fraction of pixels below it. A 20 nm box has a floor
near 0.09 1/A, so MD boxes cover the upper part of a SAXS window, not its lowest q.

**Two ways to get `A(h)`.** `direct` is the exact sum (cost `N_atoms x N_G`). `fft` deposits each species on a
grid with cardinal B-splines and takes a 3-D FFT; it is valid only for `|h_i| <= n_i / 8` and refuses points
outside that. Its aliased Bragg leakage grows as `N^2`: `info["atoms"]["fft_leakage_bound_f2"]` is the
worst case in units of one atom's scattering, and a warning appears when it exceeds 1.

### What has been checked, and how far (2026-10-02)

`/verify` returned **PROVISIONAL** (claim `8252e0c9ef19`): one lens refuted the claim as first worded, three
were uncertain. What stands:

- **The code computes the sum it says it computes.** It equals an independent exact sum over the removed atoms
  to about 1e-9 (a refuter's own implementation reproduced every number), is invariant to translation, wrapping, atom order, supercell and a
  triclinic re-description of the same crystal (1e-14), returns nothing for a perfect crystal, and gives
  exactly `f(q)^2` for one removed atom. FFT and direct agree to a median of 1e-5 inside the FFT range.
- **Absolute scale** for uncorrelated displacements is consistent with the exact result at about +-1 % (2 sigma,
  32 and 64 seeds).
- **Against continuum particle models.** A spherical void of about 1200-2000 atoms differs from the sphere form
  factor of equal volume by about 1-4 % at q R_eff of 1.5-3 (median of the worst point about 1.5 %; the error
  grows with q). A void of 180-354 atoms differs by 17-47 %: a few hundred lattice sites are not a sphere, and
  the code agrees with the exact sum there, so this is the physics of small defects, not an error. **For an
  ellipsoidal void the agreement depends on where the void sits on the lattice and can exceed 10 %** (aspect 4,
  2 nm: 6-39 % over seven centres). Treat continuum particle models as a few-percent to tens-of-percent
  approximation for MD-scale defects, and the atomistic frame as the more reliable description.

- **A relaxed vacancy Frank loop against the continuum loop amplitude** (gate G9; 4 placements of one construction, Cu
  EAM potential of Mishin, 0 K, radius 2 nm, 1.8 million atoms each, elastic constants measured from the same potential).
  The registered test returned **REFUTED**: the continuum's exact zero along the loop normal is not reproduced. The
  median intensity ratio (atoms / isotropic continuum) over the pixels is 1.13 (four replicates 1.125-1.143), which is
  inside the registered band, but the spread is large, and for q R above about 3.5 the loop-normal direction is the
  *brightest* direction of the ring. Along the normal the relaxed atoms scatter 40-200 times less than the un-relaxed
  vacancy disc alone, so the relaxation does cancel most of the Laue term, but not all of it. The collapse of the disc
  is also a few percent short of the ideal half Burgers vector (0.95-0.99 A against 1.04 A); whether that is the cause was not tested.
  *Post hoc and provisional:* using the measured cubic stiffness rotated into the box frame instead of an isotropic
  average brings the in-plane ratio to 1.02-1.08 for q R below about 2 (isotropic: 1.25-1.33), and that continuum also has
  an exact null, so anisotropic elasticity does not explain the missing null.

### What has NOT been checked

- **Anything about loops beyond that one construction:** other radii, interstitial loops, other Burgers vectors or
  loop-normal orientations relative to the beam, other potentials, finite temperature, and any comparison with a measured
  or ExaDiS-generated frame. Do not rely on the atomistic frame for loop signatures in those cases.
- Any comparison with a measured SAXS frame.
- CUDA parity and timings on large boxes (CPU, one laptop only).
- **Defect-defect correlations.** The frame treats boxes as independent copies. A dense cascade or a dislocation
  forest has an S(q) between its defects that this does not contain.
- Anything beyond kinematic, single-scattering, neutral-atom scattering (no anomalous terms, absorption, Compton or resolution).

Timing on one 8-thread laptop: a 107,631-atom box on a 1024 x 1024 panel takes about 8 s; the direct sum runs
at about 2e8 atom-lattice-point pairs per second. Boxes of 4 million atoms ran with the FFT path.

## Scope

**Three dislocation populations, treated differently.** Closed loops, the
irradiation population: exact at every q, distortion plus Laue term. Finite lines
(open lines, junction clusters that do not wind): the line-integral amplitude at
every q; an edge or mixed line scatters into a sheet perpendicular to itself, an
isotropic screw not at all, and a line ending inside the medium carries its
ends' unphysical sources off that sheet (the frame warns). Lines that close only
through a periodic DDD cell's boundary: an amplitude only on the cell's
reciprocal lattice, shown between lattice points at a stated resolution (default
two lattice spacings, `periodic_resolution_fwhm_inv_A`) as the `periodic_lines`
component, with a low-q floor of about three lattice spacings below which pixels
are not representative; a bigger simulation cell lowers it. Pure screws in cubic
or isotropic elasticity have no dilatation, and the frame says so when a
component is only float64 roundoff. The line terms are preregistered and under
adversarial verification (2026-09-11), not yet established. An earlier version
of this README said lines contribute nothing at small angle; that holds only as
`q -> 0`. `include_lines=False` reproduces the loops-only frames of 0.1.x and
reports the line length it left out.

## Units

| Quantity | Unit |
|---|---|
| distances, geometry | micrometers |
| wavelength, particle radii, electron density | angstroms |
| `q` from `midas_saxs.geometry` | **inverse angstroms** |
| `q` into the `midas_ddd` kernel | **inverse micrometers** |
| atomic positions and cell from an MD dump | **angstroms** (LAMMPS metal or real units) |

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
