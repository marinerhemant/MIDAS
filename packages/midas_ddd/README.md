# midas-ddd

Dislocation networks for MIDAS: ingest a discrete-dislocation-dynamics network
(ExaDiS / ParaDiS) and turn it into the displacement fields that three MIDAS
forward models consume.

```
q . u~(q)          ->  small-angle scattering       (midas_saxs)
(G+q) . u~(q)      ->  near-Bragg diffuse / Huang   (midas_defect.huang)
real-space beta(r) ->  DFXM contrast                (midas_dfxm)
```

Everything is torch-differentiable.

**Elastic vs total — the distinction that bites first.** `midas_ddd.fourier`
works with the TOTAL displacement; `midas_ddd.realspace` (Mura line integral over
finite segments) returns the ELASTIC distortion. Both are physical and they are
not the same thing: DFXM images the elastic part, because the lattice is
continuous across the cut surface for a perfect dislocation, while the
scattering kernels need the total. The two are tied together quantitatively —
their ball-averaged trace integrals are `(2/3)(1-2nu)/(1-nu)` and
`(1+nu)/(3(1-nu))` in units of `dV`, which sum to exactly 1 because the plastic
eigendistortion on the cut contributes `-dV`. That relation is the
cross-modality gate in `tests/test_realspace.py`. When the two kernels first
appeared to disagree by a clean functional factor, this was why.

**Scope differs between the two.** The Fourier kernel handles CLOSED LOOPS only
(a cut surface needs a closed circuit), so open deformation lines contribute
nothing at small angle and `u_tilde` reports the line length it ignored. The
real-space kernel has no such limit — an open network images perfectly well in
DFXM.

## Quick start

```python
from midas_ddd import read_paradis, validate_network, prismatic_loop
from midas_ddd import q_dot_u_tilde, isotropic_stiffness
import torch

# A network from ExaDiS, via the file bridge that always works
net = read_paradis("net.data", b_magnitude_A=2.556)
print(validate_network(net, q_max_inv_A=0.1))

# Or build one here, with no ExaDiS installed
loop = prismatic_loop(radius_um=0.005, burgers=(0, 0, 1), n_segments=64)

q = torch.tensor([[0.0, 0.0, 1e-2]], dtype=torch.float64)   # inverse micrometers
amp = q_dot_u_tilde(loop, q, isotropic_stiffness(100.0, 75.0))
```

## Three things worth knowing before you use it

**1. Burgers magnitudes are physical; they are not normalised.** A perfect
dislocation has `|b| = 1` in file units, but a junction from a dislocation
reaction has `|b| = sqrt(2)`, and that magnitude is what balances Burgers
conservation at the junction node. The 5055-node FCC-Cu network in this repo has
54 such segments; normalising them made 60 nodes fail conservation with a
residual of exactly `sqrt(2) - 1`.

**2. The cut surface is physical, not a gauge.** Stokes gives the part of the
surface form factor perpendicular to `q` from a cheap line integral, which looks
like a way to avoid choosing a cut surface. It is not: for a prismatic loop
probed along its own normal — where it scatters most strongly — the transverse
projection returns *exactly zero*. The kernel integrates an explicit fan
triangulation from each loop's centroid. Gated by
`test_transverse_gauge_would_destroy_the_signal`.

**3. Only closed loops contribute.** A finite cut surface exists only for a
closed circuit. Open lines — the deformation population — enclose no area, carry
no relaxation volume, and contribute nothing as `q -> 0`. `u_tilde` reports how
much line length it ignored rather than quietly returning a number that looks
complete. Their small-angle signature is a weak transverse streak this kernel
does not model.

## The gate

For isotropic elasticity and a prismatic loop (`b || A || n`) the small-q limit
has an exact closed form, verified to 2e-15 against the tensor expression over
1200 random directions and four moduli:

```
q . u~(q -> 0) = i dV [ kappa + (1 - kappa) (n.qhat)^2 ]
    dV = b . A          (pi R^2 |b| for a circular loop)
    kappa = lambda / (lambda + 2 mu)
```

Read that twice: **the limit is anisotropic**. A loop does not scatter like a
compact particle of volume `dV`. It scatters `kappa dV` in its own plane and
`dV` along its normal — a contrast ratio of `(lambda + 2 mu)/lambda`, which is
2.5 for `lambda = 100, mu = 75`. A void of the same volume is isotropic.

That anisotropy is the loop-versus-void discriminator, it is present at `q -> 0`
rather than only at finite q, and it is why a 2-D detector image is worth
simulating instead of a radially averaged `I(q)`.

## ExaDiS

ExaDiS needs a Kokkos/CMake build and can never be a pip dependency. Two routes:

- **File bridge (always works).** Run ExaDiS anywhere, call its
  `write_data(N, "net.data")`, read it with `read_paradis`.
- **In-process (optional).** `midas_ddd.exadis` lazily imports `pyexadis` and
  wraps `generate_prismatic_config` (irradiation loops, `radius` accepts
  `[min, max]`) and `generate_line_config` (deformation lines). Build with
  `cmake .. -DEXADIS_PYTHON_BINDING=On`; on the beamline, chiltepin is the only
  host with internet.

`midas_ddd.generate` provides small exact equivalents (`prismatic_loop`,
`straight_line`, `combine`) so the whole test suite runs without ExaDiS.

The in-process bridge is exercised by `tests/test_exadis.py`, which skips
without a build. Its load-bearing test takes BOTH routes into this package from
one ExaDiS config — objects converted in memory, and ExaDiS's own `write_data`
read back by `read_paradis` — and requires them to agree.

## Resolution

A nodal DDD code discretises a 2 nm loop into a handful of segments, and SAXS at
`q ~ 1/R` probes exactly that scale. `resolution_report` bounds the usable q at
`2 pi / L_median` and says so. The repo's FCC-Cu network has a median segment of
651 nm, giving `q_max ~ 1e-3 1/A` — a SAXS run at 0.1 1/A on it would be
reporting the polyline, not the dislocations.

## Relationship to midas_defect

The anisotropic-elasticity (Stroh) primitives here — `cubic_stiffness`,
`hexagonal_stiffness`, the sextic solver, the slip-system tables — used to live
in `midas_defect.contrast_factor`, which already carried a "do NOT re-port"
contract because `midas_dfxm` imported them across a package boundary. They
moved down when a third and fourth consumer appeared: a SAXS package should not
depend on an FF-HEDM metrology package to build a stiffness matrix.
`midas_defect.contrast_factor` re-exports every name, so no historical import
path broke, and `test_elasticity.py::test_reexport_*` fails loudly if anyone
re-ports a copy.

## Numerical traps, all of them found the hard way

**Burgers magnitudes are not normalised** (see above): 54 of the repo network's
5930 segments are `sqrt(2)` junctions, and normalising them broke conservation
at 60 nodes by exactly `sqrt(2) - 1`.

**The cut surface is subdivided RADIALLY**, not just around the circumference. A
plain fan from the centroid has an edge of length R running out to every vertex,
so the quadrature parameter `|q|h` is set by the loop radius however finely the
polygon is discretised. At `qR = 10` — mid-detector for real SAXS — a plain fan
is ~20 % wrong. `u_tilde` picks `n_rings` from `q_max` automatically.

**Polygon shape is not quadrature error.** With the quadrature converged to
1e-8, a 64-gon and a 256-gon still differ by ~1 % at `qR = 10`. That is the
polygon, a modelling choice; only `n_segments` changes it.

**Fixed-order quadrature fails the Mura line integral.** The integrand falls as
`1/R^2`, so eight Gauss nodes spread over a long segment miss the peak entirely:
a 400 um segment probed at 0.5 um came out ~1000x low and *rising* with r.
`realspace` subdivides into panels sized by the closest approach.
