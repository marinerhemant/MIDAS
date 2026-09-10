# solve-cell — determining a lattice when the cell is in question

The spine. Keep this loaded; open the phase files as you reach them.

**Scope.** Spot positions (or raw rotation frames) from a single crystal or a few domains,
where the CELL ITSELF is unknown or disputed. Out: many grains at a known cell (`ff-hedm`),
the diffuse scattering between the peaks (`defect`), continuous powder rings (`xrd-ct`),
geometry from a calibrant (`calibrate-integrate`). **Atomic positions are out of scope.**
This determines lattices, not structures — say so plainly if asked for Rietveld.

## Order of operations, and why it is this order

| # | phase | file | why it cannot move |
|---|---|---|---|
| 1 | geometry | `phase-1-geometry.md` | a cell inherits every geometry error; axial ratios are immune to scale but not to centre/tilt |
| 2 | find the lattice | `phase-2-lattice.md` | ab initio, or from reciprocal-lattice rows, or seeded — the seed decides what you can find |
| 3 | refine it | `phase-3-refine.md` | one domain, then several jointly; errors by bootstrap, never analytic |
| 4 | symmetry + distortion MODE | `phase-3-refine.md` | fitting a diagonal cell to sheared data manufactures a splitting |
| 5 | which phase / series member | `phase-4-phase-id.md` | equal provenance, line counts, and a volume-correct null |
| 6 | pressure, if asked | `phase-4-phase-id.md` | inverting through a fitted ladder; the fit can BE the answer |
| 7 | completeness + report | `ENVELOPE.md` | INDEXED / MISSED / MASKED / ABSENT, and the limit on every number |

## Install gate

```
midas-hkls>=0.11.0      lattice math, ab initio, symmetry, distortion mode, phase ID
midas-defect>=0.1.7     rows, seed_index, geometry, completeness   (see the note below)
midas-calibrate-v2      Friedel beam centre, tx from omega splitting
midas-transforms        det <-> q
```

**Where the code lives does not match what it does, and that is deliberate for now.**
`midas_defect.rows`, `.seed_index`, `.geometry` and `.completeness` are lattice/indexing
machinery that happens to live in the defect package for historical reasons. Import them from
there; do not go looking for them in `midas_hkls`. (Measured: 61 % of one project's MIDAS
import sites were cell/symmetry work, and three of its four heaviest "defect" modules were
these.)

Package-level findings — missing symbols, misleading defaults, where the code lives and why:
**`PACKAGE_NOTES.md`**.

## Two package traps that will bite you first

**1. `midas_defect.rows` DEFAULTS TO ONE MATERIAL'S CELL.** `find_lattice_rows`,
`index_from_row` and `refine_to_convergence` carry `a=3.6116, c=19.2516,
space_group_number=139` — La3Ni2O7 — as *default arguments*, and `refine_to_convergence` also
defaults `sigma_rtn=(0.0071, 0.0145, 0.0094)`, a residual budget measured on that sample.
**Pass your own explicitly, every call.** A run that omits them is silently seeded with a
nickelate.

**2. `two_pi` defaults OPPOSITE ways in the two halves of this chain.**
`midas_defect.rows.*` and `find_lattice_rows` default `two_pi=True` (q = 2π/d);
`midas_hkls.ab_initio.index_ab_initio` defaults `two_pi=False` (q = 1/d). Feed a 2π cloud to
the ab initio indexer without `two_pi=True` and it fails to index rather than erroring.
**State your q convention once, at the top of the script, and pass it everywhere.**

## The calling contracts — copy these, do not reconstruct them

```python
# --- find the lattice with NOTHING supplied ------------------------------------
from midas_hkls.ab_initio import index_ab_initio
res = index_ab_initio(g,                    # (N,3) g-vectors
                      two_pi=True,          # MUST match your cloud's convention
                      sigma_g=2.76e-3,      # measured, not guessed -- sets significance
                      min_reflections=20)

# --- or from reciprocal-lattice rows (works when ab initio will not) -----------
from midas_defect.rows import find_lattice_rows, index_from_row, refine_to_convergence
rows = find_lattice_rows(q, I, a=A, c=C, two_pi=True,      # PASS THESE
                         space_group_number=SG, identify=False)   # identify=False = cell-free

# --- centred lattices: ab initio returns the PRIMITIVE cell --------------------
from midas_hkls.conventional import to_conventional
conv = to_conventional(res.cell)          # compare against THIS, never the primitive

# --- symmetry, from the metric, per entry --------------------------------------
from midas_hkls.niggli import niggli_reduce
from midas_hkls.lattice_symmetry import holohedry
holo = holohedry(niggli_reduce(cell).cell, rel_tol=1e-3)

# --- joint refinement across domains, with bootstrap errors --------------------
from midas_hkls import DomainData, refine_cell_joint, split_with_error
fit = refine_cell_joint([DomainData(hkl=h, g=g, label=str(i)) for i, (h, g) in enumerate(D)],
                        system="orthorhombic",       # CONSTRAIN IT
                        cell0=cell0, two_pi=True, n_bootstrap=500)
delta, lo, hi = split_with_error(fit)     # bootstrap, never analytic

# --- can these reflections separate a from b AT ALL? ---------------------------
from midas_hkls import ab_separable, shear_separable, partner_multiplicity, index_asymmetry
```

## Hard rules

* **Constrain the cell to the symmetry you believe** before quoting a splitting. Six extra
  free parameters buy ~3 % in rms and absorb noise as a distortion.
* **Errors by bootstrap, never analytic.**
* **Reference every plausibility gate to a fixed nominal cell, never to a neighbour's** — a
  seed-referenced gate is circular and manufactures a correlation from data with none.
* **Two files giving one quantity two values must be reconciled, not chosen between.**
* **Report `partner_multiplicity` and `index_asymmetry` with any a/b number**, and the
  distortion MODE before the magnitude.
* **Name the q convention, the cell setting (primitive vs conventional) and the units in the
  same sentence as the number.**

## Halt conditions

Stop and ask when: the lattice will not converge from any seed; a fitted parameter rails at a
bound (a railed fit is not a fit — print bounds); the answer changes sign or ordering under a
re-seed; two records disagree on one quantity; or `ENVELOPE.md` says the ask is unobtainable.
