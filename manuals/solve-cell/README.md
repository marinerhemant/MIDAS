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
| 0 | raw frames → spots → g | `phase-1-geometry.md` §0 + the `defect` ingest contract | a Friedel centre needs spots and spots need a provisional geometry: seed, ingest, measure, re-ingest |
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
midas-defect>=0.1.7     rows, seed_index, indexing, geometry, completeness, selfcal   (see the note below)
midas-calibrate-v2      Friedel beam centre, tx from omega splitting, poni_check, ring_anisotropy
midas-transforms        det <-> q
midas-integrate-v2      poni_file_to_row_col -- a PONI is a SEED (trap 3)
# selfcal, domains, poni_check, ring_anisotropy and the P12 fix: added 2026-09-10, not yet released
```

**Where the code lives does not match what it does, and that is deliberate for now.**
`midas_defect.rows`, `.seed_index`, `.geometry` and `.completeness` are lattice/indexing
machinery that happens to live in the defect package for historical reasons. Import them from
there; do not go looking for them in `midas_hkls`. (Measured: 61 % of one project's MIDAS
import sites were cell/symmetry work, and three of its four heaviest "defect" modules were
these.)

Package-level findings — missing symbols, misleading defaults, where the code lives and why:
**`PACKAGE_NOTES.md`**.

## Three package traps that will bite you first

**1. `midas_defect.rows` DEFAULTS TO ONE MATERIAL'S CELL — in six functions.**
`find_lattice_rows`, `index_from_row`, `index_from_pairs`, `match_mask`, `cell_from_row` and
`refine_to_convergence` all carry `a=3.6116, c=19.2516, space_group_number=139` — La3Ni2O7 — as
*default arguments*. `refine_to_convergence` also defaults `sigma_rtn=(0.0071, 0.0145, 0.0094)`, a
residual budget measured on that sample; `index_from_pairs` defaults `q_max_anchor=3.2`, which
silently blocks anchors above it (diamond (220) at 4.97, (311) at 5.84), while
`space_group_number=139` forbids all-odd reflections. **Pass your own explicitly, every call.** A run
that omits them is silently seeded with a nickelate — and the project that wrote these functions
called two of them without a cell in its own production driver. Until 2026-09-10 the package's own
`index_from_cloud` did the same inside its cell convergence (`PACKAGE_NOTES.md` §10).

**2. `two_pi` defaults OPPOSITE ways in the two halves of this chain.**
`midas_defect.rows.*` and `find_lattice_rows` default `two_pi=True` (q = 2π/d);
`midas_hkls.ab_initio.index_ab_initio` defaults `two_pi=False` (q = 1/d). Feed a 2π cloud to
the ab initio indexer without `two_pi=True` and it fails to index rather than erroring.
**State your q convention once, at the top of the script, and pass it everywhere.**

**3. `BC_y` means the ROW in one MIDAS package and the COLUMN in another.** In `midas_integrate_v2`
(`poni_to_bc`, `poni_file_to_bc`) `BC_y` is the row axis; in `midas_defect.Geometry`, `bcy_px` is the
column. Each is self-consistent and tested, and crossing the boundary swaps the axes silently — this
doc set's own test run did exactly that and seeded a centre 123 px / 59 px off. **At that boundary use
`poni_file_to_row_col`**, which exists for it, and pass `Geometry(bcz_px=row, bcy_px=col)`. It returns
the calibration's row order, so a reader that disagrees still needs the flip (`phase-1-geometry.md`).

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
# conv decides the crystal system ONCE, at one tolerance: on 2604 the default gave triclinic and
# rel_tol=0.0768 gave orthorhombic-P -- never the body-centred cell. Take c from a row, test a cell.

# --- symmetry, at the tolerance the FIT justifies -- and only after the a/b gate ---
from midas_hkls.ub_refine import refine_ub_from_gvectors
from midas_hkls.lattice_symmetry import holohedry_from_fit, tolerance_from_fit
fit  = refine_ub_from_gvectors(hkl, g / (2*np.pi))   # g = 1/d for an Angstrom cell; see fit.q_convention_note
holo = holohedry_from_fit(fit)    # a hand rel_tol=1e-3 said triclinic on both test positions (ENVELOPE §16)

# --- orientation + completeness from a cloud, cell DECLARED (phase-2 Route C) ---
from midas_defect.indexing import resolve_conventions, index_from_cloud
TOL = dict(two_theta=5e-3, eta=3e-2, omega=3e-2)      # rad -- the test run's values; set yours from residuals
def cloud_q(sign):                                    # resolve_conventions wants a CALLABLE of the omega sign
    return qlab_to_qsample(qlab, torch.deg2rad(torch.as_tensor(sign*omega_deg, dtype=qlab.dtype))
                           ).detach().cpu().numpy()
conv = resolve_conventions(cloud_q, I, row, col, frame_raw, crystal, geom, d_min=d_min, tolerances=TOL)
res  = index_from_cloud(cloud_q(conv.best_omega_sign), I, row, col, frame_raw, crystal, geom, mask,
                        d_min=d_min, convention=conv, max_two_theta_rad=TOL["two_theta"],
                        max_eta_rad=TOL["eta"], max_omega_rad=TOL["omega"])
# one set of tolerances, spelled two ways. frame_raw: fractional RAW frame index (omega is rebuilt
# as omega_first_deg + omega_step_deg*frame). crystal carries YOUR space group, now forwarded.

# --- joint refinement across domains, with bootstrap errors --------------------
from midas_hkls import DomainData, refine_cell_joint, split_with_error
fit = refine_cell_joint([DomainData(hkl=h, g=g, label=str(i)) for i, (h, g) in enumerate(D)],
                        system="orthorhombic",       # CONSTRAIN IT
                        cell0=cell0, two_pi=True, n_bootstrap=500)
delta, lo, hi = split_with_error(fit)     # bootstrap, never analytic

# --- phase identification: line counts, and a same-line-count null --------------
from midas_hkls.phase_id import PhaseCandidate, identify_phase
ranked = identify_phase(d_obs, [PhaseCandidate(name, crystal, cell_source="ambient"), ...],
                        free_scale=True, n_null_draws=200)   # d_obs: ONE domain's INDEXED reflections

# --- completeness: INDEXED / MISSED / MASKED / ABSENT ----------------------------
from midas_defect.completeness import audit_completeness, window_from_residuals

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
* **Gate an a/b splitting on rank AND partners AND asymmetry before computing δ, and decide the
  crystal system after that gate** (`ENVELOPE.md` §14, §16).
* **A pressure-gauge match needs a null**: the identical line scan for a phase known to be absent
  (`ENVELOPE.md` §15).

## Test run, 2026-09-10 — what following this doc set literally did

On the two delivered La3Ni2O7 positions (2604: 40 frames, in a DAC; S5 p = 248: 12 frames), harness and logs
are in the nickelate project's own record, not in this repository — the numbers here are copied from them, so do
not go looking for the files. Every gap it hit is now in these files.

| step | 2604 | S5 |
|---|---|---|
| Friedel centre (most pairs) | row 810.31, col 737.35 — 69 pairs, unseeded | 810.86, 737.00 — 44 pairs, 150 px seed |
| same, seeded 30 px off (default window) | 6 pairs, p = 0.0 — spurious | 7 pairs, p = 0.03 — spurious |
| ab initio | primitive 3.6575 / 3.7738 / 9.9025 Å; `to_conventional` → triclinic | `success=True` on a 1.466 Å axis |
| cell-free row | 13 rungs, 9.6188 Å = c/2 | 3-rung rows only — no ladder |
| ω sign (`resolve_conventions`) | +1: 58 vs −1: 1 | +1: 11 vs −1: 2 |
| `index_from_cloud`, cell declared | 58 INDEXED / 0 MISSED / 5 MASKED / 37 ABSENT | 11 / 0 / 2 / 22 (9 INDEXED before the P12 fix) |
| series member | row repeat n = 2 +0.0 %, n = 3 +51.4 %; `identify_phase` on indexed: n = 2 first | undeterminable (c-term median 0.020) |
| gauge, unsubtracted median | Re a = 2.669 Å, 5 of 6 lines (8 matched rings), V/V₀ 0.9033, 46.0 GPa | Pt a = 3.8475 Å, 2 lines, 18.7 GPa |

Followed literally it also produced four false positives, each now blocked by a gate in `ENVELOPE.md`: an
a/b splitting of 1.80 % [0.17, 10.36] on 2604 from a set with zero partners (§14); δ = 0.94 % from 12
reflections on S5 (§5); gauge matches from dense ring lists (§15); and crystal systems decided by
`holohedry_from_fit` before any a/b gate (§16). And it found one package bug: `index_from_cloud` converged
every crystal's cell with I-centring (`PACKAGE_NOTES.md` §10). The 2604 pressure re-derived here agrees
with a claim that is recorded as PROVISIONAL, not established.

## Halt conditions

Stop and ask when: the lattice will not converge from any seed; a fitted parameter rails at a
bound (a railed fit is not a fit — print bounds); the answer changes sign or ordering under a
re-seed; two records disagree on one quantity; or `ENVELOPE.md` says the ask is unobtainable.
