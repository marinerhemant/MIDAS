# Package notes — findings from the 2026-09-09 audit and the 2026-09-10 test run

Recorded because they are properties of the CODE, not of any one analysis, and someone
teaching from this doc set will hit them.

## 1. `midas_hkls.ab_sensitive_mask` was missing entirely — FIXED 2026-09-09

Three analysis scripts imported it and it had **never been committed** to any package, so they
could not run (`ImportError`). Now added to `midas_hkls/ab_splitting.py`, exported, and pinned
by 5 tests against the contract its call sites documented. Its two scope limits are in the
docstring and in `phase-3-refine.md`: it is the `|G|`-only rule (it understates a full 3-D
vector pipeline) and it is **blind to a γ shear** (it rejects `(1,1)/(1,-1)`, which is exactly
the pair that splits under an Fmmm-subcell shear).

**The general lesson:** a symbol that only ever existed in an editable working tree is
invisible to every check that matters. `git grep` it before quoting a function as "in the
package".

## 2. `midas_defect.rows` defaults to one material's cell, in SIX functions — OPEN, by design

`find_lattice_rows`, `index_from_row`, `index_from_pairs`, `match_mask`, `cell_from_row` and
`refine_to_convergence` carry La3Ni2O7 as **default arguments**:

```
a=3.6116, c=19.2516, space_group_number=139
refine_to_convergence(..., sigma_rtn=(0.0071, 0.0145, 0.0094))   # a DAC sample's residual budget
index_from_pairs(..., q_max_anchor=3.2)                          # blocks anchors above 3.2 1/A
```

A call that omits them is silently seeded with a nickelate — no error, a worse answer — and the project
that wrote them called `find_lattice_rows` and `cell_from_row` without a cell in its own production
raster driver. **Pass all of them explicitly, every call.** Left as-is because removing a default breaks
every existing caller. (The first version of this note said three functions; the 2026-09-10 audit found
six.)

## 3. `two_pi` defaults in opposite directions across the chain — OPEN

| module | default | meaning |
|---|---|---|
| `midas_defect.rows.*` | `two_pi=True` | q = 2π/d |
| `midas_hkls.ab_initio.index_ab_initio` | `two_pi=False` | q = 1/d |

Feeding a 2π cloud to the ab initio indexer without `two_pi=True` **fails to index rather than
raising**. Declare the convention once at the top of a script and pass it everywhere.

## 4. Where the cell machinery lives does not match what it does — WON'T FIX (deliberate)

`midas_defect.rows`, `.seed_index`, `.geometry` and `.completeness` are lattice/indexing
machinery inside the *defect* package, for historical reasons. Measured on one project,
**61.3 % of MIDAS import sites were cell/symmetry work** and three of the four heaviest
"defect" modules were these. Decision 2026-09-09: **document, do not move** — a relocation
costs a coordinated two-package release and dep-floor bump for a naming wart, and every
existing import would need a shim. Import them from `midas_defect`.

## 5. There is no MIDAS package for equation-of-state inversion — OPEN

Pressure from a marker or a sample cell is a dozen lines of scipy, written ad hoc each time and
checked by nothing. The guard rails are in `phase-4-phase-id.md` instead. If this recurs across
projects it deserves a home, with the EoS parameter tables and their published spreads
alongside — the spread across four published Re equations of state was **38.5–48.0 GPa** on one
measured compression, which is exactly the kind of thing a package should carry rather than a
script.

## 6. `BC_y` names the ROW in midas_integrate_v2 and the COLUMN in midas_defect — DOCUMENTED

`midas_integrate_v2.compat.pyfai.poni_to_bc` / `poni_file_to_bc` return `(BC_y, BC_z)` with BC_y along
the ROW axis (its flip test reflects BC_y against the row count); `midas_defect.Geometry.bcy_px` is the
COLUMN. Both tested, both self-consistent; the boundary swaps axes silently. `poni_file_to_row_col` exists
for exactly this boundary. Found 2026-09-10 when this doc set's own test run seeded a centre 123/59 px off.

## 7. `rod_path` — flat detector, `omega_sign`, `q_convention` default — DOCUMENTED + PINNED

Now in its docstring, with tests (`midas_defect/tests/test_rod_path_conventions.py`): it projects onto a
FLAT detector (no tilt, `tx` or distortion; ~14 px at 0.4 deg and Lsd ~ 350 mm); `omega_sign` changes only
the REPORTED and window-tested omega, never the rotation that places the point; `q_convention` defaults to
`"1/d"` while the rest of midas_defect is 2pi/d, and a 2pi B on the default drops points instead of
raising. These were the two "API gaps to file" in the La3Ni2O7 checkpoint; neither was ever filed.

## 8. `calibrate()` does not return refined panel values — OPEN, a design decision

See `manuals/calibrate-integrate/DIAGNOSIS.md` ("Panels made the in-loop strain drop, but nothing
downstream changed") and the `calibrate()` docstring. Not a hot-fix: in the default `panel_mode="radius"`
the values are per-(panel, ring) offsets that could not transfer to a sample's reflections even if returned.

## 9. Added 2026-09-10 — capability the La3Ni2O7 work relied on and no package had

| function | where | what it replaced |
|---|---|---|
| `detector_angle_maps(geom)` | `midas_defect.geometry` | a local 2theta/eta map helper with 94 importers, eta untilted |
| `decoy_test`, `inflated_cell` | `midas_defect.honesty` | `step17` decoy cell — dispositioned PORT on 09-01, never landed |
| `feature_in_raw` | `midas_defect.honesty` | `step55` raw-frame check — same |
| `ring_harmonics` | `midas_calibrate_v2.ring_anisotropy` | a local cos 2eta ring fit; the on-frame a/b floor |
| `ab_sensitive_mask` | `midas_hkls.ab_splitting` | imported by three scripts, never committed (item 1) |
| `check_poni_against_friedel` | `midas_calibrate_v2.poni_check` | the project's hand row-flip check; per axis, by distance to Friedel centres (ring sharpness failed) |
| `selfcalibrate_from_crystals` | `midas_defect.selfcal` | `step28_selfcal.py` — detector tilts from indexed domains, α = β = 90 imposed |
| `find_domains` | `midas_defect.domains` | the project's `reduce_v4.py` (not shipped) — the per-position multi-domain driver |
| `targeted_snr`, `targeted_recovery` | `midas_defect.completeness` | the project's `predicted_recovery.py` (not shipped) — extraction at predicted sites against a same-ring null |
| `centred_L_nodes`, `diffuse_to_bragg` | `midas_defect.rod_profile` | `step23_hk_map.py` — the (h,k) rod test, normalised by the rod's own nodes |

## 10. `index_from_cloud` converged every crystal's cell with I-centring — FIXED 2026-09-10

`indexing.index_from_cloud` called `refine_to_convergence` without `space_group_number` (default 139), and
`rows.index_from_pairs` and `rows.index_by_grid` called `match_mask` three times without the
`space_group_number` they had in scope. Any crystal that is not I4/mmm had I4/mmm extinctions applied inside
the package. Forwarded now; `midas_defect/tests/test_space_group_forwarded.py` fails on any new in-package
call to a default-cell function that drops an in-scope space group, and checks sg 69 reaches the cell
convergence. On S5 (Fmmm): INDEXED 9 → 11, median residual 2.05 → 0.61 px; 2604 unchanged. The residual
budget followed on 2026-09-11: `index_from_cloud(sigma_rtn=..., tol_sigma=...)` reaches the convergence;
left unset it still runs on La3Ni2O7's measured budget, so measure yours (`window_from_residuals`) and pass
it. A held-out S5 reader measured a 0.17 % difference in `a` between that convergence and `refine_cell_joint`
on the same reflections before this argument existed.

## 11. `resolve_conventions` and `index_from_cloud` spell one set of tolerances two ways — OPEN

A dict keyed `two_theta` / `eta` / `omega` against keywords `max_two_theta_rad` / `max_eta_rad` /
`max_omega_rad`; and `resolve_conventions` wants the cloud as a callable of the ω sign, not an array. The
README contract shows both.

## 12. `ab_separable` is necessary, not sufficient — DOCUMENTED

Rank ≥ 2 without an (h,k)/(k,h) partner does not protect a splitting from a radial systematic. The docstring
now says so with the 2604 measurement (`ENVELOPE.md` §14).

## 13. `index_ab_initio(...).success` is True on a physically impossible cell — OPEN

S5: 1.466 / 2.901 / 3.285 Å, V = 11.9 Å³, 27 of 203 spots. Check the shortest length, the volume and the
indexed fraction before using the result (`phase-2-lattice.md`).

## 14. `to_conventional` decides the crystal system at one tolerance — DOCUMENTED

Default → triclinic on 2604's ab initio cell; `rel_tol = 0.0768` → non-standard orthorhombic-P; never the
body-centred cell (`phase-2-lattice.md`).

## 15. `holohedry_from_fit` on an ill-conditioned fit — OPEN, not diagnosed

"Orthorhombic" at tolerance 0.0126 on a 2604 set with zero partners and condition number 1395. Whether
`tolerance_from_fit`'s σ is understated along the unconstrained direction has not been tested. Until it is,
decide symmetry after the a/b gate (`ENVELOPE.md` §16).

## 16. `omega_smear_duplicates` pairs `hkl_old[j]` with `rc_old[j]` — DOCUMENTED (a caller trap)

A caller that concatenates per-domain hkl lists but takes positions from a union mask misaligns them once two
earlier domains exist in an order different from the global spot index — always, in practice. The La3Ni2O7 driver
did; verified ESTABLISHED on four lenses (claim 5b5f7fb80b36): a synthetic case missed an exact duplicate 5.8 px
away, and replaying the real call at five positions missed 142 duplicates and changed 15 of 26 accept/reject
decisions. `domains.find_domains` builds all three arrays per domain. On that project's full raster, aligning the call
removed 364 of 1278 domains, 241 of them pair-seeded (provisional; `manuals/defect/phase-2-index.md`).

## 17. `midas_index` finds an orientation for a KNOWN cell — ROUTING

The far-field indexer searches orientations against the cell it is given and does not determine one; the cell is
refined per grain afterwards (`midas_fit_grain`), starting from that same cell. For a disputed cell stay in this
doc set. If you do bridge spots into it, **obs column 8 is the radial displacement from the ring (µm), not
intensity** — a bridge that got this wrong reported "midas_index fails" until a synthetic positive control found
it (`packages/midas_index/README.md`).

## 18. Four traps from the held-out dry runs (2026-09-10/11)

Two fresh-context readers ran this doc set on positions whose answers are in no document (2604 p = 184, S5
p = 378) and both delivered on every rubric item. What slowed them:

* **`resolve_conventions` seeds from whatever it is given.** Gasket and anvil spots among the 30 brightest chose the
  wrong ω sign at 4 : 1 and called it decisive; `ConventionScan.decisive` now needs the winner to reach
  `min_assigned` (8) as well as twice the runner-up. Feed it seedable spots only (`phase-2-lattice.md`, Route C).
* **`ewald_crossing_omegas` takes ONE q vector** (it reshapes to 3). Loop over reflections, or vectorise it yourself.
* **`qlab_to_pixel` and `pixel_to_qlab` return tensors on the default device** (MPS on a Mac, CUDA where present), and
  `np.asarray` on those fails. Pass `device="cpu"`; `detector_angle_maps` defaults to CPU since 2026-09-10.
* **`selfcalibrate_from_crystals` has no convergence guarantee.** Its refine → re-match loop cycled between three
  tilt solutions (0.33 / 0.48 / 0.57° total, PONI 0.41°) on 2604 p = 184. Carry every solution the loop visits as a
  geometry systematic (that reader's cell span: a 3.6043–3.6079, c 19.2444–19.2554 Å); hold the distance to the
  calibrant's, because freeing it inherits c from the seed.
