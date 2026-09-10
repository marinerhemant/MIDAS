# Package notes — findings from the 2026-09-09 audit

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

## 2. `midas_defect.rows` defaults to one material's cell — OPEN, by design for now

`find_lattice_rows`, `index_from_row` and `refine_to_convergence` carry La3Ni2O7 as **default
arguments**:

```
a=3.6116, c=19.2516, space_group_number=139
refine_to_convergence(..., sigma_rtn=(0.0071, 0.0145, 0.0094))   # a DAC sample's residual budget
```

A call that omits them is silently seeded with a nickelate — it will not error, it will return
a worse answer. **Pass all four explicitly, every call.** Left as-is rather than changed
because removing a default is a breaking change for every existing caller; flagged here and in
`README.md` instead.

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
