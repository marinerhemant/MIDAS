# Phase 3 — refining the cell, and deciding its symmetry and distortion MODE

## Refine, with the symmetry you believe imposed

```python
from midas_defect.rows import refine_lattice, refine_to_convergence
from midas_hkls import DomainData, refine_cell_joint, split_with_error
```

**Constrain the cell.** Six extra free parameters (free triclinic) buy about **3 % in rms** and
absorb noise as a distortion: on one dataset an apparent ~1.6 % a/b split was exactly that,
and half of a 2 % split on the full spot list was multi-domain contamination.

**Errors by bootstrap, never analytic:**

```python
fit = refine_cell_joint(domains, system="orthorhombic", cell0=cell0,
                        two_pi=True, n_bootstrap=500)
delta, lo, hi = split_with_error(fit)
```

**Refine several domains JOINTLY when they share a cell.** It shares one cell across
orientations and is the only way to use domains that individually lack the partners.

**Pass `sigma_rtn` explicitly to `refine_to_convergence`** — it defaults to
`(0.0071, 0.0145, 0.0094)`, a radial/transverse/normal residual budget measured on one DAC
sample. It is not universal.

## Decide the distortion MODE before measuring its magnitude

**This is the step most often skipped, and it inverts conclusions.**

```python
from midas_hkls import ab_separable, shear_separable, distortion_condition
from midas_hkls.distortion_mode import (can_distinguish_modes, diagonal_B_can_express,
                                        supercell_to_subcell, supercell_hkl_to_subcell)
# distortion_mode is a MODULE, not a function
```

In a Ruddlesden-Popper **subcell**, an Fmmm supercell distortion is a **γ SHEAR** —
γ = 89.613° vs 90.387° for the two variants — with **a and b IDENTICAL**. Fitting a diagonal
(a ≠ b) cell to sheared data manufactures a splitting that is not there, and a diagonal B
matrix cannot express the real mode at all.

| question | function | what it means |
|---|---|---|
| can this set separate a from b at all? | `ab_separable` | rank ≥ 2 — **necessary, NOT sufficient**: also a partner (`partner_multiplicity`) and `index_asymmetry` (`ENVELOPE.md` §14) |
| can it separate a, b AND the γ shear? | `shear_separable` | rank == 3 — **required if a shear is possible** |
| how WELL do a and b separate? | `distortion_condition` | finite = separable; large = badly conditioned |

`supercell_to_subcell(A, B)` turns an Fmmm supercell (A, B) into the subcell `(a, b, γ)` it really is;
`supercell_hkl_to_subcell(H, K, L)` re-indexes reflections so the subcell tests can run on them;
`can_distinguish_modes(subcell_hkls)` says whether a set holds BOTH the a≠b-sensitive and the
shear-sensitive families; and `diagonal_B_can_express("gamma_shear")` is False — a diagonal B pins γ = 90°.

The two modes are probed by **different reflections, and they are complementary**: `(1,0)/(0,1)`
splits under a≠b and NOT at all under the γ shear; `(1,1)/(1,-1)` is the reverse. A pipeline
that only ever looks at one pair cannot tell you which mode it is in.

## Symmetry from the metric — per entry, and not sharing a σ

```python
from midas_hkls.ub_refine import refine_ub_from_gvectors
from midas_hkls.lattice_symmetry import holohedry_from_fit, tolerance_from_fit
fit  = refine_ub_from_gvectors(hkl, g / (2*np.pi))   # g = 1/d for an Angstrom cell (fit.q_convention_note)
holo = holohedry_from_fit(fit)
```

**A hand-picked tolerance is a guess; the fit's covariance is not — but it is only as good as the fit.** On
the test run `holohedry(rel_tol=1e-3)` said triclinic on both positions, while `holohedry_from_fit` said
**orthorhombic** on 2604 (tolerance 0.0126) and **tetragonal** on S5 (0.108, from 12 reflections). The 2604
set had zero a/b partners and a condition number of 1395: a σ from that fit is small along exactly the
direction the data do not constrain. **Decide the system only after the a/b gate** (`ENVELOPE.md` §14, §16).

**Tolerance must be applied PER ENTRY of the metric tensor, not to `max|G|`.** A single global
tolerance scaled by the largest metric entry mislabels a `c/a ≈ 5` cell as triclinic, because
the c-row entries dwarf the in-plane ones.

**A symmetry verdict and a significance test that share one σ are ONE test.** If the σ that
decided "tetragonal" is the σ that decides "the split is not significant", the agreement is
circular. Measured: with the *radial* σ instead, one dataset's verdict moved from "tetragonal,
not significant" to "monoclinic, 3.3–3.7σ". **And test stability:** on that data
**17 of 71 single-spot deletions flipped the crystal system.** Report that fraction.

**The selector must not contain the answer.** Selecting the subset with `h+k+l even` and then
reporting the lattice returns body-centred/cubic at 97.5 % — the selector imposed it.

## Reporting an a/b splitting, if the envelope allows one at all

Check `ENVELOPE.md` §1–§5 and §14 first. **Gate, in this order:** `ab_separable` (rank) → a partner, on
more than one spot (`partner_multiplicity`) → `index_asymmetry` → `shear_separable` if a shear is possible
→ only then δ, by bootstrap, against the counting requirement of §5. The order is measured: on the test run
2604 passed rank and returned **δ = 1.80 % [0.17, 10.36] from zero partners**. S5 had partners, but each pair
rested on ONE spot on one side ((0,1) 2 : 1, (1,2) 1 : 3) — so it fails the second gate too — and returned
0.94 % [0.10, 9.11] from 12 reflections. Then, always together:

```python
from midas_hkls import ab_sensitive_mask, partner_multiplicity, index_asymmetry
```

* **`partner_multiplicity`** — a pair supported on ONE spot is one blob-finding decision away
  from nothing. Measured: 6 reflections called sensitive, but the (1,0) side was a single spot;
  remove it and all six stop being sensitive.
* **`index_asymmetry`** — a consistent SIGN across domains is the artifact signature, not a
  result (`ENVELOPE.md` §3).
* **`ab_sensitive_mask`** is the `|G|`-only rule and **understates a vector pipeline**: with
  full 3-D q, the in-plane azimuth `atan2(k/b, h/a)` depends on a/b for any h, k both non-zero,
  with no partner needed. Do not use the mask to argue insensitivity.
* **Never quote the sign** (`ENVELOPE.md` §2).

## Sanity checks that have each caught a real error

* **A railed fit is not a fit.** Print every fitted parameter against its bounds. One BM3
  returned "K0 = 150 GPa" with K0′ railed at its lower bound of 2.000000.
* **Reference plausibility gates to a fixed nominal cell, never a neighbour's** — a 1 %
  seed-referenced gate produced +0.128 ± 0.042 under a null with zero effect by construction.
* **Reconcile two records that give one quantity two values.** One project carried a = 3.6008
  in its preregistration and 3.6116 in its analysis, with nothing reconciling them; they turned
  out to differ by subset, but nobody had checked.
* **Never leave a solution file behind a geometry edit** (`phase-1-geometry.md`).
