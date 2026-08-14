# Releasing midas-dt

```bash
cd packages/midas_dt
./release.sh 0.1.0 --dry-run     # build + test, no commit or tag
./release.sh 0.1.0               # + commit and tag locally
./release.sh 0.1.0 --publish     # + push, GitHub release, CI publishes to PyPI
```

The tag is `midas-dt-v<version>`; `.github/workflows/python-packages.yml`
maps it to this package and publishes to PyPI via trusted publishing.

## Release midas-tomo first

`midas-dt` reconstructs through `midas-tomo` and floors it in
`pyproject.toml`. If that floor is not on PyPI, `pip install midas-dt`
fails for users while everything here stays green — the failure surfaces at
their install, not at ours. `release.sh --publish` checks PyPI for the floor
and refuses; it warns instead of blocking if it cannot reach PyPI.

Ordering for a coordinated release: **midas-tomo → midas-dt → midas-suite.**

## Release from a machine where the midas-tomo engine is built

There is no C in this package, but it **inherits midas-tomo's skip trap.**
`tests/test_recon.py` and `tests/test_branches_e2e.py` skip at module level
when `midas_tomo.backend_c.available()` is False — that is 15 tests, and they
are the ones that cover reconstruction and the branch comparison, i.e. the
reason the package exists. On a Mac with no FFTW the suite goes green having
never reconstructed anything.

| environment | result |
|---|---|
| midas-tomo engine built + torch | **202 passed** |
| no engine (e.g. a Mac with no FFTW) | 187 passed, 2 skipped |

Both measured 2026-08-14 at 0.2.0; the 202 on chiltepin
(`/home/beams12/S1IDUSER/opt/tomo_buildtest` on `PYTHONPATH`, shared env at
`/home/beams12/S1IDUSER/opt/envs/midas/bin/python`).

`release.sh` passes `-rs`, so the skip reasons print. **If you see the
187-passed line, you tested the wrapper.** Release from a host with the engine,
or run the suite there first.

Branch C (`tests/test_direct.py`) additionally needs `torch` and
`midas-invert` -- the `[direct]` extra. Without them the whole module skips,
so a green run on a torch-free machine has not exercised direct inversion at
all. Same trap as the engine, one layer up.

Tests marked `realdata` need the U3O8 scan on haydn
(`/scratch/s1iduser/mpe_nov22_midas2/`, see `docs/u3o8_2022_dataset.md`).
None are currently enabled by default; skipping them is not a reason to hold a
release.

## Before any release that touches `conventions.py`

`conventions.py` is the pinned-truth module: the 12-channel fit-output order,
which 3 of them are additive, `RECON_SIGN`, the ω sign, the dropped first
frame. Every one of these is a silent-wrong-answer bug when it changes, not a
crash.

- [ ] `tests/test_conventions.py` passes unchanged, or the change is
      justified against the C source (`IntegratorPeakFitOMP.c` `valTypes[]`,
      `PeakFit.c` `Rfit[]`) in the commit message
- [ ] `ADDITIVE_FIT_OUTPUTS` still lists exactly the quantities that add along
      a ray — adding a non-additive one to that set makes branch A silently
      back-project a meaningless quantity
- [ ] `tests/test_branches_e2e.py::test_additive_output_is_exact_in_both_branches`
      and `::test_compare_quantifies_the_branch_gap` still show the
      additive/non-additive split (measured: `TotalIntensity` 0.0,
      `RMEAN` 0.0085)

## Before a release that changes reconstruction

- [ ] `tests/test_recon.py` phantom round-trip passes
- [ ] `variance_samples` path still runs — it is opt-in and easy to break
      without noticing, since nothing else calls it
- [ ] no `skimage.transform.iradon` anywhere: reconstruction goes through
      `midas_tomo`. A test asserts this.

## Suite registration

`midas_dt` is in `packages/midas_suite` (`SUBPACKAGES` + `dependencies`).
The suite's CI smoke test imports every declared sub-package, so an
unpublished or unimportable `midas_dt` fails the suite build, not this one.


## Before a release that touches `direct.py`

Branch C has two failure modes that produce plausible numbers rather than
errors, so both are pinned by tests. If either test is weakened, stop.

- [ ] `test_moment_seed_is_on_the_right_order_of_magnitude` -- the seed used to
      be ~384x too large (peak of the summed lineout instead of a per-ray
      peak). Because Adam steps a fixed distance in raw parameter units, the
      solver then sat 1.5 px from a planted centre and got *worse* with a
      larger learning rate, which reads exactly like a degenerate model.
- [ ] `test_laplace_default_does_not_use_the_converged_loss` -- the loss is
      already weighted by 1/variance, so using it as `noise_var` counts the
      noise twice and inflates sigma by `sqrt(loss * N)`: 0.035 px became
      446 px on a 20 px window.
- [ ] `test_no_performance_claim_is_made_in_the_docstring` -- the gate on
      claiming Branch C beats Branch B. It stays until a preregistered
      comparison has actually been run.
