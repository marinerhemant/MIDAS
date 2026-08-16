"""Lazy candidate evaluation in the batched Nelder-Mead must be bit-identical.

The loop used to evaluate all four trial points (reflection, expansion, outside
and inside contraction) on the whole active batch every iteration, plus a
full-batch re-evaluation whenever ANY simplex shrank. But the four branches are
mutually exclusive and all decided by the reflection alone, so each simplex
needs at most ONE of the other three; and only the shrinking simplices need the
shrink re-evaluation.

Skipping that work is only legitimate if the objective is invariant to batch
composition -- fn(x[subset], idx[subset]) == fn(x, idx)[subset]. These tests
assert both that invariance and the resulting bit-identity, because the fit's
output is a scientific result: the optimiser may do less work, but it must land
on exactly the same point.
"""
import numpy as np
import pytest
import torch

from midas_nf_fitorientation.torch_nm import batched_nelder_mead


def _rosenbrock_batch(shift):
    """A batch of shifted 3-D Rosenbrocks; row i depends only on row i."""
    def fn(x, idx):
        s = shift[idx]
        y = x - s
        return ((1.0 - y[:, 0]) ** 2
                + 100.0 * (y[:, 1] - y[:, 0] ** 2) ** 2
                + 10.0 * (y[:, 2] - y[:, 1] ** 2) ** 2)
    return fn


def _quartic_batch(shift):
    """Multi-modal enough to exercise every branch, incl. repeated shrinks."""
    def fn(x, idx):
        s = shift[idx]
        y = x - s
        return (y ** 4 - 3.0 * y ** 2 + 0.5 * y).sum(dim=1)
    return fn


def test_lazy_evaluation_saves_objective_calls():
    """Each simplex needs at most ONE of expansion/outside/inside contraction.

    Evaluating all four every iteration, plus a full-batch shrink re-evaluation,
    cost about twice the objective work the algorithm asks for.
    """
    for B in (1, 64, 1024):
        g = torch.Generator().manual_seed(B)
        shift = torch.randn(B, 3, generator=g, dtype=torch.float64)
        x0 = torch.randn(B, 3, generator=g, dtype=torch.float64)
        bd = torch.stack([x0 - 3.0, x0 + 3.0], dim=-1)
        res = batched_nelder_mead(
            _rosenbrock_batch(shift), x0.clone(), bd.clone(), max_iter=200)
        # eager cost is 4 per iteration + n on any shrink; lazy is 1 + <=1 + shrunk
        assert res.n_evals < 4 * res.n_iter, (
            f"B={B}: {res.n_evals} evals over {res.n_iter} iterations is "
            f"not below the eager 4-per-iteration floor")


def test_result_matches_the_pre_optimisation_values():
    """Golden values captured from a run verified bit-identical to the eager
    implementation at git HEAD (checked for B = 1, 7, 64, 257, 1024).

    If a future change to the loop moves these, it has changed the optimiser's
    answer -- which for this code is a scientific result, not an
    implementation detail.

    Compared by KIND, not by bits. The goldens are of two sorts and only one of
    them carries information:

    * the two entries of order 1 and 170 are genuine local minima of the
      shifted Rosenbrock. A change in which branch a simplex takes moves these,
      so they are pinned to a tight relative tolerance.
    * the rest are convergence residuals ~1e-12, i.e. "reached zero". Their
      digits are the accumulated rounding of the path taken there, not a
      result. Asserting them bit-exactly asserts something untrue: the goldens
      were captured on macOS/arm64 and CI is Linux/x86-64, where a different
      BLAS and vectorisation give 1.3580115637905315e-12 against the recorded
      1.3580121932013689e-12 -- a 5e-7 relative difference on a number whose
      meaningful content is "< 1e-9". That mismatch failed the 0.9.1 release
      while the optimiser was behaving identically.

    So: minima pinned, residuals required to have converged.
    """
    g = torch.Generator().manual_seed(64)
    shift = torch.randn(64, 3, generator=g, dtype=torch.float64)
    x0 = torch.randn(64, 3, generator=g, dtype=torch.float64)
    bd = torch.stack([x0 - 3.0, x0 + 3.0], dim=-1)
    res = batched_nelder_mead(
        _rosenbrock_batch(shift), x0.clone(), bd.clone(), max_iter=200)
    golden = [8.095249818836124e-13, 170.12072462155822, 1.9093557834387316,
              1.3580121932013689e-12, 1.0389418259387467e-12,
              1.3143181828195085e-12]
    CONVERGED = 1e-9
    for i, (got, want) in enumerate(zip(res.fun[:6].tolist(), golden)):
        if want < CONVERGED:
            assert got < CONVERGED, (
                f"entry {i} failed to converge: {got!r} (golden {want!r})")
        else:
            assert got == pytest.approx(want, rel=1e-9), (
                f"optimiser result moved at entry {i}: {got!r} != {want!r}")


def test_objective_is_invariant_to_batch_composition():
    """The property the whole optimisation rests on.

    If fn(x[subset]) differed from fn(x)[subset] -- e.g. an objective that
    normalised across the batch -- evaluating a compacted subset would silently
    change the answer.
    """
    g = torch.Generator().manual_seed(0)
    shift = torch.randn(32, 3, generator=g, dtype=torch.float64)
    fn = _rosenbrock_batch(shift)
    x = torch.randn(32, 3, generator=g, dtype=torch.float64)
    idx = torch.arange(32)
    full = fn(x, idx)
    sel = torch.tensor([3, 17, 0, 31, 8])
    assert torch.equal(fn(x[sel], idx[sel]), full[sel])


# ---------------------------------------------------------------------------
#  hard_fraction: the redundant bounds recomputation
# ---------------------------------------------------------------------------

def test_bounds_already_applied_is_a_no_op_when_valid_encodes_bounds():
    """Skipping the recomputation must not change the fraction.

    project_to_detector already folds the six bounds comparisons into ``valid``,
    so on the fit path ``valid * in_bounds == valid``. The flag exists so other
    callers -- who may pass a looser ``valid`` -- keep the check.
    """
    from midas_nf_fitorientation.obs_volume import ObsVolume

    D, B, M, F_, H, W = 2, 5, 7, 12, 16, 16
    g = torch.Generator().manual_seed(0)
    dense = (torch.rand(D, F_, H, W, generator=g) > 0.7)
    obs = ObsVolume(dense=dense, n_distances=D, n_frames=F_, n_y=H, n_z=W)

    frame = torch.randint(0, F_, (B, M), generator=g)
    y = torch.randint(0, H, (D, B, M), generator=g)
    z = torch.randint(0, W, (D, B, M), generator=g)
    valid = (torch.rand(B, M, generator=g) > 0.3).to(torch.float64)

    a = obs.hard_fraction(frame, y, z, valid)
    b = obs.hard_fraction(frame, y, z, valid, bounds_already_applied=True)
    assert torch.equal(a, b), "skipping in-bounds changed the fraction"


def test_bounds_check_still_applies_by_default_for_out_of_range_input():
    """A caller whose ``valid`` does NOT encode bounds must still be protected."""
    from midas_nf_fitorientation.obs_volume import ObsVolume

    D, B, M, F_, H, W = 1, 2, 4, 8, 8, 8
    # Frame indices are CLAMPED into range before the lookup, so the volume
    # must be dark at the clamp target for the two paths to differ at all --
    # otherwise an out-of-range spot still "hits" and the test proves nothing.
    dense = torch.ones(D, F_, H, W, dtype=torch.bool)
    dense[:, F_ - 1] = False                                  # clamp target dark
    obs = ObsVolume(dense=dense,
                    n_distances=D, n_frames=F_, n_y=H, n_z=W)
    frame = torch.tensor([[0, 1, 2, 99], [0, 1, 2, 3]])       # 99 out of range
    y = torch.zeros(D, B, M, dtype=torch.long)
    z = torch.zeros(D, B, M, dtype=torch.long)
    valid = torch.ones(B, M, dtype=torch.float64)             # deliberately loose

    checked = obs.hard_fraction(frame, y, z, valid)
    skipped = obs.hard_fraction(frame, y, z, valid, bounds_already_applied=True)
    # With a loose ``valid`` the two MUST differ -- that is the whole point of
    # keeping the check on by default.
    assert not torch.equal(checked, skipped), (
        "the default path must still reject the out-of-range spot")
