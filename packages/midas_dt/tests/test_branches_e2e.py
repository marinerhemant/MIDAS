"""Both branches end to end, and the gap between them. Needs the engine."""

from __future__ import annotations

import numpy as np
import pytest

from midas_dt.branches import compare, run_fit_then_recon, run_recon_then_fit
from midas_dt.channels import Channel
from midas_dt.sinogram import assemble

pytest.importorskip("midas_tomo")
pytest.importorskip("scipy")
from midas_tomo import backend_c  # noqa: E402

if not backend_c.available():
    pytest.skip(f"engine not built: {backend_c.why_unavailable()}",
                allow_module_level=True)


@pytest.fixture(scope="module")
def stack():
    """A disc whose peak SHIFTS across the sample.

    A uniform peak position would make both branches trivially agree, which
    would prove nothing. The whole question is what happens when RMEAN varies
    along a ray.
    """
    n_trans, n_omega, n_r = 32, 60, 24
    rng = np.random.default_rng(0)
    r = np.linspace(105.0, 125.0, n_r)
    omega = np.linspace(0.0, 180.0, n_omega, endpoint=False)
    x = np.linspace(-1, 1, n_trans)

    inten = np.zeros((n_trans, n_omega, 1, n_r))
    for i, th in enumerate(np.deg2rad(omega)):
        for j, xj in enumerate(x):
            # sample: a disc of radius 0.6; peak centre varies with position
            if abs(xj) > 0.6:
                continue
            centre = 113.0 + 3.0 * xj * np.cos(th)
            amp = 200.0 * np.sqrt(max(0.36 - xj * xj, 0.0))
            inten[j, i, 0] = amp * np.exp(-0.5 * ((r - centre) / 2.5) ** 2)
    inten = np.clip(inten, 0, None) + rng.uniform(0, 0.5, inten.shape)
    return assemble(inten, np.abs(inten), omega,
                    Channel(105, 125, r_bin=(125 - 105) / n_r, eta_bin=360),
                    snake=False)


def test_both_branches_produce_maps(stack):
    a = run_fit_then_recon(stack, outputs=("RMEAN", "TotalIntensity"), n_cpus=2)
    b = run_recon_then_fit(stack, outputs=("RMEAN", "TotalIntensity"), n_cpus=2)
    assert set(a.maps) == set(b.maps) == {"RMEAN", "TotalIntensity"}
    assert a.size == b.size


def test_additive_output_is_exact_in_both_branches(stack):
    """TotalIntensity adds along a ray, so Branch A needs no correction."""
    a = run_fit_then_recon(stack, outputs=("TotalIntensity",), n_cpus=2)
    assert a.linearity["TotalIntensity"] == "exact"


def test_rmean_is_flagged_weighted_not_exact(stack):
    a = run_fit_then_recon(stack, outputs=("RMEAN",), weighting="intensity", n_cpus=2)
    assert a.linearity["RMEAN"] == "weighted-moment"
    assert "RMEAN" in a.approximate_outputs()


def test_unweighted_branch_a_marks_rmean_approximate(stack):
    a = run_fit_then_recon(stack, outputs=("RMEAN",), weighting="none", n_cpus=2)
    assert a.linearity["RMEAN"] == "approximate"


def test_weighting_changes_the_rmean_map(stack):
    """The correction must actually do something.

    If weighted and unweighted agreed, the weighted form would be pointless
    and the linearity warning would be theatre.
    """
    w = run_fit_then_recon(stack, outputs=("RMEAN",), weighting="intensity", n_cpus=2)
    u = run_fit_then_recon(stack, outputs=("RMEAN",), weighting="none", n_cpus=2)
    good = np.isfinite(w.maps["RMEAN"]) & np.isfinite(u.maps["RMEAN"])
    assert good.sum() > 10
    diff = np.abs(w.maps["RMEAN"][good] - u.maps["RMEAN"][good])
    assert diff.max() > 1e-6, "weighted and unweighted RMEAN are identical"


def test_compare_quantifies_the_branch_gap(stack):
    """The headline diagnostic: measure the disagreement, do not hide it."""
    a = run_fit_then_recon(stack, outputs=("RMEAN", "TotalIntensity"),
                           weighting="intensity", n_cpus=2)
    b = run_recon_then_fit(stack, outputs=("RMEAN", "TotalIntensity"), n_cpus=2)
    stats = compare(a, b)
    for name in ("RMEAN", "TotalIntensity"):
        assert stats[name]["n"] > 0, f"no overlapping voxels for {name}"
        assert np.isfinite(stats[name]["rel_rms"])
    print("\n  branch gap:", {k: round(v["rel_rms"], 4) for k, v in stats.items()})


def test_reject_unknown_weighting(stack):
    with pytest.raises(ValueError, match="weighting must be"):
        run_fit_then_recon(stack, weighting="linear", n_cpus=2)
