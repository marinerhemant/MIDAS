"""friedel_wedge on QUANTISED omegas (single-frame spots report the frame label).

Real far-field peak lists have half of their omegas exactly on the frame grid. The
legacy MAD-scaled Huber loss reads the lattice-valued separation errors (0, +-step)
as outliers and returns a wedge biased toward 0 (measured on AlON FF layers,
2026-10-12: Huber -0.0268 vs least squares -0.0307 deg; snapping ALL omegas of a
synthetic control under Huber returns W = 0). These tests snap a known fraction of
the omegas of a forward-simulated truth and pin the behaviour of each estimator.
"""
import math

import numpy as np
import pytest

from midas_joint_ff_calibrate.friedel_wedge import (
    find_friedel_pairs, fit_wedge_omega_scale, snapped_members,
)
from tests.test_friedel_wedge import LSD, OME_REF, simulate

STEP = 0.25
TRUTH = -0.04


@pytest.fixture(scope="module")
def spots():
    return simulate(250, TRUTH, 0.0, seed=11)


def _snap(o, frac, seed=0):
    """Snap each SPOT independently with probability ``frac`` to the nearest frame label."""
    rng = np.random.default_rng(seed)
    pick = rng.random(o.size) < frac
    lab = OME_REF + STEP * np.round((o - OME_REF) / STEP)
    return np.where(pick, lab, o)


def _fit(spots, frac, loss, *, junk=0.0):
    g, r, o, y, z, pos = spots
    o = _snap(o, frac)
    if junk:
        rng = np.random.default_rng(5)
        bad = rng.random(o.size) < junk
        o = np.where(bad, o + rng.choice([-1, 1], o.size) * rng.uniform(0.3, 0.9, o.size), o)
    pairs = find_friedel_pairs(g, r, o, y, z, pos_by_grain=pos)
    cls, phase, f_on = snapped_members(pairs, OME_REF, STEP)
    x, s, n, boots, info = fit_wedge_omega_scale(
        pairs, lsd_um=LSD, ome_ref=OME_REF, n_boot=0, loss=loss, omega_step=STEP,
        cls=cls, return_info=True)
    return float(x[0]), float(x[1]), cls, f_on, info


def test_grid_detection_fraction_and_phase(spots):
    _, _, cls0, f0, _ = _fit(spots, 0.0, "quant")
    assert f0 < 0.08 and not cls0.any()               # nothing snapped -> nothing flagged
    _, _, cls5, f5, _ = _fit(spots, 0.5, "quant")
    assert 0.40 < f5 < 0.60                            # ~half of the members sit on the grid
    _, _, cls10, f10, _ = _fit(spots, 1.0, "quant")
    assert f10 > 0.95 and (cls10 == 2).mean() > 0.95   # all snapped: nearly all pairs have both


def test_clean_data_all_estimators_agree(spots):
    wq = _fit(spots, 0.0, "quant")[0]
    wh = _fit(spots, 0.0, "huber")[0]
    wl = _fit(spots, 0.0, "ls")[0]
    for w in (wq, wh, wl):
        assert w == pytest.approx(TRUTH, abs=0.003)
    assert wq == pytest.approx(wh, abs=0.002)


@pytest.mark.parametrize("frac", [0.5, 1.0])
def test_quant_recovers_truth_under_snapping(spots, frac):
    w = _fit(spots, frac, "quant")[0]
    assert w == pytest.approx(TRUTH, abs=0.004)


def test_legacy_huber_collapses_to_zero_when_most_omegas_are_snapped(spots):
    """Documents the pathology the quantisation-aware estimator fixes.

    When more than half of the pair residuals are exactly 0 (both omegas on the grid
    and the separation unchanged) the MAD scale is 0, every non-zero residual gets
    weight 0 and the Huber fit sits at W = 0 whatever the data say. With a partly
    snapped list the legacy bias is milder and data dependent (AlON FF layers:
    -0.0268 vs -0.0307 deg); only the collapse is pinned here."""
    wh, _, _, _, info = _fit(spots, 1.0, "huber")
    assert abs(wh) < 0.005                               # truth is -0.04
    wq = _fit(spots, 1.0, "quant")[0]
    assert wq == pytest.approx(TRUTH, abs=0.004)


def test_legacy_huber_collapses_earlier_for_a_small_wedge():
    sp = simulate(250, -0.015, 0.0, seed=11)
    assert abs(_fit(sp, 0.8, "huber")[0]) < 0.003        # truth -0.015, 80 % snapped
    assert _fit(sp, 0.8, "quant")[0] == pytest.approx(-0.015, abs=0.003)
    assert _fit(sp, 1.0, "quant")[0] == pytest.approx(-0.015, abs=0.003)


def test_plain_ls_is_unbiased_under_pure_rounding(spots):
    assert _fit(spots, 1.0, "ls")[0] == pytest.approx(TRUTH, abs=0.004)      # rounding is zero-mean


def test_trimming_protects_against_one_sided_wrong_pairs():
    """15 % of spots shifted by +0.3..+0.9 deg (wrong pairs). Plain LS is dragged; the
    quantisation-aware fit trims at the physical bound (step/2 per snapped member)."""
    g, r, o, y, z, pos = simulate(250, TRUTH, 0.0, seed=11, ome_noise=0.04)
    rng = np.random.default_rng(1)
    bad = rng.random(o.size) < 0.15
    oo = np.where(bad, o + rng.uniform(0.3, 0.9, o.size), o)
    pairs = find_friedel_pairs(g, r, oo, y, z, pos_by_grain=pos)
    wl = fit_wedge_omega_scale(pairs, lsd_um=LSD, ome_ref=OME_REF, n_boot=0, loss="ls")[0][0]
    wq = fit_wedge_omega_scale(pairs, lsd_um=LSD, ome_ref=OME_REF, n_boot=0, loss="quant",
                               omega_step=STEP)[0][0]
    assert abs(wl - TRUTH) > 0.003                        # untrimmed LS is biased
    assert abs(wq - TRUTH) < 0.0015


def test_class_weights_beat_equal_weights():
    """Class-0 pairs precise, class-2 pairs corrupted by heavy zero-mean noise: inverse-variance
    weights recover the truth; equal weights are dragged by the noisy class."""
    g, r, o, y, z, pos = simulate(250, TRUTH, 0.0, seed=1, ome_noise=0.0)
    rng = np.random.default_rng(101)
    flag = rng.random(o.size) < 0.5
    oc = np.where(flag, o + rng.normal(0, 0.3, o.size), o)
    pairs = find_friedel_pairs(g, r, oc, y, z, pos_by_grain=pos)
    cls = flag[pairs.idx_a].astype(int) + flag[pairs.idx_b].astype(int)
    wl = fit_wedge_omega_scale(pairs, lsd_um=LSD, ome_ref=OME_REF, n_boot=0, loss="ls")[0][0]
    wq = fit_wedge_omega_scale(pairs, lsd_um=LSD, ome_ref=OME_REF, n_boot=0, loss="quant",
                               omega_step=STEP, cls=cls)[0][0]
    assert abs(wq - TRUTH) < 0.0005
    assert abs(wl - TRUTH) > 0.0015


def test_class_variances_reflect_rounding(spots):
    *_, info = _fit(spots, 0.5, "quant")
    cs = info["class_sigma_deg"]
    assert cs[0] < cs[1] < cs[2]                        # more snapped members -> larger scatter
    assert cs[2] < 0.35 * STEP + 0.1                   # bounded by the physical trim, not a MAD
    assert info["class_counts"][0] > 100 and info["class_counts"][2] > 100


def test_per_class_wedges_agree_for_one_true_wedge(spots):
    """One fixed axis tilt => the same wedge in every quantisation class."""
    g, r, o, y, z, pos = spots
    pairs = find_friedel_pairs(g, r, _snap(o, 0.5), y, z, pos_by_grain=pos)
    cls, _, _ = snapped_members(pairs, OME_REF, STEP)
    ws = []
    for c in (0, 1, 2):
        sub = pairs.subset(cls == c)
        x, _, n, bs = fit_wedge_omega_scale(sub, lsd_um=LSD, ome_ref=OME_REF, fit_omega_scale=False,
                                            eps0=0.0, wedge0=TRUTH, max_cot=np.inf, n_boot=30,
                                            loss="quant", omega_step=STEP, cls=cls[cls == c])
        ws.append((float(x[0]), float(bs[:, 0].std()), n))
    for w, se, n in ws:
        assert n > 100
        assert abs(w - TRUTH) < max(4 * se, 0.004)


def test_default_estimator_is_quantisation_aware():
    import inspect
    from midas_joint_ff_calibrate.friedel_wedge import friedel_wedge
    assert inspect.signature(friedel_wedge).parameters["loss"].default == "quant"
    assert inspect.signature(fit_wedge_omega_scale).parameters["loss"].default == "quant"
    from midas_joint_ff_calibrate import cli
    src = inspect.getsource(cli)
    assert '"--loss"' in src and 'default="quant"' in src


def test_unknown_loss_is_rejected(spots):
    g, r, o, y, z, pos = spots
    pairs = find_friedel_pairs(g, r, o, y, z, pos_by_grain=pos)
    with pytest.raises(ValueError, match="loss"):
        fit_wedge_omega_scale(pairs, lsd_um=LSD, ome_ref=OME_REF, n_boot=0, loss="nope")
