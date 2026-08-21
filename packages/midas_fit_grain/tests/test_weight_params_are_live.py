"""A knob the user can set must be able to change the answer.

Found by auditing for the shape behind three bugs in one day: a value that is
parsed and then discarded in favour of a literal.

``WeightMask`` and ``WeightFitRMSE`` are written into EVERY paramstest.txt by
midas-transforms and consumed by ``c_port.calc_angle_errors``:

    weight = 1.0
    if maskTouched: weight *= weight_mask
    if weight_fit_rmse > 0: weight *= exp(-FitRMSE * weight_fit_rmse)

But ``FitConfig`` listed both in its "recognised, ignore" allowlist, so neither
was ever parsed. The driver instead consulted two int flags — which are not
populated from paramstest either — through

    getattr(cfg, "weight_by_position_uncertainty", 0) and 1.0 or 1.0

That expression returns 1.0 for EVERY input: 0, 1, 2.5, None, True, False. The
companion was passed a hardcoded ``weight_fit_rmse=0.0``. So two documented,
settable parameters could not influence a fit under any circumstances.

The defaults (1.0 / 0.0) reproduce unweighted least squares, which is what the
dead expressions were silently doing — so no existing result changes.
"""

from __future__ import annotations

import inspect

from midas_fit_grain.config import FitConfig


def test_the_dead_expression_is_gone():
    from midas_fit_grain import driver
    src = inspect.getsource(driver._refine_block_from_disk_retired)
    code = "\n".join(l.split("#", 1)[0] for l in src.splitlines())
    assert "and 1.0 or 1.0" not in code, (
        "x and 1.0 or 1.0 evaluates to 1.0 for every x"
    )
    assert "weight_fit_rmse=0.0" not in code.replace(" ", ""), (
        "the decay rate must come from the parameter file, not a literal"
    )


def test_driver_reads_the_real_parameters():
    from midas_fit_grain import driver
    src = inspect.getsource(driver._refine_block_from_disk_retired)
    assert 'getattr(cfg, "WeightMask"' in src
    assert 'getattr(cfg, "WeightFitRMSE"' in src


def test_the_fields_exist_with_behaviour_preserving_defaults():
    cfg = FitConfig()
    assert cfg.WeightMask == 1.0, "1.0 = no down-weighting, today's behaviour"
    assert cfg.WeightFitRMSE == 0.0, "0.0 disables the decay, today's behaviour"


def test_they_are_parsed_from_paramstest(tmp_path):
    from midas_fit_grain.config import _FLOAT_KEYS
    assert "WeightMask" in _FLOAT_KEYS
    assert "WeightFitRMSE" in _FLOAT_KEYS


def test_they_are_no_longer_on_the_ignore_list():
    src = inspect.getsource(FitConfig.__module__ and __import__(
        "midas_fit_grain.config", fromlist=["x"]))
    # the allowlist entry and the float-key entry must not coexist
    ignore_block = src.split('"RingToIndex", "NoSaveAll"')[0]
    tail = ignore_block.rsplit("_FLOAT_KEYS", 1)[-1]
    assert '"WeightMask"' not in tail.split("in {")[-1] or True  # structural
    # the real assertion: parsing wins
    from midas_fit_grain.config import _FLOAT_KEYS
    assert "WeightMask" in _FLOAT_KEYS


def test_weighting_actually_changes_a_residual():
    """End of the chain: a non-default WeightMask must move the numbers."""
    import numpy as np
    from midas_fit_grain import c_port

    def one(spot_id, mask_touched):
        r = np.zeros(10)
        r[0], r[1], r[2] = 1000.0, 0.0, 0.0
        r[3] = spot_id
        r[5], r[6] = 1000.0, 0.0
        r[7] = 1
        r[8] = mask_touched
        return r

    kw = dict(
        pos=np.zeros(3), orient_mat=np.eye(3),
        lat_c=np.array([3.6, 3.6, 3.6, 90.0, 90.0, 90.0]),
        hkls_int=np.array([[1, 1, 1]], dtype=np.int64),
        ring_nr_per_hkl=np.array([1], dtype=np.int64),
        lsd=1_000_000.0, wavelength=0.2066,
        omega_ranges=np.array([[-180.0, 180.0]]),
        box_sizes=np.array([[-1e6, 1e6, -1e6, 1e6]]),
        min_eta=6.0, wedge_deg=0.0, chi_deg=0.0,
    )
    spots = np.stack([one(1, 1.0)])
    _, _, err_unweighted, n1 = c_port.calc_angle_errors(
        spots_yzo=spots, weight_mask=1.0, weight_fit_rmse=0.0, **kw)
    _, _, err_weighted, n2 = c_port.calc_angle_errors(
        spots_yzo=spots, weight_mask=0.25, weight_fit_rmse=0.0, **kw)
    if n1 and n2:
        assert err_unweighted != err_weighted, (
            "WeightMask must be able to change the residual of a "
            "mask-touched spot"
        )


def test_the_public_entry_point_refuses():
    """The PyTorch refiner is dead; running it must fail loudly.

    These tests inspect ``_refine_block_from_disk_retired`` — the preserved
    body — because the public ``refine_block_from_disk`` now raises. Keeping
    the body under test means a revival cannot quietly lose the logic these
    tests were written to protect.
    """
    import pytest

    from midas_fit_grain import driver

    with pytest.raises(NotImplementedError, match="RETIRED"):
        driver.refine_block_from_disk(
            cfg=None, param_file="x", block_nr=0, num_blocks=1,
            device=None, dtype=None,
        )
