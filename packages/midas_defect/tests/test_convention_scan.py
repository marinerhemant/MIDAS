"""The omega-sign scan's verdict needs an absolute floor, not only a ratio (held-out 2604 p=184, 2026-09-10)."""
from midas_defect.indexing import ConventionScan


def _scan(a, b, **kw):
    return ConventionScan(best_omega_sign=+1 if a >= b else -1,
                          table=[{"omega_sign": +1, "n_assigned": a}, {"omega_sign": -1, "n_assigned": b}], **kw)


def test_a_ratio_between_tiny_counts_is_not_decisive():
    assert not _scan(4, 1).decisive            # gasket-seeded scan that chose the wrong sign
    assert not _scan(7, 0).decisive
    assert "NOT DECISIVE" in str(_scan(4, 1))


def test_the_delivered_positions_stay_decisive():
    assert _scan(58, 1).decisive               # 2604
    assert _scan(11, 2).decisive               # S5
    assert not _scan(12, 7).decisive           # ratio rule still applies


def test_the_floor_is_a_parameter():
    assert _scan(4, 1, min_assigned=3).decisive
