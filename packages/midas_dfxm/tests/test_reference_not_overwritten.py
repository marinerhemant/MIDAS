"""reference_deg must be the median centre it was built from, not the repeat-excess trimmed mean.

Before 2026-09-27 the repeat-excess block (scans with repeat halves and >= 50 lit px) reused the
name `ref`, so RockingMaps.reference_deg was saved as a variance ratio (S173: 5.83 'deg' on a scan
at 12.74-12.78 deg; S996: 1.08 on 8.29-8.34). `value` was unaffected (computed before)."""
import numpy as np
from midas_dfxm.rocking import RockingScan, reduce_rocking


def test_reference_is_the_median_centre_with_repeats():
    rng = np.random.default_rng(0)
    th = 15.0 + np.arange(31) * 0.001
    lam = 100 + 500 * np.exp(-0.5 * ((th - th[15]) / 0.0012) ** 2)[:, None, None] * np.ones((1, 32, 32))
    A = rng.poisson(lam).astype(np.float32); B = rng.poisson(lam).astype(np.float32)
    scan = RockingScan.from_arrays(0.5 * (A + B), {"th": th}, halves=(A, B), n_repeats=2)
    m = reduce_rocking(scan)
    assert m.repeat_excess is not None                      # the block that used to overwrite ran
    good = m.lit & ~m.truncated
    assert th[0] <= m.reference_deg <= th[-1]
    assert abs(m.reference_deg - np.nanmedian(m.centre_deg[good])) < 1e-9


def test_window_peak_drops_a_clean_broad_peak_and_support_keeps_it():
    """The S173 failure on planted data: FWHM 7 mdeg on a 31-point, 30 mdeg scan. window="peak"
    opens +-2 x (points above half max) = the whole scan, so no baseline frames, SNR NaN, 0 lit."""
    from midas_dfxm.support import reduce_support
    rng = np.random.default_rng(1)
    th = 15.0 + np.arange(31) * 0.001
    lam = 100 + 500 * np.exp(-0.5 * ((th - th[15]) / 0.003) ** 2)[:, None, None] * np.ones((1, 32, 32))
    A = rng.poisson(lam).astype(np.float32); B = rng.poisson(lam).astype(np.float32)
    scan = RockingScan.from_arrays(0.5 * (A + B), {"th": th}, halves=(A, B), n_repeats=2)
    assert reduce_rocking(scan).lit.mean() == 0.0            # documents the limitation
    sm = reduce_support(scan)
    assert sm.lit.mean() > 0.95 and sm.truncated[sm.lit].mean() < 0.02
    assert abs(np.nanmedian(sm.centre_deg[sm.lit, 0]) - th[15]) < 2e-4
