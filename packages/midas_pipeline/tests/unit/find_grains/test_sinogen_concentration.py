"""Concentration filter — drop contaminated sino rows.

The filter is calibrated on 20-ID ``pf_nf709`` set A, where 16 of 958
rows sit below concentration 0.35. These tests pin the behaviour on a
synthetic grain where the ground truth is known by construction.
"""

from __future__ import annotations

import numpy as np

from midas_pipeline.find_grains import (
    apply_concentration_filter,
    sinogram_concentration,
    write_clean_variant,
)


N_SCANS = 51
S_VALS = -25.0 + np.arange(N_SCANS, dtype=np.float64)


def synth_grain(n_hkl=40, sx=6.0, sy=-4.0, width=5.0, stripe_rows=(3, 17, 29)):
    """One grain riding a known sinusoid, with a few smeared rows.

    Clean rows are a Gaussian blob of width ``width`` centred on
    ``s(ω) = sx·sin ω + sy·cos ω``; stripe rows are flat across every
    scan position, which is what a contaminating spot looks like.
    """
    omegas = np.linspace(-180.0, 180.0, n_hkl, endpoint=False)
    w = np.radians(omegas)
    track = sx * np.sin(w) + sy * np.cos(w)
    sino = np.zeros((1, n_hkl, N_SCANS), dtype=np.float64)
    for h in range(n_hkl):
        if h in stripe_rows:
            sino[0, h, :] = 100.0
        else:
            d = S_VALS - track[h]
            sino[0, h, :] = 1000.0 * np.exp(-0.5 * (d / (width / 2.355)) ** 2)
    return sino, omegas.reshape(1, n_hkl), np.array([n_hkl], dtype=np.int32)


def test_concentration_separates_stripes_from_clean_rows():
    stripes = (3, 17, 29)
    sino, omegas, nr_hkls = synth_grain(stripe_rows=stripes)
    conc = sinogram_concentration(sino, omegas, nr_hkls, scan_positions=S_VALS)

    assert conc.shape == (1, 40)
    is_stripe = np.zeros(40, dtype=bool)
    is_stripe[list(stripes)] = True
    # Clean rows put nearly all of their intensity on the track;
    # flat rows put only the band width / scan width fraction there.
    assert conc[0, ~is_stripe].min() > 0.8
    assert conc[0, is_stripe].max() < 0.35


def test_filter_zeroes_only_the_stripe_rows():
    stripes = (3, 17, 29)
    sino, omegas, nr_hkls = synth_grain(stripe_rows=stripes)
    conc = sinogram_concentration(sino, omegas, nr_hkls, scan_positions=S_VALS)
    clean, dropped = apply_concentration_filter(sino, conc, 0.35)

    assert dropped.sum() == len(stripes)
    assert list(np.flatnonzero(dropped[0])) == list(stripes)
    assert np.all(clean[0, list(stripes), :] == 0.0)
    # Everything else is bit-identical to the input.
    keep = np.ones(40, dtype=bool)
    keep[list(stripes)] = False
    np.testing.assert_array_equal(clean[0, keep], sino[0, keep])
    # And the input was not mutated.
    assert sino[0, stripes[0], 0] == 100.0


def test_filter_recovers_the_true_grain_position():
    """The point of the filter: stripes bias the sinusoid fit."""
    sx, sy = 6.0, -4.0
    sino, omegas, nr_hkls = synth_grain(sx=sx, sy=sy)
    conc = sinogram_concentration(sino, omegas, nr_hkls, scan_positions=S_VALS)
    _, dropped = apply_concentration_filter(sino, conc, 0.35)

    inten = np.clip(sino[0], 0.0, None)
    tot = inten.sum(axis=1)
    cen = (inten * S_VALS).sum(axis=1) / tot
    w = np.radians(omegas[0])
    design = np.column_stack([np.sin(w), np.cos(w)])

    ok = tot > 0
    fit_all, *_ = np.linalg.lstsq(design[ok], cen[ok], rcond=None)
    keep = ok & ~dropped[0]
    fit_kept, *_ = np.linalg.lstsq(design[keep], cen[keep], rcond=None)

    err_all = np.hypot(*(fit_all - np.array([sx, sy])))
    err_kept = np.hypot(*(fit_kept - np.array([sx, sy])))
    assert err_kept < err_all
    assert err_kept < 0.5           # micrometres


def test_nan_rows_are_never_dropped():
    """Unpopulated rows have NaN concentration and must survive."""
    sino, omegas, nr_hkls = synth_grain(stripe_rows=())
    sino[0, 5, :] = 0.0             # empty row
    conc = sinogram_concentration(sino, omegas, nr_hkls, scan_positions=S_VALS)
    assert np.isnan(conc[0, 5])
    _, dropped = apply_concentration_filter(sino, conc, 0.35)
    assert not dropped[0, 5]


def test_too_few_rows_to_fit_yields_all_nan():
    sino, omegas, nr_hkls = synth_grain(n_hkl=4, stripe_rows=())
    conc = sinogram_concentration(sino, omegas, nr_hkls, scan_positions=S_VALS)
    assert np.all(np.isnan(conc))


def test_scan_positions_offset_is_removed():
    """An absolute (uncentred) position axis must give the same answer."""
    sino, omegas, nr_hkls = synth_grain()
    a = sinogram_concentration(sino, omegas, nr_hkls, scan_positions=S_VALS)
    b = sinogram_concentration(sino, omegas, nr_hkls,
                               scan_positions=S_VALS + 1234.5)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def test_default_axis_is_centred_bins():
    """No scan_positions ⇒ centred bin indices (1 unit per step)."""
    sino, omegas, nr_hkls = synth_grain()
    a = sinogram_concentration(sino, omegas, nr_hkls, scan_positions=S_VALS)
    b = sinogram_concentration(sino, omegas, nr_hkls)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def test_write_clean_variant_emits_both_files(tmp_path):
    sino, omegas, nr_hkls = synth_grain()
    clean_path, conc_path = write_clean_variant(
        tmp_path, sino, omegas, nr_hkls, conc_threshold=0.35,
        scan_positions=S_VALS,
    )
    assert clean_path.endswith(f"sinos_clean_1_40_{N_SCANS}.bin")
    assert conc_path.endswith("sinoConc_1_40.bin")
    clean = np.fromfile(clean_path, dtype=np.float64).reshape(1, 40, N_SCANS)
    conc = np.fromfile(conc_path, dtype=np.float64).reshape(1, 40)
    assert clean.shape == sino.shape
    assert np.isfinite(conc).all()
    assert (clean.sum(axis=2)[0] == 0).sum() == 3


def test_threshold_zero_emits_nothing(tmp_path):
    """Default config must leave the emitted file set unchanged."""
    from midas_pipeline.find_grains import generate_sinograms_tolerance
    from midas_pipeline.find_grains import SpotData, SpotList

    spots = [SpotData(omega=om, eta=10.0 + i * 0.05, ring_nr=1,
                      merged_id=i + 1, scan_nr=0, grain_nr=0, spot_nr=i)
             for i, om in enumerate((0.0, 1.0, 2.0, 3.0, 4.0, 5.0))]
    sl = SpotList(spots=spots, max_n_hkls=len(spots))
    rows = [[0.0, 0.0, sd.omega, 100.0 + sc, sd.merged_id + sc * 100,
             sd.ring_nr, sd.eta, 5.0, 1.0, sc]
            for sc in range(5) for sd in spots]
    out = generate_sinograms_tolerance(
        spot_list=sl, n_unique=1, all_spots=np.asarray(rows, dtype=np.float64),
        n_scans=5, tol_ome=0.5, tol_eta=0.1, output_dir=tmp_path,
    )
    assert "clean" not in out.sino_paths
    assert not list(tmp_path.glob("sinos_clean_*.bin"))
    assert not list(tmp_path.glob("sinoConc_*.bin"))


# ---------------------------------------------------------------------------
# Occupancy — the signal the concentration filter is blind to.
# ---------------------------------------------------------------------------


def wide_grain(n_hkl=40, width=90.0):
    """A grain far wider than the scanned field: every row is fully lit."""
    omegas = np.linspace(-180.0, 180.0, n_hkl, endpoint=False)
    w = np.radians(omegas)
    track = 3.0 * np.sin(w) - 2.0 * np.cos(w)
    sino = np.zeros((1, n_hkl, N_SCANS), dtype=np.float64)
    for h in range(n_hkl):
        d = S_VALS - track[h]
        sino[0, h, :] = 1000.0 * np.exp(-0.5 * (d / (width / 2.355)) ** 2)
    return sino, omegas.reshape(1, n_hkl), np.array([n_hkl], dtype=np.int32)


def test_occupancy_separates_contained_from_field_filling():
    from midas_pipeline.find_grains import sinogram_occupancy
    small, om_s, nr_s = synth_grain(stripe_rows=())          # 5 um blob
    big, om_b, nr_b = wide_grain()                           # 90 um blob
    o_small = sinogram_occupancy(small, nr_s)[0]
    o_big = sinogram_occupancy(big, nr_b)[0]
    assert o_small < 0.3
    assert o_big > 0.9
    assert o_big > o_small


def test_concentration_is_blind_to_a_field_filling_grain():
    """The band adapts to the grain's own width, so it scores as clean.

    This is the whole reason occupancy exists as a separate diagnostic.
    """
    from midas_pipeline.find_grains import sinogram_occupancy
    big, om, nr = wide_grain()
    conc = sinogram_concentration(big, om, nr, scan_positions=S_VALS)
    assert np.nanmin(conc) > 0.35          # no threshold would catch it
    assert sinogram_occupancy(big, nr)[0] > 0.9   # occupancy does


def test_write_occupancy_roundtrips(tmp_path):
    from midas_pipeline.find_grains import write_occupancy
    sino, om, nr = synth_grain(stripe_rows=())
    path = write_occupancy(tmp_path, sino, nr)
    assert path.endswith("sinoOccupancy_1.bin")
    occ = np.fromfile(path, dtype=np.float64)
    assert occ.shape == (1,)
    assert 0.0 < occ[0] < 1.0


def test_occupancy_nan_for_empty_grain():
    from midas_pipeline.find_grains import sinogram_occupancy
    sino, om, nr = synth_grain(stripe_rows=())
    sino[:] = 0.0
    assert np.isnan(sinogram_occupancy(sino, nr)[0])
