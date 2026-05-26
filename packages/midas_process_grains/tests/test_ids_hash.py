"""IDsHash.csv reader tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midas_process_grains.io.ids_hash import (
    IDsHash,
    build_ids_hash_from_inputall,
    load_ids_hash,
)


def test_load_ids_hash(tmp_path: Path):
    p = tmp_path / "IDsHash.csv"
    p.write_text(
        "3 1 738333 1.270918\n"
        "4 738333 1702106 1.083843\n"
        "5 1702106 2287190 1.037701\n"
    )
    h = load_ids_hash(p)
    assert isinstance(h, IDsHash)
    np.testing.assert_array_equal(h.ring_nrs, [3, 4, 5])
    np.testing.assert_array_equal(h.id_starts, [1, 738333, 1702106])
    np.testing.assert_allclose(h.d_spacings, [1.270918, 1.083843, 1.037701])


def test_d_for_spot_id_in_range(tmp_path: Path):
    p = tmp_path / "IDsHash.csv"
    p.write_text(
        "3 1 1000 1.27\n"
        "4 1000 2000 1.08\n"
    )
    h = load_ids_hash(p)
    assert h.d_for_spot_id(1) == 1.27
    assert h.d_for_spot_id(999) == 1.27
    assert h.d_for_spot_id(1000) == 1.08
    assert h.d_for_spot_id(1999) == 1.08


def test_ring_for_spot_id_out_of_range(tmp_path: Path):
    p = tmp_path / "IDsHash.csv"
    p.write_text("3 1 1000 1.27\n")
    h = load_ids_hash(p)
    assert h.ring_for_spot_id(0) == -1
    assert h.ring_for_spot_id(2000) == -1
    assert h.ring_for_spot_id(500) == 3


def test_d_for_spot_ids_vectorised(tmp_path: Path):
    p = tmp_path / "IDsHash.csv"
    p.write_text("3 1 1000 1.27\n4 1000 2000 1.08\n")
    h = load_ids_hash(p)
    sids = np.array([100, 1500, 2500, 0])
    d = h.d_for_spot_ids(sids)
    np.testing.assert_allclose(d, [1.27, 1.08, 0.0, 0.0])


def test_synth_ids_hash_uses_reference_not_observed_d(tmp_path: Path):
    """Regression: ``build_ids_hash_from_inputall`` must take the per-ring d0
    from the STRAIN-FREE reference (hkls.csv), NOT the observed-median 2θ.

    Using the observed median bakes the sample's mean isotropic strain into
    the reference, which cancels the diagonal of every per-spot strain tensor
    (the IDsHash half of the FF c-omp "missing diagonal" bug). Here the
    observed 2θ corresponds to a ~+1000 µε dilated lattice; the synthesized d0
    must still equal the reference hkls value, not the dilated observed one.
    """
    wavelength = 0.2066
    # Reference ring d-spacings (hkls.csv).
    ref_d = {1: 2.078461, 2: 1.800000}
    (tmp_path / "hkls.csv").write_text(
        "h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius\n"
        f"1 1 1 {ref_d[1]} 1 0 0 0 0 0 0\n"
        f"2 0 0 {ref_d[2]} 2 0 0 0 0 0 0\n"
    )
    # InputAll with OBSERVED 2θ from a +1000 µε dilated lattice (d_obs larger
    # ⇒ 2θ smaller than the reference). Cols: SpotID(4) RingNumber(5) Ttheta(7).
    eps = 1.0e-3
    rows = []
    sid = 1
    for ring, d0 in ref_d.items():
        d_obs = d0 * (1.0 + eps)
        tth_obs = 2.0 * np.degrees(np.arcsin(wavelength / (2.0 * d_obs)))
        for _ in range(5):
            cols = [0.0] * 8
            cols[4] = float(sid)
            cols[5] = float(ring)
            cols[7] = tth_obs
            rows.append(cols)
            sid += 1
    header = " ".join(f"c{i}" for i in range(8))
    body = "\n".join(" ".join(f"{v:.8f}" for v in r) for r in rows)
    (tmp_path / "InputAllExtraInfoFittingAll.csv").write_text(header + "\n" + body + "\n")

    h = build_ids_hash_from_inputall(tmp_path, wavelength, [1, 2], write=False)
    assert h is not None
    # d0 must match the reference, NOT the +1000 µε observed-median value.
    np.testing.assert_allclose(h.d_spacings, [ref_d[1], ref_d[2]], rtol=1e-6)
    observed_d = {r: ref_d[r] * (1.0 + eps) for r in ref_d}
    for got, ring in zip(h.d_spacings, h.ring_nrs):
        assert abs(got - observed_d[int(ring)]) > 1e-4  # provably not observed
