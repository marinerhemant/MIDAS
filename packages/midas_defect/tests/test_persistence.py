"""midas_defect.persistence: HDF5 save/load round-trips exactly, and
reduce_one_position(ingested=...) gives IDENTICAL results to a live ingest --
the one thing a caching layer must never get wrong."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from midas_defect.geometry import Geometry
from midas_defect.synthetic import synthetic_position_frames
from midas_defect.raster import reduce_one_position, _ingest_position, IngestBundle
from midas_defect.persistence import save_position_hdf5, load_ingest_hdf5, load_position_result_hdf5

A, B, C = 4.0, 3.9, 10.0
SG = 123
SIGMA_RTN = (0.02, 0.02, 0.02)


def _geom(n_pix=768):
    return Geometry(lsd_um=120_000.0, bcy_px=n_pix / 2, bcz_px=n_pix / 2, px_um=172.0,
                    wavelength_A=0.4246, n_pix_y=n_pix, n_pix_z=n_pix,
                    omega_first_deg=-19.5, omega_step_deg=1.0, n_frames=40, label="synthetic")


def _make_bundle_and_result(geom, **synth_kw):
    U0 = Rotation.from_euler("zyx", [15.0, 25.0, -8.0], degrees=True).as_matrix()
    frames, truth = synthetic_position_frames([U0], a=A, b=B, c=C, space_group_number=SG,
                                               geom=geom, hmax=8, kmax=8, lmax=14, seed=0, **synth_kw)
    spots, qlab, omega_deg, mask, sub, counts = _ingest_position(frames, geom)
    bundle = IngestBundle(spots=spots, qlab=qlab, omega_deg=omega_deg, mask=mask,
                          sub=sub, ingest_counts=counts)
    res = reduce_one_position(frames, geom, a=A, c=C, space_group_number=SG, sigma_rtn=SIGMA_RTN,
                              point=7, n_bootstrap=0, ingested=bundle)
    return frames, bundle, res


def test_ingest_bundle_round_trips_exactly(tmp_path):
    geom = _geom()
    frames, bundle, res = _make_bundle_and_result(geom)
    path = tmp_path / "p7.h5"
    save_position_hdf5(path, bundle=bundle, res=res, save_dense=True)

    reloaded = load_ingest_hdf5(path)
    assert reloaded.spots.equals(bundle.spots)
    assert np.allclose(reloaded.qlab.numpy(), bundle.qlab.numpy())
    assert np.allclose(reloaded.omega_deg, bundle.omega_deg)
    assert np.array_equal(reloaded.mask, bundle.mask)
    assert np.allclose(reloaded.sub, bundle.sub)
    assert reloaded.ingest_counts == bundle.ingest_counts


def test_reduce_one_position_from_cache_matches_live_exactly(tmp_path):
    """The load-bearing correctness guarantee: reduce_one_position fed a
    RELOADED bundle must find the exact same domains as the live run that
    produced it -- not merely a similar count."""
    geom = _geom()
    frames, bundle, res_live = _make_bundle_and_result(geom)
    path = tmp_path / "p7.h5"
    save_position_hdf5(path, bundle=bundle, res=res_live, save_dense=True)

    reloaded = load_ingest_hdf5(path)
    res_cached = reduce_one_position(frames, geom, a=A, c=C, space_group_number=SG,
                                     sigma_rtn=SIGMA_RTN, point=7, n_bootstrap=0, ingested=reloaded)

    assert [d.n for d in res_live.domains.domains] == [d.n for d in res_cached.domains.domains]
    for d1, d2 in zip(res_live.domains.domains, res_cached.domains.domains):
        assert np.allclose(d1.U, d2.U)
        assert np.array_equal(d1.claim, d2.claim)
    assert res_live.quotable == res_cached.quotable


def test_save_dense_false_gives_sparse_only_and_reduce_one_position_refuses_it(tmp_path):
    geom = _geom()
    frames, bundle, res = _make_bundle_and_result(geom)
    path = tmp_path / "p7_sparse.h5"
    save_position_hdf5(path, bundle=bundle, res=res, save_dense=False)

    reloaded = load_ingest_hdf5(path)
    assert reloaded.sub is None
    assert reloaded.spots.equals(bundle.spots)   # the sparse part is still there and correct

    with pytest.raises(ValueError, match="save_dense=False"):
        reduce_one_position(frames, geom, a=A, c=C, space_group_number=SG, sigma_rtn=SIGMA_RTN,
                            point=7, n_bootstrap=0, ingested=reloaded)


def test_save_dense_true_file_is_larger_than_false(tmp_path):
    geom = _geom()
    frames, bundle, res = _make_bundle_and_result(geom)
    p_dense = tmp_path / "dense.h5"
    p_sparse = tmp_path / "sparse.h5"
    save_position_hdf5(p_dense, bundle=bundle, res=res, save_dense=True)
    save_position_hdf5(p_sparse, bundle=bundle, res=res, save_dense=False)
    assert p_dense.stat().st_size > 10 * p_sparse.stat().st_size


def test_position_result_round_trips_with_claim_arrays_restored(tmp_path):
    geom = _geom()
    frames, bundle, res = _make_bundle_and_result(geom)
    path = tmp_path / "p7.h5"
    save_position_hdf5(path, bundle=bundle, res=res, save_dense=True)

    d = load_position_result_hdf5(path)
    assert d["quotable"] == res.quotable
    assert d["point"] == res.point
    # to_dict() itself only keeps a COUNT (n_claim); the full boolean claim mask
    # is what this module adds back on top of it
    assert "n_claim" not in d["domains_full"][0]
    assert np.array_equal(d["domains_full"][0]["claim"], res.domains.domains[0].claim)
    assert np.array_equal(d["domains_full"][0]["frag"], res.domains.domains[0].frag)
