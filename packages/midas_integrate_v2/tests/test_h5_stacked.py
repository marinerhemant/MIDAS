"""The stacked HDF5 writer must reproduce the streaming-GPU layout.

Reference: ``integrator_stream_process_h5_stacked.py`` (the copland-only
post-processor for ``IntegratorFitPeaksGPUStream``).
"""
import numpy as np
import pytest
import torch

h5py = pytest.importorskip("h5py")

from midas_integrate_v2.io.h5_stacked import StackedH5Writer, write_stacked_h5
from midas_integrate_v2.spec import IntegrationSpec


@pytest.fixture
def spec():
    return IntegrationSpec(
        Lsd=torch.tensor(1_000_000.0),
        BC_y=torch.tensor(1024.0), BC_z=torch.tensor(1024.0),
        Wavelength=torch.tensor(0.413263),
        NrPixelsY=2048, NrPixelsZ=2048, pxY=200.0, pxZ=200.0,
        RMin=10.0, RMax=90.0, RBinSize=10.0,           # -> 8 R bins
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=60.0,  # -> 6 eta bins
    )


def _cake(spec, seed):
    rng = np.random.default_rng(seed)
    return rng.random((spec.n_eta_bins, spec.n_r_bins)) * 100.0


def test_stacked_layout_round_trip(tmp_path, spec):
    cakes = [_cake(spec, i) for i in range(6)]
    lineouts = [c.mean(axis=0) for c in cakes]
    names = [f"scan_{i:03d}" for i in range(6)]
    area = np.ones((spec.n_eta_bins, spec.n_r_bins))
    out = tmp_path / "t.h5"
    write_stacked_h5(out, cakes, spec=spec, n_frames=6, bin_area=area,
                     frame_names=names, lineouts=lineouts,
                     omegas=[i * 0.5 for i in range(6)],
                     omega_sum_frames=2)

    n_r, n_eta = spec.n_r_bins, spec.n_eta_bins
    with h5py.File(out, "r") as f:
        # Consolidated arrays, not one dataset per frame.
        assert f["OmegaSumFrame"].shape == (3, n_r, n_eta)
        assert f["lineouts"].shape == (6, n_r)
        assert f["frame_names"].shape == (6,)
        assert set(f["geometry_maps"].keys()) == {
            "R_map", "TTh_map", "Eta_map", "Area_map", "Q_map"}
        for k in f["geometry_maps"]:
            assert f["geometry_maps"][k].shape == (n_r, n_eta)

        # Groups of 2, transposed to (n_r, n_eta).
        assert np.allclose(f["OmegaSumFrame"][0], (cakes[0] + cakes[1]).T)
        assert np.allclose(f["OmegaSumFrame"][2], (cakes[4] + cakes[5]).T)

        # Lineouts are intensity only; R lives in the geometry map.
        assert np.allclose(f["lineouts"][3], lineouts[3])
        assert np.allclose(f["r_axis_px"][:], f["geometry_maps"]["R_map"][:, 0])

        got = [n.decode() if isinstance(n, bytes) else n
               for n in f["frame_names"][:]]
        assert got == names
        assert np.allclose(f["Omegas"][:], [i * 0.5 for i in range(6)])
        assert f.attrs["num_frames"] == 6


def test_cake_transposed_to_c_convention(tmp_path, spec):
    cake = _cake(spec, 3)
    out = tmp_path / "t.h5"
    write_stacked_h5(out, [cake], spec=spec, n_frames=1,
                     bin_area=np.ones_like(cake), omega_sum_frames=-1)
    with h5py.File(out, "r") as f:
        assert f["OmegaSumFrame"].shape == (1, spec.n_r_bins, spec.n_eta_bins)
        assert np.allclose(f["OmegaSumFrame"][0], cake.T)


def test_geometry_maps_match_reta_map_rows(tmp_path, spec):
    from midas_integrate_v2.io.zarr_gsas import reta_map
    area = np.arange(spec.n_eta_bins * spec.n_r_bins, dtype=float).reshape(
        spec.n_eta_bins, spec.n_r_bins)
    m = reta_map(spec, area)
    out = tmp_path / "t.h5"
    write_stacked_h5(out, [_cake(spec, 0)], spec=spec, n_frames=1,
                     bin_area=area)
    with h5py.File(out, "r") as f:
        g = f["geometry_maps"]
        for row, name in enumerate(
                ["R_map", "TTh_map", "Eta_map", "Area_map", "Q_map"]):
            assert np.allclose(g[name][:], m[row]), name
        assert g["Area_map"].attrs["units"] == "fractional pixels"
        assert g["Q_map"].attrs["units"] == "inv_Angstrom"


def test_trailing_partial_group(tmp_path, spec):
    cakes = [_cake(spec, i) for i in range(5)]
    out = tmp_path / "t.h5"
    write_stacked_h5(out, cakes, spec=spec, n_frames=5,
                     bin_area=np.ones((spec.n_eta_bins, spec.n_r_bins)),
                     omega_sum_frames=2)
    with h5py.File(out, "r") as f:
        assert f["OmegaSumFrame"].shape[0] == 3          # ceil(5/2)
        assert np.allclose(f["OmegaSumFrame"][2], cakes[4].T)


def test_sum_all_frames_into_one_group(tmp_path, spec):
    cakes = [_cake(spec, i) for i in range(4)]
    out = tmp_path / "t.h5"
    write_stacked_h5(out, cakes, spec=spec, n_frames=4,
                     bin_area=np.ones((spec.n_eta_bins, spec.n_r_bins)),
                     omega_sum_frames=-1)
    with h5py.File(out, "r") as f:
        assert f["OmegaSumFrame"].shape[0] == 1
        assert np.allclose(f["OmegaSumFrame"][0], np.sum(cakes, axis=0).T)


def test_omega_sum_frames_zero_omits_dataset(tmp_path, spec):
    out = tmp_path / "t.h5"
    write_stacked_h5(out, [_cake(spec, 0)], spec=spec, n_frames=1,
                     bin_area=np.ones((spec.n_eta_bins, spec.n_r_bins)),
                     omega_sum_frames=0)
    with h5py.File(out, "r") as f:
        assert "OmegaSumFrame" not in f


def test_fit_block(tmp_path, spec):
    fits = [np.arange(2 * 7, dtype=float).reshape(2, 7) + i for i in range(3)]
    out = tmp_path / "t.h5"
    write_stacked_h5(out, [_cake(spec, i) for i in range(3)], spec=spec,
                     n_frames=3, n_peaks=2, fits=fits,
                     bin_area=np.ones((spec.n_eta_bins, spec.n_r_bins)))
    with h5py.File(out, "r") as f:
        assert f["fit"].shape == (3, 2, 7)
        assert np.allclose(f["fit"][1], fits[1])


def test_short_run_truncates_rather_than_zero_pads(tmp_path, spec):
    """Allocating for 10 but writing 3 must not leave 7 zero-filled frames a
    reader would take for real, empty data."""
    cakes = [_cake(spec, i) for i in range(3)]
    out = tmp_path / "t.h5"
    with StackedH5Writer(out, spec=spec, n_frames=10,
                         bin_area=np.ones((spec.n_eta_bins, spec.n_r_bins)),
                         omega_sum_frames=1) as w:
        for c in cakes:
            w.add_frame(c, lineout=c.mean(axis=0))
    with h5py.File(out, "r") as f:
        assert f["lineouts"].shape == (3, spec.n_r_bins)
        assert f["OmegaSumFrame"].shape[0] == 3
        assert f["frame_names"].shape == (3,)
        assert f.attrs["num_frames"] == 3
        assert f.attrs["n_frames_allocated"] == 10


def test_rejects_more_frames_than_allocated(tmp_path, spec):
    out = tmp_path / "t.h5"
    w = StackedH5Writer(out, spec=spec, n_frames=1,
                        bin_area=np.ones((spec.n_eta_bins, spec.n_r_bins)))
    w.add_frame(_cake(spec, 0))
    with pytest.raises(ValueError, match="more frames added"):
        w.add_frame(_cake(spec, 1))
    w.close()


def test_rejects_wrong_shapes(tmp_path, spec):
    out = tmp_path / "t.h5"
    with StackedH5Writer(out, spec=spec, n_frames=2,
                         bin_area=np.ones((spec.n_eta_bins, spec.n_r_bins)),
                         write_lineouts=True) as w:
        with pytest.raises(ValueError, match="neither"):
            w.add_frame(np.zeros((3, 3)))
        with pytest.raises(ValueError, match="lineout shape"):
            w.add_frame(_cake(spec, 0), lineout=np.zeros(3))


def test_nan_masked_bins_do_not_poison_the_group(tmp_path, spec):
    a = _cake(spec, 1)
    b = _cake(spec, 2)
    b[0, 0] = np.nan
    out = tmp_path / "t.h5"
    write_stacked_h5(out, [a, b], spec=spec, n_frames=2,
                     bin_area=np.ones_like(a), omega_sum_frames=-1)
    with h5py.File(out, "r") as f:
        g = f["OmegaSumFrame"][0]
        assert np.isfinite(g).all()
        assert g[0, 0] == pytest.approx(a.T[0, 0])


def test_torch_tensor_cakes_accepted(tmp_path, spec):
    cake = torch.rand(spec.n_eta_bins, spec.n_r_bins, dtype=torch.float64)
    out = tmp_path / "t.h5"
    write_stacked_h5(out, [cake], spec=spec, n_frames=1,
                     bin_area=torch.ones(spec.n_eta_bins, spec.n_r_bins),
                     omega_sum_frames=-1)
    with h5py.File(out, "r") as f:
        assert np.allclose(f["OmegaSumFrame"][0], cake.numpy().T)


def test_metadata_attribute_is_json(tmp_path, spec):
    import json
    from midas_integrate_v2.io import build_provenance
    md = build_provenance(spec, integrate_mode="polygon")
    out = tmp_path / "t.h5"
    write_stacked_h5(out, [_cake(spec, 0)], spec=spec, n_frames=1,
                     bin_area=np.ones((spec.n_eta_bins, spec.n_r_bins)),
                     metadata=md)
    with h5py.File(out, "r") as f:
        d = json.loads(f.attrs["metadata"])
        assert d["package"] == "midas_integrate_v2"
        assert d["n_r_bins"] == spec.n_r_bins


def test_n_frames_must_be_positive(tmp_path, spec):
    with pytest.raises(ValueError, match="n_frames must be positive"):
        StackedH5Writer(tmp_path / "t.h5", spec=spec, n_frames=0)
