"""Real-data ingest (io.py) — geometry parsing + GrainPatchData assembly.

These tests exercise the contract without any external dataset: the
synthetic plant stands in for real data, and a written paramstest/voxel_grid/
Results scaffold stands in for a real run directory. Geometry correctness on
*actual* data is validated separately against observed spots on chiltepin.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from midas_pf_odf import fit_grain_peakshape
from midas_pf_odf.simulate import plant_single_grain, simulate_grain_patches
import pytest

from midas_pf_odf.io import (
    parse_paramstest,
    geometry_from_paramstest,
    PFGrainDataset,
    assemble_grain_patch_data,
    crop_patches_from_frames,
    zarr_frame_reader,
    load_pf_grain,
)
from tests.conftest import (
    make_fcc_hkls,
    small_scan_config,
    build_model,
    standard_pf_geometry,
)


_PARAMSTEST = """\
LatticeParameter 2.952870 2.952870 4.686200 90.000000 90.000000 120.000000;
Wavelength 0.172979;
px 172.000000;
BeamSize 1.000000;
RingNumbers 1;
RingNumbers 3;
RingToIndex 1;
LsdFit 751533.604946
YBCFit 694.514369
ZBCFit 873.996545
OmeBinSize 0.100000;
OmegaRange -180.000000 180.000000;
Wedge 0.000000;
tyFit 0.186506
tzFit 0.484426
"""


def test_parse_paramstest_and_geometry(tmp_path):
    p = tmp_path / "paramstest.txt"
    p.write_text(_PARAMSTEST)
    params = parse_paramstest(p)
    assert params["LsdFit"][0][0].startswith("751533")
    assert len(params["RingNumbers"]) == 2          # repeated key accumulates

    # Frame step comes from the acquisition (NrFilesPerSweep), NOT OmeBinSize.
    geom = geometry_from_paramstest(
        params, n_pixels_y=1475, n_pixels_z=1679, n_frames=1440,
    )
    assert abs(geom.Lsd - 751533.604946) < 1e-3
    assert abs(geom.y_BC - 694.514369) < 1e-6
    assert abs(geom.z_BC - 873.996545) < 1e-6
    assert abs(geom.px - 172.0) < 1e-9
    assert abs(geom.omega_start - (-180.0)) < 1e-9
    assert geom.n_frames == 1440                      # from NrFilesPerSweep
    assert abs(geom.omega_step - 0.25) < 1e-9         # 360 / 1440, NOT OmeBinSize 0.1
    assert abs(geom.ty - 0.186506) < 1e-6
    assert abs(geom.tz - 0.484426) < 1e-6
    assert geom.wedge == 0.0
    assert geom.flip_y is True

    # Fallback to OmeBinSize warns (and is the wrong step for frame cropping).
    import pytest
    with pytest.warns(UserWarning):
        g2 = geometry_from_paramstest(params, n_pixels_y=1475, n_pixels_z=1679)
    assert g2.n_frames == 3600                         # 360 / 0.1 (OmeBinSize fallback)


_PARAMSTEST_MULTIRANGE = _PARAMSTEST.replace(
    "OmegaRange -180.000000 180.000000;",
    "OmegaRange -180.000000 -106.000000;\n"
    "OmegaRange -76.000000 74.000000;\n"
    "OmegaRange 105.000000 180.000000;",
)


def test_multirange_without_step_raises(tmp_path):
    """P0-3: shadow-gapped multi-OmegaRange + no explicit step must ERROR,
    never infer 74/1440 = 0.0514° from the first span (SOH failure)."""
    import pytest
    p = tmp_path / "paramstest.txt"
    p.write_text(_PARAMSTEST_MULTIRANGE)
    params = parse_paramstest(p)
    with pytest.raises(ValueError, match="OmegaRange"):
        geometry_from_paramstest(params, n_pixels_y=1475, n_pixels_z=1679)


def test_multirange_with_explicit_keys(tmp_path):
    """OmegaStart/OmegaStep keys (midas-transforms >= 0.8.0) win."""
    p = tmp_path / "paramstest.txt"
    p.write_text(_PARAMSTEST_MULTIRANGE + "OmegaStart -180\nOmegaStep 0.25\n")
    params = parse_paramstest(p)
    geom = geometry_from_paramstest(params, n_pixels_y=1475, n_pixels_z=1679)
    assert abs(geom.omega_start - (-180.0)) < 1e-9
    assert abs(geom.omega_step - 0.25) < 1e-9
    assert geom.n_frames == 1440


def test_multirange_with_n_frames_uses_full_span(tmp_path):
    """With n_frames given, the step must come from the FULL acquisition
    span across all ranges (360°), not the first range's 74°."""
    p = tmp_path / "paramstest.txt"
    p.write_text(_PARAMSTEST_MULTIRANGE)
    params = parse_paramstest(p)
    geom = geometry_from_paramstest(
        params, n_pixels_y=1475, n_pixels_z=1679, n_frames=1440)
    assert abs(geom.omega_step - 0.25) < 1e-9
    assert abs(geom.omega_start - (-180.0)) < 1e-9


def test_false_zero_omegastep_key_ignored(tmp_path):
    """A literal ``OmegaStep 0.0`` (E5-class false metadata) must not be
    treated as an explicit step."""
    import pytest
    p = tmp_path / "paramstest.txt"
    p.write_text(_PARAMSTEST_MULTIRANGE + "OmegaStep 0.0\n")
    params = parse_paramstest(p)
    with pytest.raises(ValueError, match="OmegaRange"):
        geometry_from_paramstest(params, n_pixels_y=1475, n_pixels_z=1679)


def _write_layer_scaffold(tmp_path, paramstest_text):
    """paramstest + hkls.csv (rings 1 + 3) + positions.csv for model builds."""
    layer = tmp_path / "LayerNr_1"
    layer.mkdir(parents=True, exist_ok=True)
    (layer / "paramstest.txt").write_text(paramstest_text)
    hk = "h k l D-spacing RingNr g1 g2 g3 Theta(deg) 2Theta(deg) Radius\n"
    for h, k, l, d, rn in [(1, 0, 0, 2.55, 1), (-1, 0, 0, 2.55, 1),
                           (1, 0, 1, 2.34, 3), (-1, 0, -1, 2.34, 3)]:
        th = math.degrees(math.asin(0.172979 / (2 * d)))
        hk += f"{h} {k} {l} {d} {rn} 0 0 0 {th:.4f} {2*th:.4f} 60000.0\n"
    (layer / "hkls.csv").write_text(hk)
    (layer / "positions.csv").write_text("-2.0\n0.0\n2.0\n")
    return layer


def test_ring_numbers_override(tmp_path):
    """P1-5: ``ring_numbers`` kwarg overrides paramstest RingNumbers
    (indexing rings are generally not the optimal strain rings)."""
    from midas_pf_odf.io import build_model_from_paramstest
    layer = _write_layer_scaffold(tmp_path, _PARAMSTEST)

    _, rn_default = build_model_from_paramstest(
        layer, n_pixels_y=64, n_pixels_z=64, n_frames=1440)
    assert set(int(r) for r in rn_default) == {1, 3}   # paramstest behaviour

    _, rn_over = build_model_from_paramstest(
        layer, n_pixels_y=64, n_pixels_z=64, n_frames=1440,
        ring_numbers=[3])
    assert set(int(r) for r in rn_over) == {3}


def test_distortion_v1_mapping_and_plumbing(tmp_path):
    """P1-4: paramstest p0..p14 are v1-ordered — p3 is phi4 (a PHASE in
    degrees). The v2 vector must place it in the phi4 slot, and the
    forward model must receive the distortion only when opted in."""
    from midas_distortion import P_COEF_NAMES, V1_TO_V2_DISTORTION
    from midas_pf_odf.io import (
        build_model_from_paramstest, distortion_from_paramstest,
    )

    pt = _PARAMSTEST + "MaxRingRad 200000.0;\np0 -0.64\np3 35.5\n"
    layer = _write_layer_scaffold(tmp_path, pt)
    params = parse_paramstest(layer / "paramstest.txt")

    p_v2, rho_d = distortion_from_paramstest(params)
    assert rho_d == 200000.0
    assert p_v2[P_COEF_NAMES.index(V1_TO_V2_DISTORTION[3])] == 35.5   # phi4
    assert p_v2[P_COEF_NAMES.index(V1_TO_V2_DISTORTION[0])] == -0.64  # a2
    # The naive-copy failure mode: 35.5 must NOT land in v2 slot 3.
    assert P_COEF_NAMES[3] != V1_TO_V2_DISTORTION[3] or p_v2[3] == 35.5

    # Opt-in plumbing: default build carries no distortion...
    model_off, _ = build_model_from_paramstest(
        layer, n_pixels_y=64, n_pixels_z=64, n_frames=1440)
    assert not model_off.apply_distortion
    # ...opted-in build carries the mapped coefficients + rho_d.
    model_on, _ = build_model_from_paramstest(
        layer, n_pixels_y=64, n_pixels_z=64, n_frames=1440,
        apply_distortion=True)
    assert model_on.apply_distortion
    import numpy as np
    np.testing.assert_allclose(
        model_on.p_distortion.detach().cpu().numpy(), p_v2)
    assert model_on.rho_d == 200000.0


def test_apply_distortion_without_coeffs_warns(tmp_path):
    import pytest
    from midas_pf_odf.io import build_model_from_paramstest
    layer = _write_layer_scaffold(tmp_path, _PARAMSTEST)
    with pytest.warns(UserWarning, match="no distortion"):
        model, _ = build_model_from_paramstest(
            layer, n_pixels_y=64, n_pixels_z=64, n_frames=1440,
            apply_distortion=True)
    assert not model.apply_distortion


def _tiny_model_and_plant():
    G_cart, thetas, hkls_int = make_fcc_hkls()
    scan = small_scan_config(sample_size_um=12.0, n_scans=7, beam_size_um=4.0)
    model = build_model(scan, hkls_int, G_cart, thetas)
    plant = plant_single_grain(
        grid_shape=(4, 4), voxel_size_um=2.0,
        lattice=(3.61, 3.61, 3.61, 90.0, 90.0, 90.0),
        eps_gradient_voigt=0, eps_gradient_amp=2e-3, eps_gradient_dir="y",
    )
    return model, plant, scan


def _dataset_from_plant(model, plant, grain_id=1):
    G = plant.n_voxels
    Gx, Gy = plant.grid_shape
    ij = np.stack(np.unravel_index(np.arange(G), (Gx, Gy)), axis=1)
    return PFGrainDataset(
        grain_id=grain_id,
        voxel_idx=np.arange(G, dtype=np.int64),
        voxel_pos=plant.voxel_pos.clone(),
        R_init=plant.R_voxel.clone(),
        eps_init=torch.zeros(G, 6, dtype=plant.R_voxel.dtype),
        lattice_init=plant.lattice.clone(),
        model=model,
        grid_shape=plant.grid_shape,
        grid_ij=ij,
        metadata={},
    )


def test_assemble_roundtrip_matches_simulate_layout():
    """io assembly must reproduce the simulate path's spot layout + anchors
    when fed the simulate path's own measured patches."""
    model, plant, scan = _tiny_model_and_plant()
    sim = simulate_grain_patches(plant, model, patch_F=5, patch_P=15)

    ds = _dataset_from_plant(model, plant)
    data = assemble_grain_patch_data(
        ds, measured_patches=sim.measured_patches,
        patch_F=5, patch_P=15,
    )
    # Same S (=2M), same Σ.
    assert data.measured_patches.shape == sim.measured_patches.shape
    assert data.spot_observed.shape == sim.spot_observed.shape
    # Anchors agree to sub-pixel. They are NOT bit-identical: the io path
    # anchors each spot at the robust nanMEDIAN over its valid voxels
    # (immune to a mis-canonicalized outlier voxel — see _forward_anchors),
    # whereas the simulate path uses the valid-weighted MEAN. Over a strain
    # gradient these differ by ~4e-3 px, far below the integer patch crop.
    assert torch.allclose(data.anchor_y, sim.anchor_y, atol=0.05, equal_nan=True)
    assert torch.allclose(data.anchor_z, sim.anchor_z, atol=0.05, equal_nan=True)
    # Observed mask agrees.
    assert torch.equal(data.spot_observed, sim.spot_observed)


def test_assembled_data_is_consumable_by_inversion():
    """The assembled GrainPatchData drives fit_grain_peakshape end-to-end."""
    model, plant, scan = _tiny_model_and_plant()
    sim = simulate_grain_patches(plant, model, patch_F=5, patch_P=15)
    ds = _dataset_from_plant(model, plant)
    data = assemble_grain_patch_data(ds, measured_patches=sim.measured_patches,
                                     patch_F=5, patch_P=15)
    res = fit_grain_peakshape(
        data, model,
        voxel_pos=ds.voxel_pos, R_init=ds.R_init,
        eps_init=ds.eps_init, lattice_init=ds.lattice_init,
        optimizer="adam", inner_steps=3, lr_eps=1e-3,
    )
    assert res.eps_fit.shape == (ds.n_voxels, 6)
    assert len(res.losses) == 3
    assert np.isfinite(res.losses[-1])


def test_crop_patches_from_frames_picks_up_intensity():
    """crop_patches_from_frames windows the right region of synthetic frames."""
    n_y, n_z, n_scans, n_frames = 40, 40, 3, 10
    # one observed spot anchored at (y=20, z=15, f=5)
    anchor_y = torch.tensor([20.0])
    anchor_z = torch.tensor([15.0])
    anchor_f = torch.tensor([5.0])
    spot_observed = torch.tensor([True])
    frames = np.zeros((n_scans, n_frames, n_y, n_z), dtype=np.float64)
    frames[1, 5, 20, 15] = 7.0          # bright pixel at the anchor, scan 1, frame 5

    def reader(scan_idx, frame_idx):
        return frames[scan_idx, frame_idx]

    out = crop_patches_from_frames(
        reader, anchor_y, anchor_z, anchor_f, n_scans, spot_observed,
        patch_F=3, patch_P=5, n_pixels_y=n_y, n_pixels_z=n_z,
    )
    assert out.shape == (1, n_scans, 3, 5, 5)
    # center of patch (F=1 of 3, P=2 of 5) at scan 1 holds the bright pixel.
    assert out[0, 1, 1, 2, 2].item() == 7.0
    # other scans are zero.
    assert out[0, 0].abs().sum().item() == 0.0


def test_load_pf_grain_from_scaffold(tmp_path):
    """load_pf_grain reads voxel_grid.csv + Results CSVs for one grain."""
    model, plant, scan = _tiny_model_and_plant()
    G = plant.n_voxels
    layer = tmp_path / "LayerNr_1"
    (layer / "Output").mkdir(parents=True)
    (layer / "Results").mkdir(parents=True)

    # voxel_grid.csv — all G voxels are grain 1.
    rows = ["voxel_idx x_um y_um z_um grain_id"]
    for v in range(G):
        x, y, z = plant.voxel_pos[v].tolist()
        rows.append(f"{v} {x:.4f} {y:.4f} {z:.4f} 1")
    (layer / "Output" / "voxel_grid.csv").write_text("\n".join(rows) + "\n")

    # Results CSVs — OM cols 1-9, lattice cols 15-20.
    hdr = " ".join(f"c{i}" for i in range(40))
    for v in range(G):
        om = plant.R_voxel[v].reshape(-1).tolist()
        row = [0.0] * 40
        row[1:10] = om
        row[15:21] = list(plant.lattice.tolist())
        (layer / "Results" / f"Result_OrientPos_voxel_{v}.csv").write_text(
            hdr + "\n" + " ".join(f"{x:.9f}" for x in row) + "\n"
        )

    # space_group passed explicitly: the scaffold has no paramstest.txt, and
    # OM canonicalization (default on) reads SpaceGroup from there on real runs.
    ds = load_pf_grain(
        layer, grain_id=1, n_pixels_y=40, n_pixels_z=40, model=model,
        space_group=225,
    )
    assert ds.n_voxels == G
    assert ds.R_init.shape == (G, 3, 3)
    assert torch.allclose(ds.R_init, plant.R_voxel, atol=1e-6)
    assert torch.allclose(ds.lattice_init, plant.lattice, atol=1e-6)
    assert torch.allclose(ds.eps_init, torch.zeros(G, 6, dtype=ds.eps_init.dtype))


def test_subtract_background_removes_pedestal():
    """subtract_background strips a flat pedestal from observed patches while
    preserving the spot, and is a no-op-ish for the pedestal when off."""
    model, plant, scan = _tiny_model_and_plant()
    sim = simulate_grain_patches(plant, model, patch_F=5, patch_P=15)
    ds = _dataset_from_plant(model, plant)
    PED = 500.0
    raw = sim.measured_patches + PED               # flat pedestal under everything

    on = assemble_grain_patch_data(ds, measured_patches=raw, patch_F=5, patch_P=15,
                                   subtract_background=True, background_border=3)
    off = assemble_grain_patch_data(ds, measured_patches=raw, patch_F=5, patch_P=15,
                                    subtract_background=False)

    obs = on.spot_observed.reshape(-1).bool()
    assert obs.any()
    # central-frame border of observed spots: ~0 after subtraction, ~PED before.
    def border_median(data):
        cf = data.measured_patches[obs][:, :, 2]   # (n_obs, Σ, 15, 15)
        ring = torch.cat([cf[:, :, :3, :].reshape(*cf.shape[:2], -1),
                          cf[:, :, -3:, :].reshape(*cf.shape[:2], -1)], dim=-1)
        return ring.median().item()
    assert border_median(on) < 5.0
    assert border_median(off) > PED - 1.0
    # spot survives: peak of an observed patch is still positive after subtraction.
    assert on.measured_patches[obs].amax() > 0.0
    # non-negative everywhere (clamped).
    assert on.measured_patches.min().item() >= 0.0


def test_zarr_frame_reader_reads_frames(tmp_path):
    """zarr_frame_reader indexes per-scan zarr stores as frame_reader(σ, f)."""
    zarr = pytest.importorskip("zarr")
    frames0 = np.arange(4 * 5 * 6).reshape(4, 5, 6).astype("u2")
    frames1 = (frames0 + 100).astype("u2")
    paths = []
    for k, fr in enumerate((frames0, frames1)):
        p = tmp_path / f"scan{k}.zip"
        store = zarr.ZipStore(str(p), mode="w")       # true ZipStore, like *.MIDAS.zip
        z = zarr.open(store=store, mode="w")
        d = z.create_dataset("exchange/data", shape=fr.shape, chunks=(1, 5, 6),
                             dtype="u2")
        d[:] = fr
        store.close()
        paths.append(str(p))

    reader = zarr_frame_reader(paths)
    got = reader(0, 2)
    assert got.shape == (5, 6)
    np.testing.assert_array_equal(got, frames0[2])
    np.testing.assert_array_equal(reader(1, 3), frames1[3])   # second store + cache
