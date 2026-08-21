"""``c_parity`` writes the same signed-residual sidecar the spot-aware path did.

The decomposition arithmetic itself is covered by
``test_residual_decomposition.py``; these tests cover the *wiring* that used to
be missing — c_parity returned without producing
``processgrains_diagnostics.h5`` at all — plus the guarantee that the c_parity
and spot-aware/legacy callers share one implementation of the arithmetic.
"""

import math

import numpy as np
import pytest

from midas_process_grains.compute.c_parity_emit import gather_per_grain_spot_data
from midas_process_grains.compute.c_parity_run import (
    CParityKeptGrain,
    write_residual_diagnostics,
)
from midas_process_grains.compute.residual_decomposition import (
    SPOT_RESIDUAL_COLS,
    build_spot_residual_block,
    build_spot_residual_row,
)
from midas_process_grains.io.ids_hash import IDsHash

LSD_UM = 1_000_000.0
WAVELENGTH_A = 0.1729


# ---------------------------------------------------------------------------
# One implementation, two callers
# ---------------------------------------------------------------------------


def test_block_matches_scalar_row():
    """The vectorised block builder and the scalar row builder agree exactly.

    This is the anti-drift guard: c_parity goes through the block path and the
    spot-aware/legacy pipeline through the scalar wrapper, so a divergence here
    would make the same run report different residuals depending on mode.
    """
    rng = np.random.default_rng(11)
    fb = np.zeros((40, 22))
    fb[:, 1] = rng.uniform(-200_000, 200_000, 40)     # y_obs
    fb[:, 2] = rng.uniform(-200_000, 200_000, 40)     # z_obs
    fb[:, 3] = rng.uniform(-180, 180, 40)             # omega_obs
    fb[:, 7] = fb[:, 1] + rng.normal(0, 50, 40)       # y_exp
    fb[:, 8] = fb[:, 2] + rng.normal(0, 50, 40)       # z_exp
    fb[:, 9] = fb[:, 3] + rng.normal(0, 0.05, 40)     # omega_exp
    fb[:, 19] = rng.uniform(0, 0.5, 40)               # internal angle
    sids = np.arange(1, 41, dtype=np.int64)
    rings = np.full(40, 3, dtype=np.int64)

    block = build_spot_residual_block(7, fb, sids, rings)
    scalar = np.asarray([
        build_spot_residual_row(7, float(s), 3.0, fb[i])
        for i, s in enumerate(sids)
    ])
    np.testing.assert_allclose(block, scalar, rtol=0, atol=0)


def test_block_drops_padding_rows_like_scalar():
    """Zero-padded FitBest slots are rejected identically by both builders."""
    fb = np.zeros((3, 22))
    fb[1, 1], fb[1, 2] = 0.0, 1010.0
    fb[1, 7], fb[1, 8] = 0.0, 1000.0
    # row 0 is all-zero padding; row 2 has an expected position but sits
    # exactly on the beam centre (observed radius 0 -> basis undefined).
    fb[2, 7], fb[2, 8] = 100.0, 100.0

    block = build_spot_residual_block(
        0, fb, np.array([1, 2, 3]), np.array([1, 1, 1]),
    )
    assert block.shape == (1, len(SPOT_RESIDUAL_COLS))
    assert block[0, 1] == 2.0                                  # spot_id
    assert block[0, 6] == pytest.approx(10.0)                  # drad_um
    assert build_spot_residual_row(0, 1, 1, fb[0]) is None
    assert build_spot_residual_row(0, 3, 1, fb[2]) is None


# ---------------------------------------------------------------------------
# Synthetic c_parity run
# ---------------------------------------------------------------------------


def _make_run(bias_by_grain=(5.0, -5.0), n_spots=40, n_seeds=3):
    """FitBest + kept grains with a known radial bias injected per grain.

    Grain ``gi`` gets every one of its spots pushed ``bias_by_grain[gi]`` µm
    radially outward from the prediction, at a spread of azimuths, so the
    per-grain median dRad has an exactly known answer.
    """
    fb = np.zeros((n_seeds, n_spots, 22))
    kept = []
    ring_of_grain = [1, 2]
    for gi, bias in enumerate(bias_by_grain):
        rep = gi                       # seed index this grain represents
        ring = ring_of_grain[gi]
        # SpotIDs are blocked per ring in the MIDAS convention; ring 1 owns
        # 1..1000, ring 2 owns 1001..2000 (see IDsHash below).
        sid0 = 1 + 1000 * (ring - 1)
        for j in range(n_spots):
            eta = -math.pi + 2 * math.pi * j / n_spots
            r_exp = 200_000.0
            y_exp, z_exp = -r_exp * math.sin(eta), r_exp * math.cos(eta)
            r_obs = r_exp + bias
            y_obs, z_obs = -r_obs * math.sin(eta), r_obs * math.cos(eta)
            fb[rep, j, 0] = sid0 + j                 # SpotID (1-based, > 0)
            fb[rep, j, 1] = y_obs
            fb[rep, j, 2] = z_obs
            fb[rep, j, 3] = 12.0                     # omega obs
            fb[rep, j, 4:7] = (1.0, 0.0, 0.0)        # g vector (unused here)
            fb[rep, j, 7] = y_exp
            fb[rep, j, 8] = z_exp
            fb[rep, j, 9] = 12.0                     # omega exp -> dome = 0
            fb[rep, j, 19] = 0.25                    # internal angle
        kept.append(CParityKeptGrain(
            grain_id=sid0,
            rep_pos=rep,
            member_positions=np.arange(gi + 1),      # cluster size = gi + 1
            member_ids=np.arange(gi + 1),
            orient_mat=np.eye(3),
            position=np.zeros(3),
            lattice=np.array([3.6, 3.6, 3.6, 90.0, 90.0, 90.0]),
            diff_pos=abs(bias), diff_ome=0.0, diff_angle=0.25,
            grain_radius=50.0, confidence=0.9,
        ))
    ids_hash = IDsHash(
        ring_nrs=np.array([1, 2], dtype=np.int64),
        id_starts=np.array([1, 1001], dtype=np.int64),
        id_ends=np.array([1001, 2001], dtype=np.int64),
        d_spacings=np.array([2.08, 1.80]),
    )
    return fb, kept, ids_hash


def test_gather_collects_residuals():
    fb, kept, ids_hash = _make_run()
    cache = gather_per_grain_spot_data(
        kept, fb, distance_um=LSD_UM, wavelength_a=WAVELENGTH_A,
        ids_hash=ids_hash, progress=False,
    )
    assert cache[0] is not None and "resid" in cache[0]
    r0 = cache[0]["resid"]
    assert r0.shape == (40, len(SPOT_RESIDUAL_COLS))
    # Injected +5 µm radial bias recovered on every spot, no tangential leak.
    np.testing.assert_allclose(r0[:, 6], 5.0, atol=1e-6)
    np.testing.assert_allclose(r0[:, 7], 0.0, atol=1e-6)
    assert set(r0[:, 2].astype(int)) == {1}          # ring from IDsHash
    assert set(r0[:, 0].astype(int)) == {0}          # grain_idx
    np.testing.assert_allclose(cache[1]["resid"][:, 6], -5.0, atol=1e-6)
    assert set(cache[1]["resid"][:, 2].astype(int)) == {2}


def test_collect_residuals_off():
    fb, kept, ids_hash = _make_run()
    cache = gather_per_grain_spot_data(
        kept, fb, distance_um=LSD_UM, wavelength_a=WAVELENGTH_A,
        ids_hash=ids_hash, progress=False, collect_residuals=False,
    )
    assert "resid" not in cache[0]
    assert "ds_obs" in cache[0]          # the strain path is unaffected


def test_write_residual_diagnostics_h5(tmp_path):
    h5py = pytest.importorskip("h5py")
    fb, kept, ids_hash = _make_run()
    cache = gather_per_grain_spot_data(
        kept, fb, distance_um=LSD_UM, wavelength_a=WAVELENGTH_A,
        ids_hash=ids_hash, progress=False,
    )
    out = tmp_path / "processgrains_diagnostics.h5"
    written = write_residual_diagnostics(
        out_path=out, kept_grains=kept, spot_cache=cache,
    )
    assert written == out and out.exists()

    with h5py.File(out, "r") as f:
        assert f["attrs"].attrs["mode"] == "c_parity"
        st = f["residuals/spot_table"]
        assert st.shape == (80, len(SPOT_RESIDUAL_COLS))
        assert st.attrs["columns"] == ",".join(SPOT_RESIDUAL_COLS)

        # grain_idx is Grains.csv row order: grain 0 is the +5 µm one.
        assert f["residuals/grain_med_drad_um"][0] == pytest.approx(5.0, abs=1e-6)
        assert f["residuals/grain_med_drad_um"][1] == pytest.approx(-5.0, abs=1e-6)
        assert f["residuals/grain_n_spots"][:].tolist() == [40, 40]
        assert f["residuals/grain_med_internal_angle_deg"][0] == pytest.approx(0.25)

        # Per-ring dR/R: +5 µm on a 200 mm radius = +25 ppm on ring 1.
        rings = f["residuals/ring_nr"][:].tolist()
        ppm = f["residuals/ring_drad_ppm"][:]
        assert ppm[rings.index(1)] == pytest.approx(25.0, rel=1e-3)
        assert ppm[rings.index(2)] == pytest.approx(-25.0, rel=1e-3)

        # cluster_sizes is real (seeds merged per grain); the spot-aware-only
        # counters are OMITTED, not zero-filled, so a reader cannot mistake
        # "c_parity does not compute this" for "measured zero".
        assert f["diagnostics/cluster_sizes"][:].tolist() == [1, 2]
        for absent in ("n_resolved_hkls", "n_majority_hkls",
                       "n_residual_tie_hkls", "n_forward_sim_hkls"):
            assert absent not in f["diagnostics"]


def test_no_residuals_writes_nothing(tmp_path):
    """No FitBest -> no sidecar, and no exception."""
    _, kept, _ = _make_run()
    out = tmp_path / "processgrains_diagnostics.h5"
    assert write_residual_diagnostics(
        out_path=out, kept_grains=kept, spot_cache=[None, None],
    ) is None
    assert not out.exists()
    assert write_residual_diagnostics(
        out_path=out, kept_grains=kept, spot_cache=None,
    ) is None


def test_ring_for_spot_ids_vectorised():
    _, _, ids_hash = _make_run()
    got = ids_hash.ring_for_spot_ids(np.array([1, 1000, 1001, 2000, 2001, 0]))
    assert got.tolist() == [1, 1, 2, 2, -1, -1]
    # Matches the scalar method element for element.
    for sid in (1, 1000, 1001, 2000, 2001, 0):
        assert int(ids_hash.ring_for_spot_ids(np.array([sid]))[0]) == \
            ids_hash.ring_for_spot_id(sid)
