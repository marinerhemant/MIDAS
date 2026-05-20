"""Tests for Indexer.run_scanning (P5b) — voxel grid + consolidated output.

These tests verify the orchestration layer end-to-end without requiring
a full real-data fixture:

- Voxel grid is built as the Cartesian product of 1-D scan_positions
  (matches IndexerScanningOMP.c:1667-1683).
- ``scan_kwargs()`` returns empty dict when ``scan_positions`` is None
  (FF default), threading-safe no-op.
- A minimal scan run writes a valid IndexBest_all.bin readable by the
  pf_MIDAS parser (consolidated I/O round-trip).

No real C-binary parity here; that's P5c, gated on a frozen golden
fixture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from midas_index.indexer import Indexer, _seeds_to_record_block
from midas_index.io.consolidated import read_index_best_all
from midas_index.params import IndexerParams
from midas_index.pipeline import IndexerContext
from midas_index.result import IndexerResult, SeedResult


# ---------------------------------------------------------------------------
# scan_kwargs() helper on IndexerContext
# ---------------------------------------------------------------------------


def _empty_context() -> IndexerContext:
    """Build a context just enough for scan_kwargs(); no real obs needed."""
    p = IndexerParams(
        px=200.0, Distance=1e6, Wavelength=0.18,
        SpaceGroup=225,
        EtaBinSize=0.1, OmeBinSize=0.1,
        StepsizeOrient=0.5, MarginEta=2.0, MarginOme=0.5,
        RingRadii={1: 30000.0},
    )
    return IndexerContext(
        params=p,
        hkls_real=np.zeros((1, 6)),
        hkls_int=np.zeros((1, 4)),
        obs=np.zeros((1, 9)),
        bin_data=np.zeros(0, dtype=np.int32),
        bin_ndata=np.zeros(0, dtype=np.int32),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )


def test_scan_kwargs_empty_when_off():
    ctx = _empty_context()
    assert ctx.scan_kwargs(n_tuples=5) == {}


def test_scan_kwargs_returns_voxel_xy_when_set():
    ctx = _empty_context()
    ctx.scan_positions = torch.tensor([0.0, 5.0, 10.0], dtype=torch.float64)
    ctx.current_voxel_xy = torch.tensor([5.0, 0.0], dtype=torch.float64)
    ctx.scan_pos_tol_um = 2.0
    kw = ctx.scan_kwargs(n_tuples=3)
    assert kw["scan_pos_tol_um"] == 2.0
    # Default is single-sided (matches C; see params.py docstring).
    assert kw["friedel_symmetric_scan_filter"] is False
    assert kw["voxel_xy"].shape == (3, 2)
    np.testing.assert_array_equal(
        kw["voxel_xy"][0].numpy(), [5.0, 0.0],
    )


# ---------------------------------------------------------------------------
# Voxel grid construction (mirrors C IndexerScanningOMP.c:1667-1683)
# ---------------------------------------------------------------------------


def test_voxel_grid_cartesian_product_layout():
    """grid[v=i*nScans+j] = (positions[j], positions[i]) — j is x, i is y."""
    scan_positions = torch.tensor([0.0, 10.0, 20.0], dtype=torch.float64)
    n_scans = 3
    idx = torch.arange(n_scans * n_scans)
    i_idx = idx // n_scans
    j_idx = idx % n_scans
    voxel_xy = torch.stack(
        [scan_positions[j_idx], scan_positions[i_idx]], dim=-1,
    )
    assert voxel_xy.shape == (9, 2)
    # v=0 → (j=0, i=0) → (0, 0)
    np.testing.assert_array_equal(voxel_xy[0].numpy(), [0.0, 0.0])
    # v=1 → (j=1, i=0) → (10, 0)
    np.testing.assert_array_equal(voxel_xy[1].numpy(), [10.0, 0.0])
    # v=3 → (j=0, i=1) → (0, 10)
    np.testing.assert_array_equal(voxel_xy[3].numpy(), [0.0, 10.0])
    # v=8 → (j=2, i=2) → (20, 20)
    np.testing.assert_array_equal(voxel_xy[8].numpy(), [20.0, 20.0])


# ---------------------------------------------------------------------------
# _seeds_to_record_block column ordering
# ---------------------------------------------------------------------------


def test_seeds_to_record_block_layout():
    """Pinning: column 0 = spot_id, col 1 = avg_ia, cols 2-10 = OM,
    cols 11-13 = pos, col 14 = nExpected, col 15 = nMatches."""
    seed = SeedResult(
        spot_id=42,
        best_or_mat=torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]],
                                 dtype=torch.float64),
        best_pos=torch.tensor([1.5, 2.5, 3.5], dtype=torch.float64),
        n_matches=20,
        n_t_spots=24,
        n_t_frac_calc=22,
        frac_matches=20 / 22,
        avg_ia=0.0012,
        matched_ids=torch.tensor([1, 2, 3], dtype=torch.int64),
    )
    vals, keys, ids = _seeds_to_record_block(IndexerResult(0, 1, [seed]))
    assert vals.shape == (1, 16)
    row = vals[0]
    assert row[0] == 42.0                                   # spot_id
    assert abs(row[1] - 0.0012) < 1e-12                     # avg_ia
    np.testing.assert_array_equal(row[2:11], [1, 0, 0, 0, 1, 0, 0, 0, 1])
    np.testing.assert_array_equal(row[11:14], [1.5, 2.5, 3.5])
    assert row[14] == 24.0                                  # nExpected
    assert row[15] == 20.0                                  # nMatches
    # Keys: [SpotID, nMatches, nIDs, reserved]
    assert keys.shape == (1, 4)
    np.testing.assert_array_equal(keys[0], [42, 20, 3, 0])
    # IDs: concatenated matched_ids
    np.testing.assert_array_equal(ids, [1, 2, 3])


def test_seeds_to_record_block_drops_unmatched():
    """Seeds with n_matches=0 should be dropped — only accepted hits emit rows."""
    bad_seed = SeedResult(
        spot_id=99,
        best_or_mat=torch.eye(3, dtype=torch.float64),
        best_pos=torch.zeros(3, dtype=torch.float64),
        n_matches=0,
        n_t_spots=10,
        n_t_frac_calc=10,
        frac_matches=0.0,
        avg_ia=0.0,
        matched_ids=torch.zeros(0, dtype=torch.int64),
    )
    vals, keys, ids = _seeds_to_record_block(IndexerResult(0, 1, [bad_seed]))
    assert vals.shape == (0, 16)
    assert keys.shape == (0, 4)
    assert ids.shape == (0,)


# ---------------------------------------------------------------------------
# End-to-end: write_index_best_all roundtrips through the consolidated reader
# ---------------------------------------------------------------------------


def test_run_scanning_writes_readable_indexbest_all(tmp_path: Path):
    """Smoke: run_scanning on an empty-obs setup produces a valid empty
    IndexBest_all.bin (n_vox = nScans×nScans, all blocks empty).

    This exercises the per-voxel loop + writer path without needing a
    real fixture. Real-data validation lands in P5c.
    """
    p = IndexerParams(
        px=200.0, Distance=1e6, Wavelength=0.18,
        SpaceGroup=225,
        EtaBinSize=0.1, OmeBinSize=0.1,
        StepsizeOrient=0.5,
        MarginEta=2.0, MarginOme=0.5, MarginRad=10.0, MarginRadial=10.0,
        RingNumbers=[1],
        RingRadii={1: 30000.0},
        scan_pos_tol_um=2.0,
        friedel_symmetric_scan_filter=True,
        multi_solution_output=True,
    )
    ind = Indexer(p, device="cpu")

    # Hand it minimal observations so load_observations() doesn't need files.
    # 10-col obs (PF Spots.bin layout) with one dummy spot, scan_nr=0.
    ind._observations = {
        "spots": np.zeros((1, 10), dtype=np.float64),
        "bin_data": np.zeros(0, dtype=np.int32),
        "bin_ndata": np.zeros(0, dtype=np.int32),
        "hkls_real": np.zeros((1, 6), dtype=np.float64),
        "hkls_int": np.zeros((1, 4), dtype=np.int64),
        "spot_ids": np.zeros(0, dtype=np.int64),
    }
    out = tmp_path / "IndexBest_all.bin"
    n_vox = ind.run_scanning(
        scan_positions=np.array([0.0, 5.0, 10.0]),
        out_path=out,
        num_procs=1,
    )
    assert n_vox == 9       # 3 × 3 grid
    assert out.exists()
    result = read_index_best_all(out)
    assert result.n_voxels == 9
    # No spots → no matches → all per-voxel records empty.
    np.testing.assert_array_equal(result.n_sol_arr, np.zeros(9, dtype=np.int32))


def test_run_scanning_preserves_acquisition_order_for_scan_filter(
    tmp_path: Path, monkeypatch
):
    """Non-monotonic (alternating) acquisition order must NOT be sorted into
    the scan filter.

    Regression for the scan_nr<->position mis-association: positions.csv is
    indexed by acquisition/file order (== Spots.bin col-9 scan_nr). The
    per-voxel scan filter looks up ``scan_positions[scan_nr]``, so the
    context MUST receive the acquisition-order array, while ONLY the voxel
    spatial grid is sorted. If the indexer sorted both, an alternating scan
    (0,-1,1,-2,2) would gate every acquisition against the wrong physical
    position.
    """
    import midas_index.pipeline as pl

    captured: list = []
    orig_init = pl.IndexerContext.__init__

    def spy_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        captured.append(self)

    monkeypatch.setattr(pl.IndexerContext, "__init__", spy_init)

    p = IndexerParams(
        px=200.0, Distance=1e6, Wavelength=0.18,
        SpaceGroup=225,
        EtaBinSize=0.1, OmeBinSize=0.1,
        StepsizeOrient=0.5,
        MarginEta=2.0, MarginOme=0.5, MarginRad=10.0, MarginRadial=10.0,
        RingNumbers=[1],
        RingRadii={1: 30000.0},
        scan_pos_tol_um=2.0,
        multi_solution_output=True,
    )
    ind = Indexer(p, device="cpu")
    ind._observations = {
        "spots": np.zeros((1, 10), dtype=np.float64),
        "bin_data": np.zeros(0, dtype=np.int32),
        "bin_ndata": np.zeros(0, dtype=np.int32),
        "hkls_real": np.zeros((1, 6), dtype=np.float64),
        "hkls_int": np.zeros((1, 4), dtype=np.int64),
        "spot_ids": np.zeros(0, dtype=np.int64),
    }

    # Alternating acquisition order — NOT sorted.
    acq_positions = np.array([0.0, -5.0, 5.0, -10.0, 10.0])
    n_vox = ind.run_scanning(
        scan_positions=acq_positions,
        out_path=tmp_path / "IndexBest_all.bin",
        num_procs=1,
    )

    assert n_vox == 25                       # 5 × 5 grid
    assert captured, "IndexerContext was never constructed"
    ctx = captured[0]
    # The scan filter must see the ACQUISITION-ORDER positions verbatim.
    np.testing.assert_array_equal(
        ctx.scan_positions.cpu().numpy(), acq_positions,
    )
    # Sanity: the input really was non-monotonic, so a sort would have
    # changed it — proving the assertion above is non-trivial.
    assert np.any(np.diff(acq_positions) < 0)


def test_run_scanning_rejects_single_scan():
    """FF (single-scan) callers should use run(), not run_scanning."""
    p = IndexerParams(SpaceGroup=225, RingRadii={1: 30000.0})
    ind = Indexer(p, device="cpu")
    ind._observations = {
        "spots": np.zeros((1, 10), dtype=np.float64),
        "bin_data": np.zeros(0, dtype=np.int32),
        "bin_ndata": np.zeros(0, dtype=np.int32),
        "hkls_real": np.zeros((1, 6), dtype=np.float64),
        "hkls_int": np.zeros((1, 4), dtype=np.int64),
        "spot_ids": np.zeros(0, dtype=np.int64),
    }
    with pytest.raises(ValueError, match="n_scans >= 2"):
        ind.run_scanning(
            scan_positions=np.array([0.0]),
            out_path="/tmp/should_not_exist.bin",
        )


def test_run_scanning_voxel_sharding(tmp_path: Path):
    """voxel_block_nr / voxel_n_blocks splits the voxel grid into shards."""
    p = IndexerParams(
        SpaceGroup=225, RingRadii={1: 30000.0},
        EtaBinSize=0.1, OmeBinSize=0.1,
        scan_pos_tol_um=2.0, multi_solution_output=True,
    )
    ind = Indexer(p, device="cpu")
    ind._observations = {
        "spots": np.zeros((1, 10), dtype=np.float64),
        "bin_data": np.zeros(0, dtype=np.int32),
        "bin_ndata": np.zeros(0, dtype=np.int32),
        "hkls_real": np.zeros((1, 6), dtype=np.float64),
        "hkls_int": np.zeros((1, 4), dtype=np.int64),
        "spot_ids": np.zeros(0, dtype=np.int64),
    }
    out0 = tmp_path / "block0.bin"
    out1 = tmp_path / "block1.bin"
    n0 = ind.run_scanning(
        scan_positions=np.array([0.0, 5.0, 10.0]),
        out_path=out0,
        voxel_block_nr=0, voxel_n_blocks=2,
    )
    n1 = ind.run_scanning(
        scan_positions=np.array([0.0, 5.0, 10.0]),
        out_path=out1,
        voxel_block_nr=1, voxel_n_blocks=2,
    )
    assert n0 + n1 == 9                  # full grid covered by 2 shards
