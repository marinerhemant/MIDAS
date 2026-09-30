"""Compact ring slots: the nData.bin ring axis holds one slab per configured
ring (RingSlots.csv), not one per ring number up to the highest.

Gates:
  * rings exactly 1..N  -> compact is the legacy layout, byte for byte;
  * gapped rings        -> Data.bin identical, nData.bin is the legacy table
                           with its empty slabs removed (and shrinks by
                           n_slots / highest_ring);
  * the sidecar is always written and round-trips; malformed ones are refused.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from midas_transforms.bin_data import bin_data, bin_data_scanning
from midas_transforms.bin_data.core import (
    _bin_assignment,
    _bin_to_data_ndata,
    _build_ring_radii,
)
from midas_transforms.bin_data.voxel_binner import _bin_to_data_ndata_scanning
from midas_transforms.io import binary as bio
from midas_transforms.io import csv as csv_io
from midas_transforms.params import ParamsTest, read_paramstest, write_paramstest


def _params(rings, radii) -> ParamsTest:
    p = ParamsTest()
    p.Wavelength = 0.18
    p.Lsd = 1_000_000.0
    p.px = 200.0
    p.MarginOme = 1.0
    p.MarginEta = 500.0
    p.EtaBinSize = 5.0
    p.OmeBinSize = 5.0
    p.StepSizeOrient = 0.2
    p.NoSaveAll = 0
    p.RingNumbers = list(rings)
    p.RingRadii = list(radii)
    p.LatticeConstant = (3.6, 3.6, 3.6, 90.0, 90.0, 90.0)
    p.SpaceGroup = 225
    p.RingToIndex = 1
    p.BeamSize = 5.0
    return p


def _inputall(p: ParamsTest, extra_rings=()) -> tuple[np.ndarray, np.ndarray]:
    """Spots on every configured ring, plus spots on ``extra_rings`` (rings
    that are NOT configured: the binner must drop them in both layouts)."""
    rng = np.random.default_rng(1)
    rows, sid = [], 1
    rr_of = dict(zip(p.RingNumbers, p.RingRadii))
    for rn in list(p.RingNumbers) + list(extra_rings):
        rr = rr_of.get(rn, 800.0)
        for eta in (-150.0, -120.0, -60.0, -30.0, 30.0, 60.0, 120.0, 150.0):
            yl = -rr * math.sin(math.radians(eta)) * p.px
            zl = rr * math.cos(math.radians(eta)) * p.px
            om = -90 + 180 * rng.random()
            tth = math.degrees(math.atan2(rr * p.px, p.Lsd))
            rows.append([yl, zl, om, 5.0, float(sid), float(rn), eta, tth])
            sid += 1
    spots = np.array(rows, dtype=np.float64)
    extra = np.zeros((spots.shape[0], 18))
    extra[:, :8] = spots
    extra[:, 8] = spots[:, 2]
    extra[:, 14] = 1.0
    return spots, extra


def _legacy_ff_tables(p: ParamsTest, spots: np.ndarray):
    """The pre-compact writer: ring axis = [0, highest_ring), slot = ring-1."""
    t = torch.from_numpy(spots)
    radii = _build_ring_radii(p).to(torch.float64)
    a = _bin_assignment(t, radii, margin_ome=p.MarginOme, margin_eta=p.MarginEta,
                        eta_bin_size=p.EtaBinSize, ome_bin_size=p.OmeBinSize,
                        step_size_orient=p.StepSizeOrient)
    data, ndata = _bin_to_data_ndata(
        *a, n_ring_bins=p.highest_ring_no,
        n_eta_bins=math.ceil(360 / p.EtaBinSize),
        n_ome_bins=math.ceil(360 / p.OmeBinSize))
    pairs = np.zeros((data.numel(), 2), dtype=np.uint64)
    pairs[:, 0] = data.numpy().astype(np.uint64)
    return pairs, ndata.numpy().reshape(-1, 2).astype(np.uint64)


def _expand_to_legacy(ndata_compact: np.ndarray, slots, n_legacy: int) -> np.ndarray:
    """Insert empty slabs for the rings without a slot; offsets = exclusive
    cumsum of counts (what a legacy writer produces)."""
    n_slots = len(slots)
    per = ndata_compact.shape[0] // n_slots
    counts = np.zeros((n_legacy, per), dtype=np.uint64)
    c = ndata_compact[:, 0].reshape(n_slots, per)
    for s, r in enumerate(slots):
        counts[r - 1] = c[s]
    counts = counts.ravel()
    offs = np.zeros_like(counts)
    offs[1:] = np.cumsum(counts[:-1])
    return np.stack([counts, offs], axis=1)


def test_contiguous_rings_are_byte_identical_to_legacy(tmp_path: Path):
    p = _params([1, 2, 3], [500.0, 700.0, 900.0])
    spots, extra = _inputall(p)
    bin_data(tmp_path, paramstest=p, spots_inputall=spots, extra_inputall=extra)
    data_ref, ndata_ref = _legacy_ff_tables(p, spots)
    assert (tmp_path / "nData.bin").read_bytes() == ndata_ref.tobytes()
    assert (tmp_path / "Data.bin").read_bytes() == data_ref.tobytes()
    assert (tmp_path / "RingSlots.csv").read_text() == "RingNr Slot\n1 0\n2 1\n3 2\n"


@pytest.mark.parametrize("rings", [[1, 3, 7], [3, 7, 1]])
def test_gapped_rings_ff_compact_is_legacy_minus_empty_slabs(tmp_path: Path, rings):
    radii = {1: 500.0, 3: 700.0, 7: 900.0}
    p = _params(rings, [radii[r] for r in rings])
    spots, extra = _inputall(p, extra_rings=(2, 5))   # unconfigured: dropped
    res = bin_data(tmp_path, paramstest=p, spots_inputall=spots, extra_inputall=extra)
    assert res.ring_slots == [1, 3, 7] and res.n_ring_bins == 3
    assert bio.read_ring_slots_csv(tmp_path / "RingSlots.csv") == [1, 3, 7]

    data_ref, ndata_ref = _legacy_ff_tables(p, spots)
    ndata_c = bio.read_ndata_bin_scanning(tmp_path / "nData.bin")
    n_eta = n_ome = 72
    assert ndata_c.shape[0] == 3 * n_eta * n_ome
    assert ndata_ref.shape[0] == 7 * n_eta * n_ome
    assert (tmp_path / "Data.bin").read_bytes() == data_ref.tobytes()
    assert np.array_equal(_expand_to_legacy(ndata_c, [1, 3, 7], 7), ndata_ref)
    assert ndata_c[:, 0].sum() == ndata_ref[:, 0].sum() > 0


def _write_pf_scans(d: Path, p: ParamsTest, n_scans: int = 3) -> np.ndarray:
    write_paramstest(p, d / "paramstest.txt")
    for scan in range(n_scans):
        rows = []
        for k, (rn, rr) in enumerate(list(zip(p.RingNumbers, p.RingRadii)) * 2):
            eta = (-90.0 + 37.0 * k + 30.0 * scan) % 360 - 180
            om = -50.0 + 20.0 * scan + 7.0 * k
            tth = math.degrees(math.atan2(rr * p.px, p.Lsd))
            yl = -rr * math.sin(math.radians(eta)) * p.px
            zl = rr * math.cos(math.radians(eta)) * p.px
            r = np.zeros(18)
            r[0], r[1], r[2] = yl, zl, om
            r[3] = 5.0 + 0.1 * k
            r[4] = scan * 100 + k + 1
            r[5], r[6], r[7] = rn, eta, tth
            r[8] = om
            r[9], r[10], r[11], r[12] = yl, zl, yl, zl
            r[13] = r[16] = 1000.0 + 5.0 * k
            r[17] = 0.01
            rows.append(r)
        csv_io.write_inputall_extra_csv(d / f"InputAllExtraInfoFittingAll{scan}.csv",
                                        np.array(rows))
    return np.array([-4.0, 0.0, 4.0][:n_scans])


def _legacy_pf_ndata(d: Path) -> tuple[np.ndarray, np.ndarray]:
    """Pre-compact PF writer tail (per-ring pack, slot = ring - 1)."""
    p = read_paramstest(d / "paramstest.txt")
    spots = np.fromfile(d / "Spots.bin", dtype=np.float64).reshape(-1, 10)
    st = torch.from_numpy(spots[:, :8].copy())
    scan = torch.from_numpy(spots[:, 9].copy()).long()
    radii = _build_ring_radii(p).to(torch.float64)
    n_eta, n_ome = math.ceil(360 / p.EtaBinSize), math.ceil(360 / p.OmeBinSize)
    counts = torch.zeros(p.highest_ring_no * n_eta * n_ome, dtype=torch.int64)
    parts = []
    for r in [i for i in range(radii.shape[0]) if float(radii[i]) > 0]:
        one = torch.zeros_like(radii)
        one[r] = radii[r]
        arr = list(_bin_assignment(st, one, margin_ome=p.MarginOme,
                                   margin_eta=p.MarginEta, eta_bin_size=p.EtaBinSize,
                                   ome_bin_size=p.OmeBinSize,
                                   step_size_orient=p.StepSizeOrient))
        arr.append(scan[arr[0]])
        dr, ndr = _bin_to_data_ndata_scanning(arr, n_ring_bins=p.highest_ring_no,
                                              n_eta_bins=n_eta, n_ome_bins=n_ome)
        counts += ndr[:, 0]
        if dr.shape[0]:
            parts.append(dr)
    data = torch.cat(parts).numpy().astype(np.uint64)
    offs = torch.zeros_like(counts)
    offs[1:] = torch.cumsum(counts[:-1], 0)
    return data, torch.stack([counts, offs], 1).numpy().astype(np.uint64)


def test_gapped_rings_pf_compact_is_legacy_minus_empty_slabs(tmp_path: Path):
    p = _params([1, 3, 7], [500.0, 700.0, 900.0])
    pos = _write_pf_scans(tmp_path, p)
    res = bin_data_scanning(tmp_path, n_scans=3, scan_positions=pos)
    assert res.ring_slots == [1, 3, 7]
    data_ref, ndata_ref = _legacy_pf_ndata(tmp_path)
    ndata_c = bio.read_ndata_bin_scanning(tmp_path / "nData.bin")
    assert ndata_c.shape[0] * 7 == ndata_ref.shape[0] * 3
    assert (tmp_path / "Data.bin").read_bytes() == data_ref.tobytes()
    assert np.array_equal(_expand_to_legacy(ndata_c, [1, 3, 7], 7), ndata_ref)


def test_pf_contiguous_rings_byte_identical_to_legacy(tmp_path: Path):
    p = _params([1, 2, 3], [500.0, 700.0, 900.0])
    pos = _write_pf_scans(tmp_path, p)
    bin_data_scanning(tmp_path, n_scans=3, scan_positions=pos)
    data_ref, ndata_ref = _legacy_pf_ndata(tmp_path)
    assert (tmp_path / "nData.bin").read_bytes() == ndata_ref.tobytes()
    assert (tmp_path / "Data.bin").read_bytes() == data_ref.tobytes()
    assert (tmp_path / "RingSlots.csv").exists()


@pytest.mark.parametrize("text", [
    "1 0\n2 1\n",                      # no header
    "RingNr Slot\n1 0\n1 1\n",         # duplicate ring
    "RingNr Slot\n1 0\n2 0\n",         # duplicate slot
    "RingNr Slot\n1 0\n3 2\n",         # slots not 0..N-1
    "RingNr Slot\n",                   # empty
])
def test_malformed_sidecar_is_refused(tmp_path: Path, text):
    f = tmp_path / "RingSlots.csv"
    f.write_text(text)
    with pytest.raises(ValueError):
        bio.read_ring_slots_csv(f)
