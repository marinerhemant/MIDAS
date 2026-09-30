"""Compact ring slots through the READERS (C indexer + Python loader).

The bin table's ring axis is a slot axis. A table written for a gapped ring
set ({1, 3, 5} here) in the legacy layout (slot = RingNr - 1, empty slabs for
rings 2 and 4) and the same table in the compact layout (3 slabs +
RingSlots.csv) must index to byte-identical IndexBest_all.bin, and the
compact nData.bin must be 3/5 the size. A table whose size does not fit its
layout must make the C indexer exit non-zero with a clear message instead of
silently mis-addressing bins.

Uses the tracked 5-grain PF fixture inputs (Spots.bin, hkls.csv,
positions.csv) with coarse 1-degree bins so the tables are ~2 MB per slab.
"""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from midas_index import backend_c

torch = pytest.importorskip("torch")
pytest.importorskip("midas_transforms")

from midas_transforms.bin_data.core import (  # noqa: E402
    _bin_assignment, _build_ring_radii, compact_ring_slots)
from midas_transforms.bin_data.voxel_binner import _bin_to_data_ndata_scanning  # noqa: E402
from midas_transforms.io import binary as tbio  # noqa: E402
from midas_transforms.params import read_paramstest  # noqa: E402

FIX = Path(__file__).parent / "data" / "scanning_5grain_golden"
RINGS = (1, 3, 5)
OUT_FILES = ("IndexBest_all.bin", "IndexKey_all.bin", "IndexBest_IDs_all.bin")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not backend_c.available(),
                       reason="midas_indexer C binary not built"),
    pytest.mark.skipif(not (FIX / "Spots.bin").is_file(),
                       reason=f"fixture inputs missing at {FIX}"),
]


def _gapped_paramstest(src: Path, dst: Path, out: Path) -> None:
    """Fixture paramstest restricted to RINGS, with 1-degree bins."""
    lines = src.read_text().splitlines()
    radii = {}
    keep = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("RingNumbers "):
            r = int(ln.split()[1])
            radii[r] = lines[i + 1]            # the paired RingRadii line
            i += 2
            continue
        if ln.startswith(("OutputFolder", "ResultFolder", "EtaBinSize", "OmeBinSize")):
            i += 1
            continue
        keep.append(ln)
        i += 1
    for r in RINGS:
        keep += [f"RingNumbers {r}", radii[r]]
    keep += ["EtaBinSize 1.0;", "OmeBinSize 1.0;",
             f"OutputFolder {out}", f"ResultFolder {out.parent / 'Results'}"]
    dst.write_text("\n".join(keep) + "\n")


def _write_bins(d: Path, *, compact: bool) -> None:
    """The midas_transforms PF writer tail, either layout, from Spots.bin."""
    p = read_paramstest(d / "paramstest.txt")
    spots = np.fromfile(d / "Spots.bin", dtype=np.float64).reshape(-1, 10)
    st = torch.tensor(spots[:, :8], dtype=torch.float64)
    scan = torch.tensor(spots[:, 9]).long()
    radii = _build_ring_radii(p).to(torch.float64)
    n_eta, n_ome = math.ceil(360 / p.EtaBinSize), math.ceil(360 / p.OmeBinSize)
    if compact:
        slots, lut = compact_ring_slots(p)
        n_ring = len(slots)
    else:
        slots, lut, n_ring = None, None, p.highest_ring_no
    counts = torch.zeros(n_ring * n_eta * n_ome, dtype=torch.int64)
    parts = []
    for r in [i for i in range(radii.shape[0]) if float(radii[i]) > 0]:
        one = torch.zeros_like(radii)
        one[r] = radii[r]
        arr = list(_bin_assignment(st, one, margin_ome=p.MarginOme,
                                   margin_eta=p.MarginEta, eta_bin_size=p.EtaBinSize,
                                   ome_bin_size=p.OmeBinSize,
                                   step_size_orient=p.StepSizeOrient))
        arr.append(scan[arr[0]])
        dr, ndr = _bin_to_data_ndata_scanning(arr, n_ring_bins=n_ring, n_eta_bins=n_eta,
                                              n_ome_bins=n_ome, ring_slot_lut=lut)
        counts += ndr[:, 0]
        if dr.shape[0]:
            parts.append(dr)
    offs = torch.zeros_like(counts)
    offs[1:] = torch.cumsum(counts[:-1], 0)
    tbio.write_data_ndata_bin_scanning(
        d / "Data.bin", d / "nData.bin",
        torch.cat(parts).numpy().astype(np.uint64),
        torch.stack([counts, offs], 1).numpy().astype(np.uint64))
    if compact:
        tbio.write_ring_slots_csv(d / "RingSlots.csv", slots)


def _stage(root: Path, tag: str, *, compact: bool) -> Path:
    d = root / tag
    (d / "Output").mkdir(parents=True)
    for f in ("Spots.bin", "hkls.csv", "positions.csv", "SpotsToIndex.csv"):
        shutil.copy2(FIX / f, d / f)
    _gapped_paramstest(FIX / "paramstest.txt", d / "paramstest.txt", d / "Output")
    _write_bins(d, compact=compact)
    return d


def _index(d: Path):
    return backend_c.run_indexer(d / "paramstest.txt", n_work=15, num_procs=4,
                                 cwd=d)


@pytest.fixture(scope="module")
def dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("ring_slots")
    return {"legacy": _stage(root, "legacy", compact=False),
            "compact": _stage(root, "compact", compact=True)}


@pytest.fixture(scope="module")
def runs(dirs):
    return {k: _index(d) for k, d in dirs.items()}


def test_compact_table_is_three_fifths_of_legacy(dirs):
    leg = (dirs["legacy"] / "nData.bin").stat().st_size
    cmp_ = (dirs["compact"] / "nData.bin").stat().st_size
    assert leg == 5 * 360 * 360 * 16 and cmp_ == 3 * 360 * 360 * 16
    assert (dirs["legacy"] / "Data.bin").read_bytes() == \
        (dirs["compact"] / "Data.bin").read_bytes()


def test_indexer_output_byte_identical_legacy_vs_compact(dirs, runs):
    for k, proc in runs.items():
        assert proc.returncode == 0, f"{k}: " + proc.stderr.decode(errors="replace")
    assert b"compact ring slots from RingSlots.csv (3 slots)" in runs["compact"].stdout
    assert b"compact ring slots" not in runs["legacy"].stdout
    for f in OUT_FILES:
        a = (dirs["legacy"] / "Output" / f).read_bytes()
        b = (dirs["compact"] / "Output" / f).read_bytes()
        assert a == b, f"{f} differs between legacy-with-gaps and compact"
    # Not trivially empty: the gapped ring set still indexes grains.
    n_vox = int.from_bytes((dirs["compact"] / "Output" / "IndexBest_all.bin")
                           .read_bytes()[:4], "little")
    assert n_vox > 0
    best = (dirs["compact"] / "Output" / "IndexBest_all.bin").stat().st_size
    assert best > 4 + 12 * n_vox, "no solutions recorded"


def test_python_loader_expands_compact_to_legacy(dirs):
    from midas_index.io.binary import read_bins_scanning
    g = dict(n_eta_bins=360, n_ome_bins=360, highest_ring=5)
    d_l, n_l = read_bins_scanning(dirs["legacy"], **g)
    d_c, n_c = read_bins_scanning(dirs["compact"], **g)
    assert np.array_equal(d_l, d_c) and np.array_equal(n_l, n_c)


def _mismatch_dir(dirs, root: Path, tag: str) -> Path:
    d = root / tag
    shutil.copytree(dirs["compact"], d, ignore=shutil.ignore_patterns("Output"))
    (d / "Output").mkdir()
    txt = (d / "paramstest.txt").read_text().replace(
        str(dirs["compact"] / "Output"), str(d / "Output"))
    (d / "paramstest.txt").write_text(txt)
    return d


def test_compact_table_without_sidecar_fails_loudly(dirs, tmp_path):
    d = _mismatch_dir(dirs, tmp_path, "no_sidecar")
    (d / "RingSlots.csv").unlink()
    proc = _index(d)
    err = proc.stderr.decode(errors="replace")
    assert proc.returncode != 0
    assert "nData.bin is" in err and "RingSlots.csv" in err, err


def test_sidecar_that_does_not_match_table_fails_loudly(dirs, tmp_path):
    d = _mismatch_dir(dirs, tmp_path, "wrong_sidecar")
    tbio.write_ring_slots_csv(d / "RingSlots.csv", [1, 3])     # 2 slots, table has 3
    proc = _index(d)
    err = proc.stderr.decode(errors="replace")
    assert proc.returncode != 0
    assert "RingSlots.csv lists 2 ring slots" in err, err
    from midas_index.io.binary import read_bins_scanning
    with pytest.raises(ValueError, match="RingSlots.csv lists 2"):
        read_bins_scanning(d, n_eta_bins=360, n_ome_bins=360, highest_ring=5)


def test_malformed_sidecar_fails_loudly(dirs, tmp_path):
    d = _mismatch_dir(dirs, tmp_path, "bad_sidecar")
    (d / "RingSlots.csv").write_text("RingNr Slot\n1 0\n3 0\n5 2\n")
    proc = _index(d)
    assert proc.returncode != 0
    assert "slot 0 listed twice" in proc.stderr.decode(errors="replace")
