"""The C refiner's nData.bin size gate (compact ring slots, RingSlots.csv).

With DoDynamicReassignment on, FitUnified mmaps nData.bin and addresses it
by ring slot. A table whose size does not fit its layout (compact table
without its sidecar, or a sidecar that lists the wrong number of slots) must
stop the refiner with a clear message instead of handing every theoretical
spot another ring's candidates. The gate runs right after ExtraInfo.bin is
mapped, so these inputs never need to be a real refinement.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from midas_fit_grain import backend_c

pytestmark = pytest.mark.skipif(
    not backend_c.available(), reason="midas_fitgrain C binary not built")

N_ETA = N_OME = 72          # 5-degree bins
SLAB = N_ETA * N_OME        # bins per ring slot


def _stage(d: Path, *, n_slabs: int, sidecar: str | None) -> Path:
    (d / "Results").mkdir(parents=True)
    (d / "Output").mkdir()
    lines = [
        "Wavelength 0.18", "Lsd 1000000", "px 200",
        "LatticeParameter 3.6 3.6 3.6 90 90 90", "SpaceGroup 225",
        "EtaBinSize 5", "OmeBinSize 5", "DoDynamicReassignment 1",
        "RingNumbers 1", "RingRadii 500", "RingNumbers 3", "RingRadii 700",
        "RingNumbers 7", "RingRadii 900",
        f"OutputFolder {d / 'Output'}", f"ResultFolder {d / 'Results'}",
    ]
    (d / "paramstest.txt").write_text("\n".join(lines) + "\n")
    np.zeros((4, 16)).tofile(d / "ExtraInfo.bin")
    np.zeros((4, 10)).tofile(d / "Spots.bin")
    np.zeros((0, 2), dtype=np.uint64).tofile(d / "Data.bin")
    np.zeros((n_slabs * SLAB, 2), dtype=np.uint64).tofile(d / "nData.bin")
    if sidecar is not None:
        (d / "RingSlots.csv").write_text(sidecar)
    return d


def _run(d: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(backend_c.binary_path()), str(d / "paramstest.txt"), "0", "1", "1", "1"],
        cwd=d, capture_output=True, timeout=120)


SIDECAR_3 = "RingNr Slot\n1 0\n3 1\n7 2\n"


def test_compact_table_without_sidecar_is_refused(tmp_path):
    proc = _run(_stage(tmp_path, n_slabs=3, sidecar=None))   # legacy wants 7
    err = proc.stderr.decode(errors="replace")
    assert proc.returncode != 0
    assert "nData.bin is" in err and "RingSlots.csv" in err, err


def test_sidecar_slot_count_mismatch_is_refused(tmp_path):
    proc = _run(_stage(tmp_path, n_slabs=7, sidecar=SIDECAR_3))
    err = proc.stderr.decode(errors="replace")
    assert proc.returncode != 0
    assert "RingSlots.csv lists 3 ring slots" in err, err


@pytest.mark.parametrize("n_slabs,sidecar", [(3, SIDECAR_3), (7, None)])
def test_matching_table_passes_the_gate(tmp_path, n_slabs, sidecar):
    """Consistent layouts get past the gate (the run may fail later on these
    dummy inputs; only the gate's verdict is under test)."""
    proc = _run(_stage(tmp_path, n_slabs=n_slabs, sidecar=sidecar))
    out = proc.stdout.decode(errors="replace")
    err = proc.stderr.decode(errors="replace")
    assert "nData.bin is" not in err, err
    assert "Bin dims: rings=%d" % (3 if sidecar else 7) in out, out
