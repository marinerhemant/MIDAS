"""Ring numbers must not be able to run off the ends of the fixed tables.

`RingHKL`, `RingTtheta`, `RingRadii` and `etamargins` are all sized
`MAX_N_RINGS` and indexed by the ring number itself. An unchecked store was a
live defect: `hkls.csv` legitimately contains rings far beyond the detector,
because the generator's reach is `MaxRingRad` (aliased to `RhoD`), so an
oversized value produces hundreds of them.

Measured on 20-ID Varex data before the fix: `RhoD 2000000` on a 2880 px
detector generated 745 rings; writing `RingHKL[745]` ran past the array,
through `RingTtheta`, and into the `data` / `ndata` bin pointers declared after
it. The indexer then matched nothing, exited 0, and reported success --
0 of 4569 seeds indexed. After the fix the same input indexes 3497 of 4569,
matching both the ring-capped control and the independent Python backend.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from midas_index import backend_c

DATA = Path(__file__).parent / "data" / "ref_dataset_unified"
MAX_N_RINGS = 500          # keep in step with c_src/IndexerUnified.c


def _fixture_ready() -> bool:
    return (DATA / "hkls.csv").exists() and (DATA / "paramstest.txt").exists()


pytestmark = [
    pytest.mark.skipif(not backend_c.available(),
                       reason="C indexer binary not built"),
    pytest.mark.skipif(not _fixture_ready(),
                       reason="unified reference fixture missing"),
]


def _stage(tmp_path: Path) -> Path:
    """Copy the reference dataset somewhere writable."""
    run = tmp_path / "run"
    shutil.copytree(DATA, run)
    (run / "Output").mkdir(exist_ok=True)
    (run / "Results").mkdir(exist_ok=True)
    pt = run / "paramstest.txt"
    text = [ln for ln in pt.read_text().splitlines()
            if ln.split()[:1] not in (["OutputFolder"], ["ResultFolder"])]
    text += [f"OutputFolder {run / 'Output'}", f"ResultFolder {run / 'Results'}"]
    pt.write_text("\n".join(text) + "\n")
    return run


def _append_far_rings(hkls: Path, ring: int, n: int = 50) -> None:
    """Append rows on a ring beyond MAX_N_RINGS, as an oversized MaxRingRad
    would. Column 5 (1-indexed) is the ring number."""
    lines = hkls.read_text().splitlines()
    template = lines[1].split()
    out = []
    for i in range(n):
        row = list(template)
        row[4] = str(ring + i)
        out.append(" ".join(row))
    hkls.write_text("\n".join(lines + out) + "\n")


def _run(run: Path, n_seeds: int = 8, procs: int = 2):
    return subprocess.run(
        [str(backend_c.binary_path()), "paramstest.txt", "0", "1",
         str(n_seeds), str(procs)],
        cwd=str(run), capture_output=True, text=True, timeout=600)


def test_rings_beyond_the_table_are_ignored_not_written(tmp_path):
    """The overflow row must be dropped, and the run must still work.

    Silently corrupting memory and silently dropping the row look identical in
    the exit code, so this asserts on the warning as well: the operator needs
    to know their ring cap is wrong even though the run survived.
    """
    run = _stage(tmp_path)
    _append_far_rings(run / "hkls.csv", MAX_N_RINGS + 245)      # 745, as measured
    proc = _run(run)

    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (run / "Output" / "IndexBest_all.bin").exists()
    assert "outside [0, 500)" in proc.stderr
    assert "745" in proc.stderr or "794" in proc.stderr          # highest seen


def test_out_of_range_rings_do_not_change_the_answer(tmp_path):
    """Dropping unusable rings must be a no-op on the result.

    They are beyond the detector by construction, so the indexed-seed count has
    to match a run without them. This is what distinguishes a real fix from one
    that merely stops the crash.
    """
    clean = _stage(tmp_path / "clean")
    dirty = _stage(tmp_path / "dirty")
    _append_far_rings(dirty / "hkls.csv", MAX_N_RINGS + 245)

    a, b = _run(clean), _run(dirty)
    assert a.returncode == 0 and b.returncode == 0
    assert (clean / "Output" / "IndexBest_all.bin").read_bytes() == \
           (dirty / "Output" / "IndexBest_all.bin").read_bytes()


def test_clean_input_produces_no_warning(tmp_path):
    """No false alarms on a normal file."""
    run = _stage(tmp_path)
    proc = _run(run)
    assert proc.returncode == 0
    assert "outside [0," not in proc.stderr


def test_requested_ring_beyond_the_table_is_fatal(tmp_path):
    """A ring in RingNumbers is different: it was asked for, and it indexes
    RingRadii / etamargins directly. Ignoring it would silently change what was
    measured, so this one must stop the run."""
    run = _stage(tmp_path)
    pt = run / "paramstest.txt"
    lines = pt.read_text().splitlines()
    for i, ln in enumerate(lines):
        if ln.split()[:1] == ["RingNumbers"]:
            lines[i] = "RingNumbers 900;"
            break
    else:
        pytest.skip("fixture has no RingNumbers entry to corrupt")
    pt.write_text("\n".join(lines) + "\n")

    proc = _run(run)
    assert proc.returncode != 0
    assert "RingNumbers 900" in proc.stderr and "outside [0, 500)" in proc.stderr
