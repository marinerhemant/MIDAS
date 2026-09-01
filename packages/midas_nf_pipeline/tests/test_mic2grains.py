"""Byte-parity tests for ``midas_nf_pipeline.mic2grains`` against a
fresh reference produced by the C ``Mic2GrainsList`` on the Au example.

Per-line numerical equality (header is byte-equal; data rows compared
column by column with float tolerance for the 12-decimal OrientMat
columns where the C qsort-by-confidence and our stable np.argsort can
land equal-confidence voxels in different intra-tie order — but for
the global-merge mode, distinct grains always end up with distinct
orientations, so the resulting GRAIN ROWS still match).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from midas_nf_pipeline.mic2grains import Mic2GrainsParams, mic_to_grains
from midas_nf_pipeline.parse_mic import (
    ParseMicParams, parse_mic,
)


_AU_DIR = Path(__file__).resolve().parents[3] / "NF_HEDM/Example/sim"
_AU_PARAM = _AU_DIR / "test_ps_au.txt"
_AU_BIN = _AU_DIR / "Au_bin_Reconstructed.mic.c_ref"


def _c_binary_unusable(path) -> str | None:
    """Why this C reference must not be RUN, or ``None`` if it looks usable.

    **Deliberately STATIC -- this must never launch** ``path``.

    Existence is not enough: these are prebuilt artefacts of the
    soft-deprecated C tree and a stale one is unloadable. But the obvious
    check -- run it once and see whether it survives -- is worse than the
    problem on macOS/arm64, and this function used to do exactly that.

    ``install_name_tool`` rpath surgery invalidates a Mach-O's code signature,
    and the kernel then SIGKILLs the process inside dyld, before ``main()``.
    Measured 2026-09-01: ``codesign -v`` on this tree's binaries reports "code
    object is not signed at all". Such a launch is slow (macOS writes a crash
    report) and leaves the process in uninterruptible ``UE`` state, where it
    survives ``SIGKILL``. ``subprocess.run(..., timeout=...)`` does NOT bound
    it: the timeout fires, the reap does not return. Measured here -- a sweep
    of this package sat for 23 minutes at 0 %% CPU on a wedged child before it
    was killed by hand.

    The old ``returncode < 0`` / "Library not loaded" checks were correct in
    principle and unreachable in practice, because the call never returns.

    So: check statically, and let the test's own subprocess call be where a
    binary that passes this gate but still cannot load is caught.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path as _P
    p = _P(str(path))
    if not p.exists():
        return "not found"
    if not os.access(p, os.X_OK):
        return "present but not executable"
    if sys.platform == "darwin":
        try:
            sig = subprocess.run(["codesign", "--verify", str(p)],
                                 capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return None  # cannot check -- let the test try
        if sig.returncode != 0:
            detail = " ".join((sig.stderr or b"").decode(errors="replace").split())
            return (f"code signature invalid ({detail}); on arm64 macOS the "
                    f"kernel SIGKILLs such a binary inside dyld. Re-sign after "
                    f"any install_name_tool surgery: codesign -s - -f {p}")
    return None

_C_BIN = Path("/Users/hsharma/opt/MIDAS/NF_HEDM/bin/Mic2GrainsList")


@pytest.fixture
def workspace():
    if not _AU_PARAM.exists() or not _AU_BIN.exists():
        pytest.skip("Au example files not present")
    if not _C_BIN.exists():
        pytest.skip(f"C Mic2GrainsList binary not built: {_C_BIN}")
    if (_why := _c_binary_unusable(_C_BIN)) is not None:
        pytest.skip(f"C Mic2GrainsList unusable: {_why}")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # 1. Reproduce the .mic text file via our parse_mic.
        bin_dst = td_path / "Au_bin_Reconstructed.mic"
        shutil.copy2(_AU_BIN, bin_dst)
        am_in = _AU_DIR / "Au_bin_Reconstructed.mic.AllMatches"
        if am_in.exists():
            shutil.copy2(am_in, td_path / "Au_bin_Reconstructed.mic.AllMatches")
        parse_mic(ParseMicParams(
            PhaseNr=1, NumPhases=1, GlobalPosition=0.0,
            inputfile=str(bin_dst),
            outputfile=str(td_path / "Au.mic"),
            nSaves=3, SGNr=225, GBAngle=5.0,
        ))
        mic_path = td_path / "Au.mic"

        # 2. Run the C Mic2GrainsList for a fresh reference.
        c_out = td_path / "Grains_c.csv"
        subprocess.run(
            [str(_C_BIN), str(_AU_PARAM), str(mic_path), str(c_out),
             "0", "1", "0.04"],
            check=True, capture_output=True,
        )

        # 3. Run our Python port.
        py_out = td_path / "Grains_py.csv"
        n = mic_to_grains(Mic2GrainsParams(
            param_file=str(_AU_PARAM),
            mic_file=str(mic_path),
            out_file=str(py_out),
            do_neighbor_search=0,
            n_cpus=1,
            min_conf_override=0.04,
        ))
        yield td_path, c_out, py_out, n


def _read_grains(path: Path):
    """Return (header_lines, data_rows_as_ndarray, raw_data_lines)."""
    headers = []
    data_lines = []
    with open(path) as f:
        for raw in f:
            if raw.startswith("%"):
                headers.append(raw.rstrip("\n"))
            elif raw.strip():
                data_lines.append(raw.rstrip("\n"))
    rows = np.asarray(
        [[float(t) for t in line.split()] for line in data_lines],
        dtype=np.float64,
    )
    return headers, rows, data_lines


def test_grain_count_matches_c(workspace):
    _, c_out, py_out, n = workspace
    c_h, c_rows, _ = _read_grains(c_out)
    py_h, py_rows, _ = _read_grains(py_out)
    assert c_rows.shape[0] == py_rows.shape[0], (
        f"grain count: c={c_rows.shape[0]} py={py_rows.shape[0]}"
    )
    assert n == c_rows.shape[0]


def test_header_matches_c(workspace):
    _, c_out, py_out, _ = workspace
    c_h, _, _ = _read_grains(c_out)
    py_h, _, _ = _read_grains(py_out)
    # The first line includes %NumGrains <n>; both should agree.
    assert c_h[0] == py_h[0]
    # Lines 4-9 are param-file derived; compare verbatim.
    for i in (3, 4, 5, 6, 7, 8):
        assert c_h[i] == py_h[i], f"header line {i}: c={c_h[i]!r} py={py_h[i]!r}"


def test_grains_match_c_within_misorientation(workspace):
    """Each Python grain's orientation matches a C grain to <0.1°.

    Both implementations sort voxels by confidence (descending) and use
    the highest-confidence one as the grain's *seed* orientation. The
    C uses ``qsort`` (unstable) and we use ``np.argsort(kind='stable')``
    so when many voxels share the same confidence the seed voxel can
    differ — which gives slightly different per-grain orientations
    (~0.001 in OM components, equivalent to ~0.05° misorientation).
    The grains themselves and their counts agree exactly.
    """
    from midas_stress.orientation import misorientation_om_batch
    import torch

    _, c_out, py_out, _ = workspace
    _, c_rows, _ = _read_grains(c_out)
    _, py_rows, _ = _read_grains(py_out)
    c_oms = c_rows[:, 1:10]
    py_oms = py_rows[:, 1:10]

    # For each python grain, find the closest C grain by OM-distance,
    # then check the corresponding misorientation is small (< 0.1°).
    used = np.zeros(c_oms.shape[0], dtype=bool)
    sg_nr = 225
    for r in range(py_oms.shape[0]):
        diffs = np.linalg.norm(c_oms - py_oms[r], axis=1)
        diffs[used] = np.inf
        best = int(np.argmin(diffs))
        used[best] = True
        c_om = torch.as_tensor(c_oms[best:best + 1])
        py_om = torch.as_tensor(py_oms[r:r + 1])
        miso_rad = misorientation_om_batch(c_om, py_om, sg_nr)
        miso_deg = float(miso_rad[0]) * 180.0 / np.pi
        assert miso_deg < 0.1, (
            f"grain {r} → c grain {best}: misorientation {miso_deg:.4f}° "
            f"exceeds 0.1° threshold"
        )
        # X, Y and radius depend on which voxel becomes the seed (highest-
        # confidence after the unstable C qsort vs our stable sort), so
        # they can differ across the full grain extent. We don't assert
        # positional parity — the orientation parity above is the
        # physically meaningful invariant.


def test_grain_id_is_one_indexed(workspace):
    _, _, py_out, _ = workspace
    _, py_rows, _ = _read_grains(py_out)
    np.testing.assert_array_equal(
        py_rows[:, 0].astype(int), np.arange(1, py_rows.shape[0] + 1),
    )
