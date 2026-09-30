"""The C refiner must survive a per-seed FitBest_*.csv it cannot open.

FitUnified.c writes one ``Results/FitBest_<vox>_<SpId>.csv`` per seed inside
an ``omp critical`` block. It checked ``fopen`` for NULL, printed a message,
and then called ``fclose(outF)`` unconditionally -- ``fclose(NULL)`` is a
SIGSEGV in glibc. Because stdout is a pipe the message was still in the stdio
buffer when the process died, so the run ended with exit -11 and no
diagnostic at all.

Seen on a production Linux host, 2026-09-27 (a crack-tip FF run, 61.5k seeds, n_cpus 16):
three production crashes, all three systemd-coredump cores fault at
``fclose+11`` with ``rdi = 0`` (the FILE*), the call site is the
``fclose`` immediately before ``GOMP_critical_end``, and errno in the faulting
thread is 13 (EACCES) -- a transient open failure on a network filesystem, which is
why the crash point moved between runs and one run finished cleanly.

This test forces the same failure deterministically: a directory sits at the
path of the seed's CSV, so ``fopen(..., "w")`` fails with EISDIR. Before the
fix the binary exits -11; after it, it exits 0, reports the failure on
stderr, and still writes the seed's binary outputs. (macOS libc tolerates
``fclose(NULL)``, so there the old binary exits 0 and the test fails only on
the missing stderr report; the -11 reproduces on glibc.)

The binary under test is ``$MIDAS_FITGRAIN_BIN`` when set (a build of this
tree), else the installed ``midas_fit_grain`` binary.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from midas_fit_grain import backend_c


def _binary() -> Path | None:
    env = os.environ.get("MIDAS_FITGRAIN_BIN")
    if env:
        return Path(env)
    return backend_c.binary_path() if backend_c.available() else None


BIN = _binary()
pytestmark = pytest.mark.skipif(
    BIN is None or not BIN.is_file(),
    reason="midas_fitgrain C binary not available",
)

A = 3.5912
WL = 0.172978
LSD = 900000.0
N_SPOTS = 6


def _consolidated(path: Path, per_voxel: list[bytes], counts: list[int]) -> None:
    """[int32 nV][int32 count[nV]][int64 absolute offset[nV]] + data."""
    nv = len(per_voxel)
    header = 4 + 4 * nv + 8 * nv
    offs, pos = [], header
    for blob in per_voxel:
        offs.append(pos)
        pos += len(blob)
    with open(path, "wb") as f:
        f.write(np.int32(nv).tobytes())
        f.write(np.asarray(counts, np.int32).tobytes())
        f.write(np.asarray(offs, np.int64).tobytes())
        for blob in per_voxel:
            f.write(blob)


def _make_ff_layer(layer: Path) -> None:
    out, res = layer / "Output", layer / "Results"
    out.mkdir(parents=True)
    res.mkdir()

    d111 = A / np.sqrt(3.0)
    theta = np.degrees(np.arcsin(WL / (2 * d111)))
    radius = LSD * np.tan(np.radians(2 * theta))
    lines = ["h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius"]
    for h in (-1, 1):
        for k in (-1, 1):
            for l in (-1, 1):
                lines.append(f"{h} {k} {l} {d111} 1 0 0 0 {theta} {2 * theta} {radius}")
    (layer / "hkls.csv").write_text("\n".join(lines) + "\n")

    # ExtraInfo.bin: 16 doubles per spot; the values only need to be finite.
    rng = np.random.default_rng(0)
    ei = np.zeros((N_SPOTS, 16))
    eta = rng.uniform(-170, 170, N_SPOTS)
    ome = rng.uniform(-170, 170, N_SPOTS)
    y, z = -radius * np.sin(np.radians(eta)), radius * np.cos(np.radians(eta))
    ei[:, 0], ei[:, 1], ei[:, 2] = y, z, ome
    ei[:, 3] = 30.0                       # grain radius
    ei[:, 4] = np.arange(1, N_SPOTS + 1)  # SpotID (1-based row)
    ei[:, 5] = 1                          # RingNr
    ei[:, 8], ei[:, 9], ei[:, 10] = ome, y, z
    ei.tofile(layer / "ExtraInfo.bin")

    # One seed, one solution: identity orientation at the origin.
    sol = np.zeros(16)
    sol[2:11] = np.eye(3).ravel()
    sol[14] = 2 * N_SPOTS                 # NrExpected
    sol[15] = N_SPOTS                     # NrObserved == nSpotsBest
    _consolidated(out / "IndexBest_all.bin", [sol.tobytes()], [1])
    ids = np.arange(1, N_SPOTS + 1, dtype=np.int32)
    _consolidated(out / "IndexBest_IDs_all.bin", [ids.tobytes()], [N_SPOTS])

    (layer / "SpotsToIndex.csv").write_text("1\n")
    (layer / "paramstest.txt").write_text(
        f"LatticeParameter {A} {A} {A} 90 90 90;\n"
        f"Wavelength {WL};\n"
        f"Distance {LSD};\n"
        f"MaxRingRad {2 * radius};\n"
        "Rsample 2000;\nHbeam 2000;\npx 150;\nSpaceGroup 225;\n"
        "ExcludePoleAngle 6;\nRingNumbers 1;\n"
        f"RingRadii {radius};\n"
        "Wedge 0;\nOmegaRange -180 180;\n"
        "BoxSize -1000000 1000000 -1000000 1000000;\n"
        "MargABC 2.5;\nMargABG 2.5;\nOmeBinSize 0.1;\nEtaBinSize 0.1;\n"
        f"OutputFolder {out}\n"
        f"ResultFolder {res}\n"
    )


def _run(layer: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [str(BIN), str(layer / "paramstest.txt"), "0", "1", "1", "1"],
        cwd=layer, capture_output=True, timeout=120,
    )


def test_writable_results_dir_runs_clean(tmp_path):
    """Control: the fixture itself refines without error."""
    layer = tmp_path / "layer"
    _make_ff_layer(layer)
    proc = _run(layer)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert (layer / "Results" / "FitBest_000000_000000001.csv").is_file()


def test_unopenable_fitbest_csv_does_not_segfault(tmp_path):
    layer = tmp_path / "layer"
    _make_ff_layer(layer)
    blocker = layer / "Results" / "FitBest_000000_000000001.csv"
    blocker.mkdir()  # fopen(blocker, "w") -> EISDIR, like the transient EACCES on a network filesystem

    proc = _run(layer)
    assert proc.returncode != -11, "refiner died with SIGSEGV (fclose(NULL))"
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    err = proc.stderr.decode(errors="replace")
    assert "FitBest_000000_000000001.csv" in err
    # The seed's binary outputs are written before the CSV and must survive.
    opf = layer / "Results" / "OrientPosFit.bin"
    assert opf.is_file() and opf.stat().st_size > 0
