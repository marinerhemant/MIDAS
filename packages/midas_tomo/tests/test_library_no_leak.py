"""The in-process library backend must not leak memory per call.

``createPlanFile()`` and the OpenMP workers in ``tomo_init.c`` allocate a
full per-thread buffer set (the gridrec tables, the padded sinogram and
reconstruction buffers, ``readStruct.norm_sino``) on every call. Before the
fix, only part of that set was released, so every ``run_tomo`` /
``run_tomo_from_sinos`` call through the library backend grew the process by
a few hundred MB at a 1024-wide padded detector and by several GB at 4096.
The subprocess backend never showed it because the OS reclaims everything at
process exit; an in-process caller that reconstructs many channels (midas-dt)
did.

The check is on process RSS, because that is what the leak was observed as.
Measured on a Linux host (pocketfft engine, 2 threads, 2 slices, 90 angles,
1000 detector columns padded to 1024), the unpatched library grows by about
129 MB per call in the single-shift path and 133 MB per call in the 3-shift
path; the fixed library grows by well under 1 MB per call. The bound below is 0.1 x the unpatched growth, so it
fails on a leak of that size and has two orders of margin over the noise.

Skipped when the library is not built. Needs ``/proc`` or a ``ps`` that
accepts ``-o rss=``; skipped otherwise.
"""

from __future__ import annotations

import gc
import hashlib
import os
import subprocess

import numpy as np
import pytest

from midas_tomo import backend_lib, run_tomo_from_sinos

pytestmark = pytest.mark.needs_binary

if not backend_lib.available():
    pytest.skip(
        f"libmidastomo not loadable: {backend_lib.why_unavailable()}",
        allow_module_level=True,
    )

N_CALLS = 5
#: Per-call RSS growth of the UNFIXED library, MB, for the cases below
#: (measured; see module docstring). The assertion allows 10% of this.
UNFIXED_GROWTH_MB = {"single_shift": 129.0, "multi_shift": 133.0}
ALLOWED_FRACTION = 0.1


def _rss_mb() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            check=True, capture_output=True, text=True,
        ).stdout
        return int(out.split()[0]) / 1024.0
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        pytest.skip("cannot read process RSS on this platform")


def _run_repeatedly(tmp_path, shifts):
    rng = np.random.default_rng(0)
    # 1000 columns pad to 1024; 2 slices; 90 angles. Small enough to be fast,
    # large enough that one leaked buffer set is visible in RSS.
    sino = rng.normal(size=(2, 90, 1000)).astype(np.float32)
    thetas = np.linspace(0.0, 180.0, 90, endpoint=False)
    # Keep only a digest per call: holding the cubes would itself grow RSS by
    # the size of the output each call and mask the quantity being measured.
    rss, digests, shapes, finite = [], [], [], []
    for i in range(N_CALLS):
        cube = run_tomo_from_sinos(
            sino, tmp_path / f"wd{i}", thetas, shifts=shifts, n_cpus=2,
            do_log=False, do_cleanup=False, deterministic=True,
            backend="library",
        )
        digests.append(hashlib.sha256(np.ascontiguousarray(cube)).hexdigest())
        shapes.append(cube.shape)
        finite.append(bool(np.isfinite(cube).all()))
        del cube
        gc.collect()
        rss.append(_rss_mb())
    return rss, digests, shapes, finite


@pytest.mark.parametrize(
    "case, shifts",
    [
        ("single_shift", 0.0),
        ("multi_shift", (-1.0, 1.0, 1.0)),
    ],
)
def test_library_calls_do_not_grow_the_process(tmp_path, case, shifts):
    rss, digests, shapes, finite = _run_repeatedly(tmp_path, shifts)
    # Calls 0 and 1 pay one-off costs (thread pools, lazily mapped pages, the
    # allocator's high-water mark: the fixed library still rises ~20 MB at
    # call 1 in the 3-shift case and is flat afterwards), so the growth is
    # measured from the end of call 1.
    per_call = (rss[-1] - rss[1]) / (N_CALLS - 2)
    bound = ALLOWED_FRACTION * UNFIXED_GROWTH_MB[case]
    assert per_call < bound, (
        f"{case}: RSS grew {per_call:.1f} MB per call over {N_CALLS - 2} calls "
        f"(bound {bound:.1f} MB; the unfixed library leaks about "
        f"{UNFIXED_GROWTH_MB[case]:.0f} MB per call here). RSS after each call, "
        f"MB: {[round(r, 1) for r in rss]}"
    )

    # The fix only releases memory; it must not change the answer, and a
    # repeated call on the same input must reproduce the first bit for bit.
    assert shapes[0] == shapes[-1]
    assert digests[0] == digests[-1]
    assert all(finite)
