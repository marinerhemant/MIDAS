"""First-touch CUDA init must be serialised across scan-worker threads.

midas_pipeline fans PF scans over a ThreadPoolExecutor with a different
``cuda:N`` per worker. torch initialises a device lazily on first use and that
path is not thread-safe against concurrent first touches of different devices;
the losers raise ``lazy wrapper should be called at most once``.

The failure was invisible: a failed scan is logged as a WARNING and the layer
finishes from the surviving scans, reporting "ok" with fewer voxels. On
bt_1id_jun25b s4/L1 (9 scans): 0 failed -> 61 voxels, 1 -> 45-52, 2 -> 20-30.

These tests run without a GPU by faking torch.cuda, so they check the
SERIALISATION CONTRACT rather than needing hardware.
"""

import threading

import pytest
import torch

import midas_peakfit.orchestrator as orch


def _fake_cuda(monkeypatch, on_warm=None):
    """Patch every call ensure_cuda_ready makes, so this runs without a GPU.

    Includes the LINALG warm-up: torch creates its cuSOLVER handles on the
    first linalg call, and that — not device init — is what actually raced.
    """
    calls = {"init": 0, "zeros": [], "chol": []}

    def _chol(t):
        calls["chol"].append(getattr(t, "dtype", None))
        if on_warm:
            on_warm()
        return (t, 0)

    monkeypatch.setattr(torch.cuda, "init", lambda: calls.__setitem__("init", calls["init"] + 1))
    monkeypatch.setattr(torch, "zeros", lambda *a, **k: calls["zeros"].append(k.get("device")))
    monkeypatch.setattr(torch, "eye", lambda n, device=None, dtype=None: _Tensor(dtype))
    monkeypatch.setattr(torch.linalg, "cholesky_ex", _chol)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda d=None: None)
    return calls


class _Tensor:
    def __init__(self, dtype):
        self.dtype = dtype


@pytest.fixture(autouse=True)
def _clear_ready():
    orch._CUDA_READY.clear()
    yield
    orch._CUDA_READY.clear()


def test_cpu_device_is_a_noop(monkeypatch):
    calls = _fake_cuda(monkeypatch)
    orch.ensure_cuda_ready(torch.device("cpu"))
    assert calls["init"] == 0 and calls["chol"] == []


def test_each_device_is_warmed_exactly_once(monkeypatch):
    calls = _fake_cuda(monkeypatch)
    for _ in range(3):
        orch.ensure_cuda_ready(torch.device("cuda:2"))
    assert len(calls["zeros"]) == 1, "device warmed more than once"
    assert orch._CUDA_READY == {2}


def test_linalg_backend_is_warmed_in_both_dtypes(monkeypatch):
    """The actual bug: cuSOLVER handles are created on the first linalg call.

    Warming the device with torch.zeros alone left this path cold, so the race
    survived the first attempt at the fix.
    """
    calls = _fake_cuda(monkeypatch)
    orch.ensure_cuda_ready(torch.device("cuda:0"))
    assert set(calls["chol"]) == {torch.float32, torch.float64}, calls["chol"]


def test_distinct_devices_are_each_warmed(monkeypatch):
    calls = _fake_cuda(monkeypatch)
    zeros = calls["zeros"]
    for i in range(4):
        orch.ensure_cuda_ready(torch.device(f"cuda:{i}"))
    assert orch._CUDA_READY == {0, 1, 2, 3}
    assert len(zeros) == 4


def test_concurrent_first_touch_is_serialised(monkeypatch):
    """The actual regression: 4 threads, 4 devices, all at once.

    ``overlap`` counts how many warm-ups are ever inside the critical section
    simultaneously. It must never exceed 1 — that is what the real torch lazy
    init could not tolerate.
    """
    inside = 0
    overlap = 0
    guard = threading.Lock()

    def on_warm():
        nonlocal inside, overlap
        with guard:
            inside += 1
            overlap = max(overlap, inside)
        threading.Event().wait(0.01)   # widen the window a real race exploits
        with guard:
            inside -= 1

    _fake_cuda(monkeypatch, on_warm=on_warm)

    barrier = threading.Barrier(4)

    def worker(i):
        barrier.wait()          # maximise the chance of a simultaneous touch
        orch.ensure_cuda_ready(torch.device(f"cuda:{i}"))

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    assert overlap == 1, f"{overlap} concurrent CUDA first-touches — race not fixed"
    assert orch._CUDA_READY == {0, 1, 2, 3}


def test_bare_cuda_string_maps_to_device_zero(monkeypatch):
    _fake_cuda(monkeypatch)
    orch.ensure_cuda_ready(torch.device("cuda"))
    assert orch._CUDA_READY == {0}
