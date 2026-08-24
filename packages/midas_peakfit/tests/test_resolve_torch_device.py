"""``cuda:N`` must reach the GPU, not silently fall back to the CPU.

midas_pipeline shards PF scans round-robin across GPUs, passing "cuda:0",
"cuda:1", ... into midas_peakfit. An exact ``device == "cuda"`` test rejected
all of them and fell through to CPU with the warning gated on the same exact
test, so the fallback was silent: peakfit used the GPU on a single-GPU host and
quietly ran on CPU on a multi-GPU host. These tests pin the resolution so the
regression cannot come back unnoticed.
"""

import torch

from midas_peakfit.orchestrator import resolve_torch_device


class _Avail:
    """Force torch.cuda.is_available() without needing a GPU."""

    def __init__(self, monkeypatch, value):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: value)


def test_sharded_cuda_strings_resolve_to_that_gpu(monkeypatch):
    _Avail(monkeypatch, True)
    for i in range(4):
        dev, _ = resolve_torch_device(f"cuda:{i}", "float64")
        assert dev.type == "cuda", f"cuda:{i} fell back to {dev}"
        assert dev.index == i, f"cuda:{i} resolved to index {dev.index}"


def test_plain_cuda_still_works(monkeypatch):
    _Avail(monkeypatch, True)
    dev, _ = resolve_torch_device("cuda", "float64")
    assert dev.type == "cuda"


def test_cuda_falls_back_to_cpu_when_unavailable(monkeypatch):
    _Avail(monkeypatch, False)
    dev, _ = resolve_torch_device("cuda:2", "float64")
    assert dev.type == "cpu"


def test_the_fallback_is_not_silent(monkeypatch, capsys):
    """The original bug was invisible: no warning on the sharded path."""
    _Avail(monkeypatch, False)
    resolve_torch_device("cuda:3", "float64")
    out = capsys.readouterr().out
    assert "cuda:3" in out and "falling back" in out.lower(), (
        f"sharded CUDA fallback printed nothing useful: {out!r}")


def test_cpu_request_is_quiet(monkeypatch, capsys):
    _Avail(monkeypatch, False)
    dev, dt = resolve_torch_device("cpu", "float64")
    assert dev.type == "cpu" and dt == torch.float64
    assert capsys.readouterr().out == ""


def test_dtype_mapping():
    dev, dt = resolve_torch_device("cpu", "float64")
    assert dt == torch.float64
    dev, dt = resolve_torch_device("cpu", "float32")
    assert dt == torch.float32


def test_cuda_keeps_float64(monkeypatch):
    """fp64 is the parity gate; CUDA must not silently downgrade it (MPS may)."""
    _Avail(monkeypatch, True)
    _, dt = resolve_torch_device("cuda:1", "float64")
    assert dt == torch.float64
