"""The parent must hand its cached CUDA memory back before sharded fit workers start.

Regression: an in-process reduction left 42.5 GB reserved on GPU 0 and the GPU-0
fit worker died with CUDA OOM (bt_20id_jul26b nf_sampleF, 2026-09-24).
"""
import logging
import sys
import types

from midas_nf_pipeline import stages


def _fake_torch(reserved_gib):
    calls = {"empty_cache": 0}
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: calls.__setitem__("empty_cache", calls["empty_cache"] + 1),
        device_count=lambda: len(reserved_gib),
        memory_reserved=lambda i: int(reserved_gib[i] * 2**30),
    )
    return types.SimpleNamespace(cuda=cuda), calls


def test_empties_cache_and_is_quiet_when_released(monkeypatch, caplog):
    fake, calls = _fake_torch([0.2, 0.0])
    monkeypatch.setitem(sys.modules, "torch", fake)
    with caplog.at_level(logging.WARNING):
        stages._release_gpu_memory("before sharded fit")
    assert calls["empty_cache"] == 1
    assert "still reserves" not in caplog.text


def test_warns_when_memory_is_still_held(monkeypatch, caplog):
    fake, calls = _fake_torch([42.5, 0.0])
    monkeypatch.setitem(sys.modules, "torch", fake)
    with caplog.at_level(logging.WARNING):
        stages._release_gpu_memory("before sharded fit")
    assert calls["empty_cache"] == 1
    assert "GPU 0" in caplog.text and "42.5 GiB" in caplog.text


def test_no_cuda_is_a_no_op(monkeypatch):
    fake, calls = _fake_torch([])
    fake.cuda.is_available = lambda: False
    monkeypatch.setitem(sys.modules, "torch", fake)
    stages._release_gpu_memory("x")
    assert calls["empty_cache"] == 0
