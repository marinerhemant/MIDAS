"""``--peak-fit-gpu`` must actually reach the peak-fit backend.

The flag was parsed by ``cli.py`` and stored on ``PipelineConfig`` but read by
no stage, so it silently did nothing: a run asking for GPU peak fitting landed
on CPU, on 4-GPU hosts included. A dead flag is worse than a missing one -- it
reports success while doing the opposite -- so this pins the wiring.
"""

from __future__ import annotations

from types import SimpleNamespace

from midas_pipeline.stages.peakfit import _peakfit_device


def _cfg(device: str, peak_fit_gpu: bool = False):
    return SimpleNamespace(device=device, peak_fit_gpu=peak_fit_gpu)


def test_flag_promotes_cpu_run_to_gpu():
    assert _peakfit_device(_cfg("cpu", peak_fit_gpu=True)) == "cuda"


def test_flag_absent_leaves_cpu_alone():
    assert _peakfit_device(_cfg("cpu")) == "cpu"


def test_explicit_cuda_device_is_not_overwritten():
    # --device already names a specific GPU; the flag must not collapse it to
    # bare "cuda", which is what selects round-robin fan-out in _run_pf.
    assert _peakfit_device(_cfg("cuda:2", peak_fit_gpu=True)) == "cuda:2"
    assert _peakfit_device(_cfg("cuda:2")) == "cuda:2"


def test_flag_never_removes_gpu():
    assert _peakfit_device(_cfg("cuda", peak_fit_gpu=False)) == "cuda"


def test_empty_device_defaults_to_cpu():
    assert _peakfit_device(_cfg("")) == "cpu"
    assert _peakfit_device(_cfg("", peak_fit_gpu=True)) == "cuda"


def test_missing_attribute_is_tolerated():
    # Older configs / partial namespaces must not raise here: peakfit is the
    # long pole and this helper runs before any work is done.
    assert _peakfit_device(SimpleNamespace(device="cpu")) == "cpu"
