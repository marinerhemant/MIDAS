"""Determinism is a provenance requirement, not an option.

A peak fit that cannot be re-derived has no provenance: two runs of the same
data give different spots, and every downstream number — grain positions,
strain — silently inherits that. Before this was fixed the production path was
non-deterministic on GPU **always**, three ways at once: the flag defaulted to
False, midas_pipeline never passed it, and even when set it was gated on fp64
so a CUDA run (fp32 under --dtype auto) never got it.
"""

import inspect
import os

import pytest
import torch

from midas_peakfit.orchestrator import (
    NonDeterministicFit,
    enable_determinism,
    run,
)


@pytest.fixture(autouse=True)
def _restore_torch_state():
    """Determinism is global torch state — put it back for other tests."""
    was = torch.are_deterministic_algorithms_enabled()
    cfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    yield
    torch.use_deterministic_algorithms(was)
    if cfg is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cfg


def test_default_is_deterministic():
    """The default must be ON. Production callers do not pass this."""
    assert inspect.signature(run).parameters["deterministic"].default is True


def test_enabling_actually_turns_torch_determinism_on():
    torch.use_deterministic_algorithms(False)
    enable_determinism(True, torch.device("cpu"))
    assert torch.are_deterministic_algorithms_enabled()


def test_applies_on_cpu_regardless_of_dtype():
    """The old gate was `deterministic and dtype == float64`.

    CUDA defaults to fp32 under --dtype auto, so that gate excluded exactly the
    configuration that needed determinism most.
    """
    torch.use_deterministic_algorithms(False)
    enable_determinism(True, torch.device("cpu"))
    assert torch.are_deterministic_algorithms_enabled()


def test_cuda_sets_the_cublas_workspace():
    """cuBLAS GEMMs are not reproducible without a fixed workspace, and torch
    demands the variable be set before the first cuBLAS handle exists."""
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    enable_determinism(True, torch.device("cuda:0"))
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"


def test_existing_cublas_setting_is_respected():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    enable_determinism(True, torch.device("cuda:1"))
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_opting_out_is_loud(capsys):
    """Turning it off must be impossible to miss in a log."""
    enable_determinism(False, torch.device("cpu"))
    out = capsys.readouterr().out
    assert "NOT reproducible" in out and "WARNING" in out


def test_failure_raises_instead_of_printing(monkeypatch):
    """The old code printed and carried on — the run then looks fine and is not.

    If determinism was asked for and cannot be provided, that must stop the run.
    """
    def _boom(*a, **k):
        raise RuntimeError("no deterministic implementation")

    monkeypatch.setattr(torch, "use_deterministic_algorithms", _boom)
    with pytest.raises(NonDeterministicFit, match="provenance"):
        enable_determinism(True, torch.device("cpu"))
