"""find_grains' cross-voxel dedup follows the run's device and logs its path.
It used to read only MIDAS_FINDGRAINS_* env vars, so a CPU run on a GPU host
silently used GPU 0."""
from __future__ import annotations

import logging

import pytest

from midas_pipeline.find_grains import resolve_cluster_device

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("dev", ["cpu", "mps"])
def test_non_cuda_device_takes_reference_path(dev):
    method, d = resolve_cluster_device(dev, environ={})
    assert (method, d) == ("reference", None)


def test_cuda_without_a_gpu_falls_back_to_reference(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_cluster_device("cuda", environ={}) == ("reference", None)


def test_cuda_with_a_gpu_takes_gpu_path(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    method, d = resolve_cluster_device("cuda:1", environ={})
    assert method == "gpu" and str(d) == "cuda:1"


def test_env_override_wins():
    assert resolve_cluster_device(
        "cuda", environ={"MIDAS_FINDGRAINS_CLUSTER": "reference"})[0] == "reference"


def test_choice_is_logged(caplog):
    with caplog.at_level(logging.INFO, logger="midas_pipeline.find_grains"):
        resolve_cluster_device("cpu", environ={})
    assert "method=reference" in caplog.text and "device=cpu" in caplog.text
