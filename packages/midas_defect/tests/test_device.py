"""midas_defect resolves devices precision-first: never MPS implicitly, float64 by default (the shared midas_transforms contract picked mps/float32 on a Mac)."""
import pytest
import torch

from midas_defect._device import resolve_device, resolve_dtype


def test_default_is_never_mps_even_when_mps_is_available(monkeypatch):
    monkeypatch.delenv("MIDAS_DEFECT_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device(None).type == "cpu"


def test_cuda_is_used_when_present_and_env_and_argument_override(monkeypatch):
    monkeypatch.delenv("MIDAS_DEFECT_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device(None).type == "cuda"
    monkeypatch.setenv("MIDAS_DEFECT_DEVICE", "cpu")
    assert resolve_device(None).type == "cpu"
    assert resolve_device("mps").type == "mps"                      # explicit request is honoured
    assert resolve_device(torch.device("cpu")).type == "cpu"


@pytest.mark.parametrize("dev", ["cpu", "cuda"])
def test_float64_by_default_on_every_device(dev, monkeypatch):
    monkeypatch.delenv("MIDAS_DEFECT_DTYPE", raising=False)
    assert resolve_dtype(torch.device(dev), None) == torch.float64


def test_dtype_argument_env_alias_and_errors(monkeypatch):
    monkeypatch.delenv("MIDAS_DEFECT_DTYPE", raising=False)
    cpu = torch.device("cpu")
    assert resolve_dtype(cpu, "float32") == torch.float32 and resolve_dtype(cpu, torch.float32) == torch.float32
    monkeypatch.setenv("MIDAS_DEFECT_DTYPE", "fp32")
    assert resolve_dtype(cpu, None) == torch.float32
    with pytest.raises(ValueError, match="Unsupported dtype"):
        resolve_dtype(cpu, "bfloat16")


def test_float64_on_mps_is_refused_with_an_explanation_float32_is_allowed_when_asked(monkeypatch):
    monkeypatch.delenv("MIDAS_DEFECT_DTYPE", raising=False)
    mps = torch.device("mps")
    with pytest.warns(RuntimeWarning, match="float32"):
        assert resolve_dtype(mps, None) == torch.float32              # an explicit mps request without a dtype is honoured, loudly
    with pytest.raises(ValueError, match="not available on the MPS backend"):
        resolve_dtype(mps, "float64")
    assert resolve_dtype(mps, "float32") == torch.float32


def test_the_geometry_default_runs_in_float64_on_cpu():
    """End to end: the pixel -> q map with no device/dtype arguments is float64 on CPU (it was float32 on MPS on a Mac)."""
    import numpy as np
    from midas_defect.geometry import Geometry, pixel_to_qlab
    g = Geometry(lsd_um=1.0e5, bcy_px=100.0, bcz_px=100.0, px_um=200.0, wavelength_A=0.4133, n_pix_y=200, n_pix_z=200, omega_first_deg=0.0, omega_step_deg=0.25, n_frames=10)
    q = pixel_to_qlab(np.array([50.0]), np.array([60.0]), g)
    assert q.dtype == torch.float64 and q.device.type == "cpu"
