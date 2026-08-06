"""Reading externally-produced BCDI arrays.

The behaviour worth protecting is not "can it open a file" -- it is that the
loader refuses to guess when guessing would silently produce a plausible wrong
answer, and that the three array kinds each get the right amount of processing.
"""
import math

import numpy as np
import pytest
import torch

import midas_2d as m2d
from midas_2d.bcdi_io import BCDIData, load_bcdi

DT = torch.float64
SHAPE = (12, 10, 8)


@pytest.fixture
def psi():
    """A complex object with a non-trivial phase."""
    torch.manual_seed(0)
    s = torch.zeros(SHAPE, dtype=DT)
    s[3:8, 3:7, 2:6] = 1.0
    return torch.polar(s, torch.rand(SHAPE, dtype=DT) * s)


def _I(psi):
    A = m2d.object_to_amplitude(psi)
    return A.real * A.real + A.imag * A.imag


# ------------------------------------------------------------- the three kinds
@pytest.mark.unit
def test_each_kind_gets_the_right_amount_of_processing(psi):
    """object -> FFT then |.|^2 ; amplitude -> |.|^2 ; intensity -> unchanged."""
    A = m2d.object_to_amplitude(psi)
    I = _I(psi)

    from_obj = BCDIData(psi, kind="object").to_intensity()
    from_amp = BCDIData(A, kind="amplitude").to_intensity()
    from_int = BCDIData(I, kind="intensity").to_intensity()

    assert torch.allclose(from_obj, I, rtol=1e-10)
    assert torch.allclose(from_amp, I, rtol=1e-10)
    assert torch.allclose(from_int, I, rtol=1e-12)


@pytest.mark.unit
def test_object_and_amplitude_are_not_interchangeable(psi):
    """Mislabelling applies (or skips) a Fourier transform. The result looks
    perfectly plausible, which is exactly why the loader will not guess."""
    as_obj = BCDIData(psi, kind="object").to_intensity()
    as_amp = BCDIData(psi, kind="amplitude").to_intensity()
    x, y = as_obj.flatten(), as_amp.flatten()
    x, y = x - x.mean(), y - y.mean()
    assert float((x * y).sum() / (x.norm() * y.norm())) < 0.5


@pytest.mark.unit
def test_complex_file_without_kind_raises(tmp_path, psi):
    p = tmp_path / "obj.npy"
    np.save(p, psi.numpy())
    with pytest.raises(ValueError, match="ambiguous"):
        load_bcdi(p)
    assert load_bcdi(p, kind="object").kind == "object"


@pytest.mark.unit
def test_real_file_defaults_to_intensity(tmp_path, psi):
    p = tmp_path / "I.npy"
    np.save(p, _I(psi).numpy())
    d = load_bcdi(p)
    assert d.kind == "intensity"
    assert torch.allclose(d.to_intensity().to(DT), _I(psi), rtol=1e-6)


@pytest.mark.unit
def test_intensity_kind_rejects_complex_data(tmp_path, psi):
    p = tmp_path / "c.npy"
    np.save(p, psi.numpy())
    with pytest.raises(ValueError, match="complex but kind='intensity'"):
        load_bcdi(p, kind="intensity")


# ------------------------------------------------------------------ containers
@pytest.mark.unit
def test_roundtrip_npy(tmp_path, psi):
    p = tmp_path / "a.npy"
    np.save(p, psi.numpy())
    assert torch.allclose(load_bcdi(p, kind="object").array, psi, rtol=1e-12)


@pytest.mark.unit
def test_npz_picks_the_only_3d_array_and_complains_otherwise(tmp_path, psi):
    one = tmp_path / "one.npz"
    np.savez(one, data=psi.numpy(), scalar=np.array(3.0))
    assert load_bcdi(one, kind="object").array.shape == SHAPE

    two = tmp_path / "two.npz"
    np.savez(two, a=psi.numpy(), b=psi.numpy())
    with pytest.raises(ValueError, match="pass dataset="):
        load_bcdi(two, kind="object")
    assert load_bcdi(two, kind="object", dataset="b").array.shape == SHAPE


@pytest.mark.unit
def test_roundtrip_hdf5_and_cxi_default_path(tmp_path, psi):
    h5py = pytest.importorskip("h5py")
    I = _I(psi).numpy()

    p = tmp_path / "d.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("some/deep/place", data=I)
    with pytest.raises(ValueError, match="could not find|no 3-D"):
        load_bcdi(p)
    assert load_bcdi(p, dataset="some/deep/place").array.shape == SHAPE

    cxi = tmp_path / "d.cxi"
    with h5py.File(cxi, "w") as f:
        f.create_dataset("entry_1/data_1/data", data=I)
    assert load_bcdi(cxi).array.shape == SHAPE          # CXI default path found


@pytest.mark.unit
def test_list_datasets_reports_shapes(tmp_path, psi):
    h5py = pytest.importorskip("h5py")
    p = tmp_path / "d.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("a/b", data=_I(psi).numpy())
        f.create_dataset("flat", data=np.zeros(5))       # 1-D, must be ignored
    got = m2d.list_datasets(p)
    assert len(got) == 1 and "a/b" in got[0]


@pytest.mark.unit
def test_roundtrip_tiff_stack(tmp_path, psi):
    tifffile = pytest.importorskip("tifffile")
    p = tmp_path / "stack.tif"
    tifffile.imwrite(p, _I(psi).numpy().astype(np.float32))
    assert load_bcdi(p).array.shape == SHAPE


@pytest.mark.unit
def test_headerless_raw_requires_dtype_and_shape_and_validates_size(tmp_path, psi):
    I = _I(psi).numpy().astype(np.float32)
    p = tmp_path / "raw.bin"
    I.tofile(p)

    with pytest.raises(ValueError, match="headerless"):
        load_bcdi(p)
    assert load_bcdi(p, dtype="float32", shape=SHAPE).array.shape == SHAPE
    # a wrong dtype gives the wrong element count -- caught, not silently reshaped
    with pytest.raises(ValueError, match="Wrong dtype or wrong shape"):
        load_bcdi(p, dtype="float64", shape=SHAPE)


@pytest.mark.unit
def test_unsupported_container_and_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bcdi(tmp_path / "nope.npy")
    p = tmp_path / "x.docx"
    p.write_bytes(b"junk")
    with pytest.raises(ValueError, match="unsupported container"):
        load_bcdi(p)


# ------------------------------------------------------------------ conventions
@pytest.mark.unit
def test_permute_reaches_the_expected_axis_order(tmp_path, psi):
    """Rocking-first files are common; permute must fix the order on load."""
    rocking_first = psi.permute(2, 0, 1).contiguous()
    p = tmp_path / "rf.npy"
    np.save(p, rocking_first.numpy())
    d = load_bcdi(p, kind="object", transpose=(1, 2, 0))
    assert d.array.shape == SHAPE
    assert torch.allclose(d.array, psi, rtol=1e-12)


@pytest.mark.unit
def test_permute_rejects_a_non_permutation(psi):
    with pytest.raises(ValueError, match="permutation"):
        BCDIData(psi, kind="object").permute((0, 0, 1))


@pytest.mark.unit
def test_recenter_is_an_involution(psi):
    d = BCDIData(psi, kind="object", centered=True)
    back = d.recenter(False).recenter(True)
    assert back.centered is True
    assert torch.allclose(back.array, psi, rtol=1e-12)


@pytest.mark.unit
def test_centering_changes_the_transform(psi):
    """centered=True ifftshifts before the FFT; getting it wrong scrambles the
    phase, so the resulting intensity genuinely differs."""
    a = BCDIData(psi, kind="object", centered=True).to_intensity()
    b = BCDIData(psi, kind="object", centered=False).to_intensity()
    assert not torch.allclose(a, b, rtol=1e-3)
    assert float(a.sum()) == pytest.approx(float(b.sum()), rel=1e-9)   # Parseval


@pytest.mark.unit
def test_rejects_non_3d(psi):
    with pytest.raises(ValueError, match="3-D"):
        BCDIData(psi[0], kind="object")


@pytest.mark.unit
def test_summary_reports_shape_and_dynamic_range(psi):
    txt = BCDIData(_I(psi), kind="intensity").summary()
    assert "intensity" in txt and str(SHAPE[0]) in txt and "dynamic range" in txt


# ------------------------------------------------------------- into the chain
@pytest.mark.unit
def test_loaded_object_feeds_the_detector_chain(tmp_path, psi):
    """End to end: her file -> intensity -> detector rate -> counts."""
    p = tmp_path / "obj.npy"
    np.save(p, psi.numpy())
    data = load_bcdi(p, kind="object")
    rate = m2d.detector_signal(data.to_intensity(), structure_factor_sq=1.5,
                               photons_per_peak=1e4)
    assert float(rate.max()) == pytest.approx(1e4, rel=1e-9)
    counts = m2d.sample_counts(rate, generator=torch.Generator().manual_seed(0))
    assert torch.all(counts >= 0) and float(counts.sum()) > 0


@pytest.mark.autograd
def test_to_intensity_is_differentiable(psi):
    """Gradients reach an in-memory object through the loader container."""
    obj = psi.clone().requires_grad_(True)
    BCDIData(obj, kind="object").to_intensity().sum().backward()
    assert obj.grad is not None and torch.isfinite(obj.grad).all()
