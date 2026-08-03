"""Structure-factor weighting of the NF confidence.

Three metrics come from ONE hook, ``hard_fraction(refl_weight=...)``:
    ones      -> C_raw    (historical CalcFracOverlap)
    (f2 > 0)  -> C_filt   (basis-forbidden reflections dropped)
    f2        -> C_weight (weighted by |F|^2)

The regression that matters most is the first one: with no basis supplied,
nothing anywhere may change.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_nf_fitorientation.obs_volume import ObsVolume


def _volume(n_frames=4, n_y=8, n_z=8):
    dense = torch.zeros(1, n_frames, n_y, n_z, dtype=torch.uint8)
    return dense


def test_refl_weight_none_is_historical_behaviour():
    """Default path must be bit-identical to passing all-ones."""
    dense = _volume()
    dense[0, 1, 2, 3] = 1
    dense[0, 1, 4, 5] = 1
    ov = ObsVolume(dense=dense, device=torch.device("cpu"))

    frame = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    y = torch.tensor([[2.0, 4.0, 6.0, 7.0]])
    z = torch.tensor([[3.0, 5.0, 1.0, 1.0]])
    valid = torch.ones_like(frame)

    a = ov.hard_fraction(frame, y, z, valid)
    b = ov.hard_fraction(frame, y, z, valid, refl_weight=torch.ones(4))
    assert torch.equal(a, b)
    assert a.item() == pytest.approx(0.5)          # 2 of 4 matched


def test_filter_removes_forbidden_from_the_denominator():
    """Two of four reflections forbidden, and they are the two that miss.

    C_raw sees 2/4; C_filt sees 2/2 because the misses could never have been
    matched by any crystal.
    """
    dense = _volume()
    dense[0, 1, 2, 3] = 1
    dense[0, 1, 4, 5] = 1
    ov = ObsVolume(dense=dense, device=torch.device("cpu"))

    frame = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    y = torch.tensor([[2.0, 4.0, 6.0, 7.0]])
    z = torch.tensor([[3.0, 5.0, 1.0, 1.0]])
    valid = torch.ones_like(frame)

    f2 = torch.tensor([1.0, 1.0, 0.0, 0.0])        # last two forbidden
    raw = ov.hard_fraction(frame, y, z, valid)
    filt = ov.hard_fraction(frame, y, z, valid, refl_weight=(f2 > 0).double())

    assert raw.item() == pytest.approx(0.5)
    assert filt.item() == pytest.approx(1.0)


def test_weighting_favours_strong_reflections():
    """Matching the strong reflection scores higher than matching the weak one."""
    dense = _volume()
    dense[0, 1, 2, 3] = 1                          # only reflection 0 is lit
    ov = ObsVolume(dense=dense, device=torch.device("cpu"))

    frame = torch.tensor([[1.0, 1.0]])
    y = torch.tensor([[2.0, 6.0]])
    z = torch.tensor([[3.0, 1.0]])
    valid = torch.ones_like(frame)

    strong_first = torch.tensor([1.0, 0.1])
    weak_first = torch.tensor([0.1, 1.0])
    a = ov.hard_fraction(frame, y, z, valid, refl_weight=strong_first)
    b = ov.hard_fraction(frame, y, z, valid, refl_weight=weak_first)

    assert a.item() == pytest.approx(1.0 / 1.1)
    assert b.item() == pytest.approx(0.1 / 1.1)
    assert a.item() > b.item()


def test_all_forbidden_gives_zero_not_nan():
    """A zero denominator must not produce NaN."""
    dense = _volume()
    dense[0, 1, 2, 3] = 1
    ov = ObsVolume(dense=dense, device=torch.device("cpu"))
    frame = torch.tensor([[1.0, 1.0]])
    y = torch.tensor([[2.0, 6.0]])
    z = torch.tensor([[3.0, 1.0]])
    valid = torch.ones_like(frame)
    out = ov.hard_fraction(frame, y, z, valid, refl_weight=torch.zeros(2))
    assert torch.isfinite(out).all()
    assert out.item() == pytest.approx(0.0)


def test_hkltable_defaults_f2_to_ones():
    """A table built without F2 behaves as 'every reflection counts'."""
    from midas_nf_fitorientation.io import HKLTable

    t = HKLTable(
        hkls_int=np.zeros((5, 3)), hkls_cart=np.zeros((5, 3)),
        rings=np.arange(5), thetas_deg=np.zeros(5),
    )
    assert t.f2.shape == (5,)
    assert np.all(t.f2 == 1.0)


# --------------------------------------------------------------------------
#  spot_weights_from_f2: the 2-theta join that feeds screen()
# --------------------------------------------------------------------------

class _Orients:
    def __init__(self, spots):
        self.spots = np.asarray(spots, dtype=np.float64)


class _HKL:
    def __init__(self, thetas_deg, f2):
        self.thetas_deg = np.asarray(thetas_deg, dtype=np.float64)
        self.f2 = np.asarray(f2, dtype=np.float64)


LSD0 = 7228.584913


def _mk(thetas_deg, f2, spot_thetas_deg):
    """Build spots as (yl, zl, omega) -- the REAL DiffractionSpots.bin layout.

    yl/zl are lab-frame microns, so a spot on the ring of a reflection at
    ``theta`` sits at radius ``LSD0 * tan(2 theta)``. Building them as
    ``[2theta, eta, omega]`` is what the wrong docstring implied and is what
    made the join silently match nothing.
    """
    r = LSD0 * np.tan(np.radians(np.asarray(spot_thetas_deg) * 2.0))
    ang = np.linspace(0.3, 2.5, len(r))
    return _Orients(np.column_stack([
        r * np.cos(ang), r * np.sin(ang), np.zeros(len(r)),
    ])), _HKL(thetas_deg, f2)


def test_spot_weights_raw_is_none():
    from midas_nf_fitorientation.screen import spot_weights_from_f2
    o, h = _mk([1.0, 2.0], [1.0, 0.0], [2.0])
    assert spot_weights_from_f2(o, h, LSD0, metric="raw") is None


def test_spot_weights_none_when_no_basis_supplied():
    """All-ones F2 means hkls.csv had no F2 column -> nothing to weight."""
    from midas_nf_fitorientation.screen import spot_weights_from_f2
    o, h = _mk([1.0, 2.0], [1.0, 1.0], [1.0, 2.0])
    assert spot_weights_from_f2(o, h, LSD0, metric="weighted") is None


def test_spot_weights_filtered_zeroes_forbidden_rings():
    from midas_nf_fitorientation.screen import spot_weights_from_f2
    # ring A at 2theta=2 deg is allowed, ring B at 2theta=4 deg is forbidden
    o, h = _mk([1.0, 2.0], [1.0, 0.0],
               [1.0, 2.0, 1.0])
    w = spot_weights_from_f2(o, h, LSD0, metric="filtered")
    assert w.tolist() == [1.0, 0.0, 1.0]


def test_spot_weights_weighted_carries_f2():
    from midas_nf_fitorientation.screen import spot_weights_from_f2
    o, h = _mk([1.0, 2.0], [1.0, 0.25],
               [1.0, 2.0])
    w = spot_weights_from_f2(o, h, LSD0, metric="weighted")
    np.testing.assert_allclose(w.numpy(), [1.0, 0.25])


def test_confidence_metric_rejects_garbage():
    from midas_nf_fitorientation.screen import spot_weights_from_f2
    o, h = _mk([1.0], [0.5], [1.0])
    with pytest.raises(ValueError, match="raw|filtered|weighted"):
        spot_weights_from_f2(o, h, LSD0, metric="nonsense")


def test_no_ring_match_raises_instead_of_returning_neutral_weights():
    """A join that matches nothing must FAIL, not return all-ones.

    All-ones is indistinguishable from 'the weighting had no effect' and is
    exactly how the 2-theta-join bug hid: the run reproduced the unweighted
    numbers to 4 decimal places and looked like a legitimate null result.
    """
    from midas_nf_fitorientation.screen import spot_weights_from_f2
    o, h = _mk([1.0, 2.0], [1.0, 0.0], [1.0, 2.0])
    o.spots[:, 0] += 50_000.0                      # push every spot off-ring
    with pytest.raises(ValueError, match="matched no spots"):
        spot_weights_from_f2(o, h, LSD0, metric="weighted")


def test_refl_weight_accepts_foreign_device_tensor():
    """The weight arrives from numpy on the CPU while the volume may be on GPU.

    ``hard_fraction`` moves it; ``screen`` does the same once at entry. A
    missing ``.to(device)`` in screen surfaced only on real hardware as
    'Expected all tensors to be on the same device' -- this pins the contract
    on whatever devices are available here.
    """
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    elif torch.backends.mps.is_available():
        devices.append(torch.device("mps"))

    for dev in devices:
        dense = torch.zeros(1, 4, 8, 8, dtype=torch.uint8, device=dev)
        dense[0, 1, 2, 3] = 1
        ov = ObsVolume(dense=dense, device=dev)
        frame = torch.tensor([[1.0, 1.0]], device=dev)
        y = torch.tensor([[2.0, 6.0]], device=dev)
        z = torch.tensor([[3.0, 1.0]], device=dev)
        valid = torch.ones_like(frame)
        w_cpu = torch.tensor([1.0, 0.5])            # deliberately on CPU
        out = ov.hard_fraction(frame, y, z, valid, refl_weight=w_cpu)
        assert out.device.type == dev.type
        assert out.item() == pytest.approx(1.0 / 1.5)


def test_polish_hard_frac_accepts_a_reflection_weight():
    """The refine path must take the weight too, not just the search.

    Wiring refl_weight into screen() alone weights which orientation WINS but
    leaves the confidence written to the .mic unweighted -- so a weighted run
    reproduced the raw map exactly (max C 0.4938 both ways) and looked like
    'weighting has no effect' rather than 'half the wiring is missing'.
    """
    import inspect
    from midas_nf_fitorientation.hard_polish import polish_hard_frac

    sig = inspect.signature(polish_hard_frac)
    assert "refl_weight" in sig.parameters
    assert sig.parameters["refl_weight"].default is None   # opt-in


@pytest.mark.parametrize(
    "refine,device_type,has_triton,obs_packed",
    [
        ("nm-triton",  "cuda", True,  True),   # the unconditional-on path
        ("nm-triton",  "cpu",  False, False),  # nm-triton ignores the rest
        ("nm-batched", "cuda", True,  True),   # the conditional path
    ],
)
def test_triton_is_gated_off_whenever_a_weight_is_requested(
    refine, device_type, has_triton, obs_packed
):
    """fused_hard_frac has no weight input, so it must not run when weighting.

    Otherwise the refine silently computes an UNWEIGHTED fraction while the
    paramfile asks for a weighted one.

    Exercises the gate itself. The previous version of this test asserted on
    the literal source text of fit_orientation.py, which passed whether or not
    the gate worked and broke on a rename -- and needed a cwd-dependent path
    fallback to find the file at all.
    """
    from midas_nf_fitorientation.fit_orientation import should_use_triton

    kw = dict(device_type=device_type, has_triton=has_triton,
              obs_packed=obs_packed)
    # unweighted: each of these configurations WOULD use the fused kernel
    assert should_use_triton(refine, want_weight=False, **kw) is True
    # weighted: the same configuration must not
    assert should_use_triton(refine, want_weight=True, **kw) is False


def test_triton_gate_still_honours_the_non_weight_conditions():
    """The weight short-circuit must not paper over the original conditions."""
    from midas_nf_fitorientation.fit_orientation import should_use_triton

    base = dict(want_weight=False, device_type="cuda", has_triton=True,
                obs_packed=True)
    assert should_use_triton("nm-batched", **{**base, "device_type": "cpu"}) is False
    assert should_use_triton("nm-batched", **{**base, "has_triton": False}) is False
    assert should_use_triton("nm-batched", **{**base, "obs_packed": False}) is False
    assert should_use_triton("nm-serial", **base) is False
    assert should_use_triton("lbfgs+nm", **base) is False
