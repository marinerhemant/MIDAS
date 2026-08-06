"""Smoke tests for the BCDI examples.

Nothing else in the suite imports `midas_2d.examples`, so a rename or a signature
change in the library would leave these silently broken until someone tried to
run one. These execute both at the smallest useful size and assert on the
scientific content, not just on "it did not raise".
"""
import os
import re

import pytest
import torch

from midas_2d.examples import tutorial_bcdi_forward as fwd
from midas_2d.examples import tutorial_bcdi_from_data as fromdata

DT = torch.float64


# --------------------------------------------------------------- the forward
@pytest.mark.unit
def test_forward_example_runs_and_all_checks_pass(tmp_path, capsys):
    out = fwd.main(str(tmp_path), n=32, grain_nm=400.0, figure=False)
    text = capsys.readouterr().out
    assert "ALL CHECKS PASSED" in text, text[-2500:]
    assert "FAIL" not in text
    assert os.path.exists(os.path.join(out, "bcdi_forward.pt"))


@pytest.mark.unit
def test_forward_example_phase_amplitude_is_what_was_asked_for():
    """The analytic field is parameterised by peak phase; it must deliver it.

    Regression: the field used to carry a raw coefficient that did not scale
    with grain size, giving 153 rad at 400 nm -- 24 phase wraps, far outside the
    regime the envelope model is valid in.
    """
    g = fwd.build_geometry(4000.0, (32, 32, 32))
    for target in (0.5, 2.0, 5.0):
        obj = fwd.build_object(g, grain_size_A=4000.0, peak_phase_rad=target)
        assert float(obj["phase"].abs().max()) == pytest.approx(target, rel=0.02)


@pytest.mark.unit
def test_forward_example_displacement_is_odd_so_the_sign_is_recoverable():
    """An even field on a centrosymmetric support makes the twin exact.

    Tested as ``u(-r) == -u(r)`` on explicit positions rather than by permuting
    array indices: the conjugate basis ``C`` is non-orthogonal, so the lab
    coordinate ``r_x`` varies along all three index axes and no single-axis flip
    corresponds to ``r -> -r``.
    """
    g = fwd.build_geometry(4000.0, (16, 16, 16))
    r = torch.tensor([[100.0, 50.0, -20.0], [-750.0, 10.0, 300.0],
                      [1900.0, -1200.0, 40.0], [0.0, 33.0, -7.0]], dtype=DT)
    u_plus = fwd.analytic_displacement(r, 4000.0, g["G"], 2.0)
    u_minus = fwd.analytic_displacement(-r, 4000.0, g["G"], 2.0)
    assert torch.allclose(u_plus, -u_minus, atol=1e-12), "field is not odd"
    assert float(u_plus.abs().max()) > 0
    # and it displaces along x only, so the phase is -G_x u_x
    assert torch.allclose(u_plus[:, 1:], torch.zeros_like(u_plus[:, 1:]), atol=1e-15)


@pytest.mark.unit
@pytest.mark.parametrize("shape", ["cuboid", "ellipsoid", "octahedron"])
def test_forward_example_supports_every_shape(shape):
    g = fwd.build_geometry(4000.0, (24, 24, 24))
    obj = fwd.build_object(g, grain_size_A=4000.0, shape=shape)
    assert 0 < int(obj["support"].sum()) < 24 ** 3


@pytest.mark.unit
def test_forward_example_dislocation_path_is_optional():
    """It must fall back cleanly when midas-dfxm is absent, not crash."""
    g = fwd.build_geometry(4000.0, (16, 16, 16))
    obj = fwd.build_object(g, grain_size_A=4000.0, use_dislocation=True)
    assert torch.isfinite(obj["psi"]).all()


# ------------------------------------------------------------- the data path
@pytest.mark.unit
def test_from_data_example_runs_both_entry_points(tmp_path, capsys):
    out = fromdata.main(str(tmp_path), n=16, object_nm=4.0)
    text = capsys.readouterr().out
    for expected in ("an array you already have", "atomic coordinates",
                     "cross-check", "gradient demo"):
        assert expected in text, text[-2500:]
    assert os.path.exists(os.path.join(out, "bcdi_from_file.pt"))
    assert os.path.exists(os.path.join(out, "bcdi_from_md.pt"))


@pytest.mark.unit
def test_from_data_cross_check_beats_both_controls(tmp_path, capsys):
    """The headline number must actually clear its controls.

    If the exact/envelope agreement did not beat both the inverted pattern and
    the shape-only object, the comparison would be evidence of nothing.
    """
    fromdata.main(str(tmp_path), n=24, object_nm=6.0)
    text = capsys.readouterr().out

    def grab(label):
        line = next(ln for ln in text.splitlines() if label in ln)
        m = re.search(r"[+-]\d+\.\d+", line)          # the trailing prose has no sign
        assert m, f"no correlation found on: {line!r}"
        return float(m.group())

    envelope = grab("corr(exact, envelope)")
    inverted = grab("corr(exact, inverted)")
    shape_only = grab("corr(exact, shape only)")
    assert envelope > 0.99
    assert envelope > inverted + 0.2
    assert envelope > shape_only + 0.1


@pytest.mark.unit
def test_invert_pattern_is_the_true_inversion_not_a_flip():
    """flip alone is off by one voxel on an even-length fftshifted array."""
    import midas_2d as m2d

    torch.manual_seed(0)
    s = torch.zeros(16, 16, 16, dtype=DT)
    s[4:11, 5:12, 3:10] = 1.0
    psi = torch.polar(s, torch.rand(16, 16, 16, dtype=DT) * s)

    def I(p):
        A = m2d.object_to_amplitude(p)
        return A.real ** 2 + A.imag ** 2

    ref, conj = I(psi), I(psi.conj())
    assert fromdata.ncc(conj, fromdata.invert_pattern(ref)) == pytest.approx(1.0, abs=1e-9)
    assert fromdata.ncc(conj, torch.flip(ref, dims=(0, 1, 2))) < 0.999


@pytest.mark.unit
def test_synthetic_frame_returns_reference_and_deformed():
    import math

    import midas_2d as m2d
    from midas_hkls import energy_eV_to_wavelength

    lam = float(energy_eV_to_wavelength(fromdata.ENERGY_EV))
    d = fromdata.A_AU / math.sqrt(3.0)
    R = m2d.rotation_to_bragg((2 * math.pi / fromdata.A_AU)
                              * torch.tensor([1.0, 1.0, 1.0], dtype=DT), lam, d)
    ref, deformed = fromdata.synthetic_frame(25.0, R)
    assert ref.shape == deformed.shape and ref.shape[0] > 100
    u = deformed - ref
    assert float(u.abs().max()) > 0
    assert torch.allclose(u[:, 1:], torch.zeros_like(u[:, 1:]), atol=1e-12)
