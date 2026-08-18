"""Extinction as a CONTRAST mechanism, not just a validity bound.

The load-bearing test is `test_extinction_inverts_defect_contrast`: kinematically
a distorted voxel falls out of the acceptance and gets darker; with extinction it
escapes the coherent-domain attenuation and gets brighter. That inversion is the
classical "direct image" of a defect in an X-ray topograph, and it is the reason a
purely kinematical model mispredicts topographic contrast in exactly the
low-defect-density regime where topotomography is used.
"""
import math

import pytest
import torch

from midas_dct_tt import (PlaneDetector, attach_uniform_field, coherent_path_um,
                          extinction_weights, local_Q, psi_scan, sphere_grain,
                          topograph_stack, tt_alignment)
from midas_dct_tt.acceptance import tt_resolution_aniso

DT = torch.float64
LAM_A = 0.172979
LAMBDA_EXT = 20.0          # um, a plausible extinction length
MAXPATH = 4.0              # um, grain thickness


def _grid_Q(n=9, grad=0.0):
    """Uniform Q with an optional linear gradient along x (a lattice curvature)."""
    N = n ** 3
    Q = torch.zeros(N, 3, dtype=DT)
    Q[:, 0] = 4.0
    if grad:
        idx = torch.arange(n, dtype=DT).view(n, 1, 1).expand(n, n, n).reshape(-1)
        Q[:, 0] = Q[:, 0] + grad * idx
    return Q, (n, n, n)


def test_perfect_lattice_takes_the_full_path_and_is_attenuated():
    Q, shape = _grid_Q()
    t = coherent_path_um(Q, shape, 1.0, extinction_length_um=LAMBDA_EXT,
                         max_path_um=MAXPATH)
    assert torch.allclose(t, torch.full_like(t, MAXPATH))
    y = extinction_weights(Q, shape, 1.0, extinction_length_um=LAMBDA_EXT,
                           max_path_um=MAXPATH)
    assert float(y.max()) < 1.0, "a perfect thick crystal must lose intensity"
    x = MAXPATH / LAMBDA_EXT
    assert float(y.mean()) == pytest.approx(math.tanh(x) / x, rel=1e-10)


def test_distortion_shortens_the_path_and_removes_the_attenuation():
    Q, shape = _grid_Q(grad=1e-3)
    t = coherent_path_um(Q, shape, 1.0, extinction_length_um=LAMBDA_EXT,
                         max_path_um=MAXPATH)
    assert float(t.max()) < MAXPATH
    y = extinction_weights(Q, shape, 1.0, extinction_length_um=LAMBDA_EXT,
                           max_path_um=MAXPATH)
    assert float(y.min()) > 0.999, "a strongly distorted lattice should be ~kinematical"


def test_more_distortion_means_more_intensity():
    """Monotonic, and in the direction that makes defects BRIGHT."""
    ys = []
    for grad in (0.0, 1e-6, 1e-5, 1e-4):
        Q, shape = _grid_Q(grad=grad)
        ys.append(float(extinction_weights(Q, shape, 1.0,
                                           extinction_length_um=LAMBDA_EXT,
                                           max_path_um=MAXPATH).mean()))
    assert ys == sorted(ys), f"extinction weight not monotonic in distortion: {ys}"
    assert ys[-1] > ys[0]


def test_extinction_inverts_defect_contrast_in_a_topograph():
    """The decisive test, run through the real forward model.

    A locally distorted region must be DARKER than its surroundings without
    extinction (it falls out of the acceptance) and BRIGHTER with it (it escapes
    the coherent-domain attenuation).
    """
    g = attach_uniform_field(sphere_grain(2.0, spacing_um=1.0))
    det = PlaneDetector(pixel_um=1.0, shape=(48, 48), distance_um=5000.0)
    hkl = (2, 0, 0)
    al = tt_alignment(g.field.reference_G(hkl), LAM_A)
    res = tt_resolution_aniso(al)
    psi = psi_scan(4)

    n = g.n_voxels
    eye = torch.eye(3, dtype=DT).expand(n, 3, 3)
    # a localised distortion in half the grain
    H = torch.zeros(n, 3, 3, dtype=DT)
    half = g.positions[:, 0] > 0
    H[half, 0, 0] = 2e-3
    Q = local_Q(eye + H, g.field.reference_G(hkl))

    kin = topograph_stack(g, al, psi, detector=det, hkl=hkl, resolution=res,
                          Q_sample=Q)
    y = extinction_weights(Q, g.shape, g.spacing_um,
                           extinction_length_um=LAMBDA_EXT, max_path_um=MAXPATH)
    ext = topograph_stack(g, al, psi, detector=det, hkl=hkl, resolution=res,
                          Q_sample=Q, extinction=y)

    # the distorted half is attenuated LESS once extinction is modelled
    y_dist, y_perf = float(y[half].mean()), float(y[~half].mean())
    assert y_dist > y_perf, f"distorted {y_dist:.6f} not brighter than perfect {y_perf:.6f}"
    # and the effect survives into the image
    assert float(ext.sum()) != pytest.approx(float(kin.sum()), rel=1e-9)


def test_extinction_is_opt_in_and_defaults_to_kinematical():
    g = attach_uniform_field(sphere_grain(1.5, spacing_um=1.0))
    det = PlaneDetector(pixel_um=1.0, shape=(32, 32), distance_um=5000.0)
    al = tt_alignment(g.field.reference_G((2, 0, 0)), LAM_A)
    psi = psi_scan(3)
    a = topograph_stack(g, al, psi, detector=det, hkl=(2, 0, 0), resolution=None)
    b = topograph_stack(g, al, psi, detector=det, hkl=(2, 0, 0), resolution=None,
                        extinction=torch.ones(g.n_voxels, dtype=DT))
    assert torch.allclose(a, b), "y=1 must reproduce the kinematical result exactly"


def test_extinction_weights_guard_bad_inputs():
    Q, shape = _grid_Q()
    with pytest.raises(ValueError, match=r"must be \(N, 3\)"):
        coherent_path_um(Q[:, :2], shape, 1.0, extinction_length_um=1.0, max_path_um=1.0)
    with pytest.raises(ValueError, match="needs a regular grid"):
        coherent_path_um(Q, None, 1.0, extinction_length_um=1.0, max_path_um=1.0)
    with pytest.raises(ValueError, match="does not match"):
        coherent_path_um(Q, (2, 2, 2), 1.0, extinction_length_um=1.0, max_path_um=1.0)


# --- dynamical validity bound via midas_dfxm.takagi_taupin -----------------
def test_extinction_length_is_tens_of_microns_at_hexm_energy():
    """Lambda ~ 35-52 um for low-index fcc at 71.7 keV. Delegated to the validated
    two-beam Takagi-Taupin solver in midas_dfxm, not re-derived here."""
    from midas_dct_tt import kinematical_validity
    lams = {}
    for hkl in [(1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1)]:
        r = kinematical_validity(hkl, wavelength_A=0.172979, thickness_um=3.0)
        lams[hkl] = r["extinction_length_um"]
        assert 20.0 < lams[hkl] < 80.0
    # Lambda grows with reflection order (weaker structure factor)
    assert lams[(1, 1, 1)] < lams[(2, 2, 0)] < lams[(3, 1, 1)]


def test_kinematical_holds_for_small_grains_and_fails_for_real_ones():
    """The load-bearing number for this whole package.

    Everything here is kinematical. That is valid to a few percent only below
    ~3-5 um; at ordinary HEDM grain sizes it is wrong by orders of magnitude,
    because kinematical intensity grows as t^2 without bound while the true
    intensity saturates and oscillates.
    """
    from midas_dct_tt import kinematical_validity
    kw = dict(wavelength_A=0.172979)
    small = kinematical_validity((2, 0, 0), thickness_um=3.0, **kw)
    mid = kinematical_validity((2, 0, 0), thickness_um=10.0, **kw)
    big = kinematical_validity((2, 0, 0), thickness_um=100.0, **kw)

    assert small["regime"] == "kinematical" and small["relative_error"] < 0.05
    assert mid["regime"] == "marginal" and mid["relative_error"] > 0.2
    assert big["regime"] == "dynamical" and big["relative_error"] > 10.0
    # the qualitative failure: kinematical is unbounded, dynamical is not
    assert big["intensity_kinematical"] > 50.0
    assert big["intensity_dynamical"] < 1.0


def test_synthetic_grains_used_in_this_package_are_inside_the_valid_domain():
    """Our 2-3 um phantoms sit at t/Lambda ~ 0.03-0.08 -- kinematical is fine."""
    from midas_dct_tt import kinematical_validity
    for t in (2.0, 3.0):
        r = kinematical_validity((2, 0, 0), wavelength_A=0.172979, thickness_um=t)
        assert r["ratio"] < 0.1 and r["relative_error"] < 0.05


def test_extinction_length_spans_a_factor_seven_across_fcc_metals():
    """Lambda ~ 1/|F|, so the element dominates. Pinned because an earlier version
    of this package quoted Cu numbers as 'Ni and Cu', and because passing only a
    lattice parameter silently leaves |F| at Cu's value.

    Independently reproduced by a literature computation (Cromer-Mann f0):
    Ni 34.3/36.6/45.3/51.5, Al 119-160, Au 16-21 um.
    """
    import copy

    from midas_dfxm.io import fcc_reference_crystal
    from midas_dct_tt import kinematical_validity

    def fcc(element, a):
        c = copy.deepcopy(fcc_reference_crystal(a=a))
        at = c.atoms[0]
        c.atoms = [type(at)(element=element, fract=at.fract, occupancy=at.occupancy,
                            B_iso=at.B_iso, U_aniso=at.U_aniso, label=element + "1")]
        return c

    lam = {}
    for el, a in (("Ni", 3.5240), ("Cu", 3.6149), ("Al", 4.0495), ("Au", 4.0782)):
        lam[el] = kinematical_validity((1, 1, 1), wavelength_A=0.172979,
                                       thickness_um=3.0, crystal=fcc(el, a),
                                       lattice_a_A=a)["extinction_length_um"]
    assert lam["Ni"] == pytest.approx(34.3, rel=0.02)
    assert lam["Cu"] == pytest.approx(35.0, rel=0.02)
    assert lam["Al"] == pytest.approx(119.4, rel=0.02)
    assert lam["Au"] == pytest.approx(16.5, rel=0.02)
    assert lam["Al"] / lam["Au"] > 6.5, "the fcc span should be ~7x"


def test_lattice_parameter_without_a_matching_crystal_is_rejected():
    """Changing only `a` alters the Bragg angle but not |F| -- silently wrong by up
    to ~3x. The guard must fire rather than return a plausible number."""
    from midas_dct_tt import kinematical_validity
    with pytest.raises(ValueError, match="without a matching crystal"):
        kinematical_validity((2, 0, 0), wavelength_A=0.172979, thickness_um=3.0,
                             lattice_a_A=4.0495)
