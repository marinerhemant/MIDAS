"""Per-grain strain in the grain-tx geometry fit — MIDAS issue #70.

Two compounding defects were reported and are both covered here.

1. ``grain_lattice_from_reference`` tiled the NOMINAL header lattice across
   every grain, so a fit both started from and stayed at the nominal lattice
   even though ``Grains.csv`` carries each grain's own refined one. On
   shade_LSHR layer 1 (Sparks 2024 EBSD-comparison data, 4312 grains) those
   differ by a common +1100 µε with ~210 µε of shear on top.
2. ``refine_geometry_from_grains`` passed a hardcoded ``refine_grain_strain=
   False`` into ``build_joint_spec``, so its own parameter, its default, its
   docstring and the ``--no-strain`` flag were all dead. The reported symptom
   was two ``--fix grain_lattice`` runs producing byte-identical output.

The fix refines strain through a dimensionless ``grain_strain`` block rather
than by thawing ``grain_lattice``; ``test_thawing_the_lattice_destroys_the_
angles`` is the reason, and it is the test to read first.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import midas_peakfit as mp
from midas_peakfit import Parameter
from midas_peakfit.pack import pack_spec, refined_bounds, refined_indices
from midas_peakfit.reparam import u_to_x, x_to_u

from midas_calibrate.geometry_torch import build_tilt_matrix_torch
from midas_diffract import HEDMForwardModel
from midas_diffract.forward import HEDMGeometry
from midas_diffract.hkls import hkls_for_forward_model
from midas_fit_grain.matching import MatchResult
from midas_fit_grain.observations import ObservedSpots
from midas_hkls import Lattice, SpaceGroup

from midas_joint_ff_calibrate.grain_observations import (
    grain_lattice_from_reference,
    grain_lattices_for_fit,
    load_phase2_grains_and_spots,
)
from midas_joint_ff_calibrate.grain_refine import make_residual
from midas_joint_ff_calibrate.spec import build_joint_spec

DT = torch.float64
LSD, BCY, BCZ, PX = 1.0e6, 1024.0, 1024.0, 150.0
RHOD, NPIX = 1024.0 * PX, 2048
A0 = 3.585                      # shade_LSHR FCC


# ════════════════════════════════════════════ 1. the parameterisation itself

def _spec_with_grains(n_g=3, refine_strain=False, **kw):
    lat = torch.tensor(np.tile([A0, A0, A0, 90.0, 90.0, 90.0], (n_g, 1)))
    return build_joint_spec(
        powder_spec=mp.ParameterSpec(),
        grain_eulers_init=torch.zeros(n_g, 3, dtype=DT),
        grain_positions_init=torch.zeros(n_g, 3, dtype=DT),
        grain_lattices_init=lat,
        refine_grain_orientation=False, refine_grain_position=False,
        refine_grain_strain=refine_strain, **kw)


@pytest.mark.parametrize("refine_strain", [False, True])
def test_the_flag_reaches_the_spec(refine_strain):
    """The whole of issue #70 in one assertion: asking for strain gets strain.

    Before the fix ``build_joint_spec`` was called with a literal ``False``
    from ``refine_geometry_from_grains``, so this was False in both arms.
    """
    spec = _spec_with_grains(refine_strain=refine_strain)
    assert spec.parameters["grain_strain"].refined is refine_strain
    # ... and the seed lattice is frozen either way (see the next test).
    assert spec.parameters["grain_lattice"].refined is False


def test_grain_strain_is_bounded_and_starts_at_zero():
    spec = _spec_with_grains(n_g=4, refine_strain=True)
    p = spec.parameters["grain_strain"]
    assert p.bounds == (-0.02, 0.02)
    assert torch.equal(p.init, torch.zeros(4, 6, dtype=torch.float64))
    assert _spec_with_grains(refine_strain=True, strain_bound=5e-3
                             ).parameters["grain_strain"].bounds == (-5e-3, 5e-3)
    with pytest.raises(ValueError, match="strain_bound"):
        _spec_with_grains(refine_strain=True, strain_bound=0.0)


def test_thawing_the_lattice_destroys_the_angles():
    """Why strain is refined as ``grain_strain`` and not by thawing
    ``grain_lattice`` — the fix proposed on the issue.

    ``lm_minimise`` boxes each refined parameter with ONE ``(lo, hi)`` for the
    whole tensor, and with no declared bounds fabricates it from
    ``init.flatten()[0] ± fallback_span`` — the FIRST element, i.e. ``a``. The
    logit transform then clamps every angle into the ``a`` box on the way in.
    A (N,6) lattice cannot be boxed; a dimensionless strain can.
    """
    lat = torch.tensor(np.tile([A0, A0, A0, 90.0, 90.0, 90.0], (2, 1)))

    def roundtrip(spec):
        x, info = pack_spec(spec)
        lo, hi = refined_bounds(spec, info, fallback_span=2.0)
        idx = refined_indices(info)
        xr = x.index_select(0, idx)
        u = x_to_u(xr.unsqueeze(0), lo.unsqueeze(0), hi.unsqueeze(0))
        return u_to_x(u, lo.unsqueeze(0), hi.unsqueeze(0)).squeeze(0)

    naive = mp.ParameterSpec()
    naive.add(Parameter("grain_lattice", init=lat.clone(), refined=True))
    got = roundtrip(naive).view(2, 6)
    assert got[0, 0] == pytest.approx(A0)               # a survives
    assert got[0, 3] == pytest.approx(A0 + 2.0, abs=1e-4)  # 90 deg -> 5.585
    assert abs(float(got[0, 3]) - 90.0) > 80.0, (
        "expected the angle to be silently clamped into the a-box")

    # The shipped spec does not do that: the lattice is frozen, and the free
    # block is the strain, whose own box is centred on its init.
    spec = _spec_with_grains(n_g=2, refine_strain=True)
    assert torch.allclose(roundtrip(spec).view(2, 6),
                          torch.zeros(2, 6, dtype=torch.float64))
    assert torch.allclose(spec.parameters["grain_lattice"].init, lat)


# ════════════════════════════════════════════════════ 2. the per-grain seed

_PREAMBLE = (
    "%NumGrains 3\n%BeamCenter 0.0 0.0\n%BeamThickness 0.0\n"
    "%GlobalPosition 0.0\n%NumPhases 1\n%PhaseInfo\n%\tSpaceGroup:225\n"
    f"%\tLattice Parameter:{A0} {A0} {A0} 90.000000 90.000000 90.000000\n"
)
_COLS_53 = (
    ["O11", "O12", "O13", "O21", "O22", "O23", "O31", "O32", "O33",
     "X", "Y", "Z", "a", "b", "c", "alpha", "beta", "gamma",
     "DiffPos", "DiffOme", "DiffAngle", "GrainRadius", "Confidence"]
    + [f"eFab{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
    + [f"eKen{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
    + ["RMSErrorStrain", "PhaseNr", "Eul0", "Eul1", "Eul2",
       "DiffPosPre", "DiffOmePre", "DiffAnglePre",
       "DiffPosPost", "DiffOmePost", "DiffAnglePost"])
_OM = [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]

#: Row 2 is deliberately unusable (a = 0), row 3 deliberately implausible
#: (a 10 % length change is a parsing failure, not a strained grain).
_LATTICES = (
    [3.588913, 3.588981, 3.588995, 90.0002, 89.9999, 90.0002],   # real, strained
    [0.0, 0.0, 0.0, 90.0, 90.0, 90.0],
    [3.9, 3.9, 3.9, 90.0, 90.0, 90.0],
)


def _grains_csv(lattices=_LATTICES) -> str:
    hdr = "%ID\t" + "\t".join(_COLS_53) + "\n"
    rows = []
    for gid, lat in enumerate(lattices, start=1):
        vals = ([gid] + _OM + [0.0, 0.0, 0.0] + list(lat)
                + [10.0, 0.1, 0.15, 12.0, 0.9] + [0.0] * 18
                + [12.5, 1, 0.1, 0.2, 0.3, 10.0, 0.1, 0.15, 10.0, 0.1, 0.15])
        rows.append("\t".join(str(v) for v in vals))
    return _PREAMBLE + hdr + "\n".join(rows) + "\n"


def _grains_dict(path):
    from midas_joint_ff_calibrate.grain_observations import load_grains_csv
    return load_grains_csv(path)


def test_seed_is_each_grains_own_lattice(tmp_path):
    """The seed comes from ``Grains.csv`` cols a..gamma, not the header."""
    p = tmp_path / "Grains.csv"
    p.write_text(_grains_csv())
    g = _grains_dict(p)
    seed = grain_lattices_for_fit(g)
    np.testing.assert_allclose(seed[0], _LATTICES[0], rtol=0, atol=1e-9)
    # ... and it really is different from what the old function returns.
    ref = grain_lattice_from_reference(g)
    assert abs(seed[0, 0] - ref[0, 0]) > 1e-4


def test_unusable_rows_fall_back_per_row(tmp_path):
    """A zero lattice and a 10 %-off lattice are parsing failures, not
    strain: those rows take the header value, the good row keeps its own."""
    p = tmp_path / "Grains.csv"
    p.write_text(_grains_csv())
    seed = grain_lattices_for_fit(_grains_dict(p))
    np.testing.assert_allclose(seed[1], [A0, A0, A0, 90, 90, 90])
    np.testing.assert_allclose(seed[2], [A0, A0, A0, 90, 90, 90])
    np.testing.assert_allclose(seed[0], _LATTICES[0], atol=1e-9)


def test_header_source_reproduces_the_old_behaviour(tmp_path):
    p = tmp_path / "Grains.csv"
    p.write_text(_grains_csv())
    g = _grains_dict(p)
    np.testing.assert_array_equal(
        grain_lattices_for_fit(g, prefer_per_grain=False),
        grain_lattice_from_reference(g))


def test_legacy_21_column_file_has_no_per_grain_lattice():
    """cols 13-18 of a genuine 21-column file are the Voigt STRAIN, so the
    reader reports None and the seed must fall back rather than read them."""
    g = {"n_grains": 2, "lattice": (A0, A0, A0, 90.0, 90.0, 90.0),
         "lattice_per_grain": None}
    np.testing.assert_array_equal(grain_lattices_for_fit(g),
                                  grain_lattice_from_reference(g))


def test_loader_lattice_source_is_validated(tmp_path):
    with pytest.raises(ValueError, match="lattice_source"):
        load_phase2_grains_and_spots(tmp_path, lattice_source="nominal")


# ═══════════════════════════════════════════ 3. does it change the answer?

def _model():
    sg = SpaceGroup.from_number(225)
    lat = Lattice(A0, A0, A0, 90.0, 90.0, 90.0)
    hkls_cart, thetas, hkls_int = hkls_for_forward_model(
        sg, lat, wavelength_A=0.17304, two_theta_max_deg=10.0,
        expand_equivalents=True)
    geom = HEDMGeometry(
        Lsd=LSD, y_BC=BCY, z_BC=BCZ, px=PX, omega_start=-180.0, omega_step=0.25,
        n_frames=1440, n_pixels_y=NPIX, n_pixels_z=NPIX, min_eta=6.0,
        wavelength=0.17304, tx=0.0, ty=0.0, tz=0.0, wedge=0.0,
        flip_y=True, apply_tilts=False, multi_mode="layered")
    return HEDMForwardModel(hkls_cart, thetas, geom, hkls_int=hkls_int.float())


def _fixed_geo():
    return dict(Lsd=torch.tensor(LSD, dtype=DT), BC_y=torch.tensor(BCY, dtype=DT),
                BC_z=torch.tensor(BCZ, dtype=DT), ty=torch.tensor(0.0, dtype=DT),
                tz=torch.tensor(0.0, dtype=DT), px=torch.tensor(PX, dtype=DT),
                RhoD=torch.tensor(RHOD, dtype=DT),
                p_coeffs=torch.zeros(15, dtype=DT))


def _synth(tx_true, eps_true, n_grains=5, seed=0, max_spots=30):
    """Forward-model grains CARRYING eps_true, then bake tx_true into the
    observed (Y,Z). Identity matching, so tx and the strain are the only
    errors left in the observations."""
    m = _model()
    rng = np.random.default_rng(seed)
    eulers = rng.uniform(-math.pi, math.pi, size=(n_grains, 3))
    eulers[:, 1] = rng.uniform(0, math.pi, size=n_grains)
    positions = np.zeros((n_grains, 3))
    lattices = np.tile([A0, A0, A0, 90.0, 90.0, 90.0], (n_grains, 1))
    T = build_tilt_matrix_torch(torch.tensor(tx_true, dtype=DT),
                                torch.tensor(0.0, dtype=DT),
                                torch.tensor(0.0, dtype=DT))
    Rot2 = T[1:, 1:]
    eul_t = torch.from_numpy(eulers).to(DT)
    lat_t = torch.from_numpy(lattices).to(DT)
    eps_t = torch.from_numpy(eps_true).to(DT)
    obs, matches, raw, keep = [], [], [], []
    for g in range(n_grains):
        s = m(eul_t[g].view(1, 1, 3), torch.zeros(1, 1, 3, dtype=DT),
              lattice_params=lat_t[g].view(1, 6), strain=eps_t[g].view(1, 6))

        def sq(t):
            while t.dim() > 2 and t.shape[0] == 1:
                t = t.squeeze(0)
            return t
        valid = sq(s.valid).bool()
        ks, ms = torch.where(valid)
        if ks.numel() == 0:
            continue
        if ks.numel() > max_spots:
            sel = torch.from_numpy(rng.permutation(ks.numel())[:max_spots])
            ks, ms = ks[sel], ms[sel]
        M = valid.shape[1]
        flat = ks * M + ms
        om = sq(s.omega).double().reshape(-1)[flat]
        eta = sq(s.eta).double().reshape(-1)[flat]
        tth = sq(s.two_theta).double().reshape(-1)[flat]
        R = LSD * torch.tan(tth)
        pred = torch.stack([-R * torch.sin(eta), R * torch.cos(eta)], dim=-1)
        ov = pred @ Rot2
        S = ks.numel()
        obs.append(ObservedSpots(
            spot_id=torch.arange(S), ring_nr=torch.zeros(S, dtype=torch.int64),
            y_lab=ov[..., 0], z_lab=ov[..., 1], omega=om, eta=eta, two_theta=tth,
            grain_radius=torch.full((S,), 50.0, dtype=DT),
            fit_rmse=torch.zeros(S, dtype=DT), y_orig=torch.zeros(S, dtype=DT),
            z_orig=torch.zeros(S, dtype=DT), omega_ini=om.clone(),
            mask_touched=torch.zeros(S, dtype=torch.bool)))
        matches.append(MatchResult(
            k_idx=ks.long(), m_idx=ms.long(), mask=torch.ones(S, dtype=torch.bool),
            delta_omega=torch.zeros(S, dtype=DT), delta_eta=torch.zeros(S, dtype=DT)))
        raw.append((torch.zeros(S, dtype=DT), torch.zeros(S, dtype=DT)))
        keep.append(g)
    return m, obs, matches, raw, eulers[keep], positions[keep], lattices[keep]


def _fit_tx(m, obs, matches, raw, eul, pos, lat, *, refine_strain,
            strain_init=None, max_iter=60):
    spec = mp.ParameterSpec()
    spec.add(Parameter("tx", init=torch.tensor(0.0, dtype=DT), refined=True,
                       bounds=(-5.0, 5.0)))
    spec = build_joint_spec(
        powder_spec=spec,
        grain_eulers_init=torch.from_numpy(eul).to(DT),
        grain_positions_init=torch.from_numpy(pos).to(DT),
        grain_lattices_init=torch.from_numpy(lat).to(DT),
        refine_grain_orientation=False, refine_grain_position=False,
        refine_grain_strain=refine_strain)
    if strain_init is not None:
        spec.parameters["grain_strain"].init = torch.from_numpy(strain_init).to(DT)
    resid = make_residual(m, obs, matches, raw, fixed_geo=_fixed_geo(),
                          kind="angular")
    u, cost, rc = mp.lm_minimise(
        spec, resid,
        config=mp.GenericLMConfig(max_iter=max_iter, ftol_rel=1e-12,
                                  xtol_rel=1e-12),
        fallback_span=2.0)
    return float(u["tx"]), float(cost), u


def _random_strain(n_g, sd_ue, seed=991):
    return np.random.default_rng(seed).normal(0.0, sd_ue * 1e-6, size=(n_g, 6))


def test_zero_strain_is_the_no_strain_path():
    """A frozen zero ``grain_strain`` must reproduce the pre-#70 residual, so
    ``--no-strain`` is a true restoration of the old behaviour and not a
    third, subtly different code path."""
    eps0 = np.zeros((4, 6))
    m, obs, mt, raw, eul, pos, lat = _synth(0.3, eps0, n_grains=4)
    resid = make_residual(m, obs, mt, raw, fixed_geo=_fixed_geo(), kind="angular")
    base = {"tx": torch.tensor(0.11, dtype=DT),
            "grain_euler": torch.from_numpy(eul).to(DT),
            "grain_pos": torch.from_numpy(pos).to(DT),
            "grain_lattice": torch.from_numpy(lat).to(DT)}
    r_without = resid(base)                                    # no grain_strain key
    r_with = resid({**base, "grain_strain": torch.zeros(len(eul), 6, dtype=DT)})
    assert torch.allclose(r_with, r_without, atol=1e-9, rtol=0)


def test_free_strain_removes_the_tx_bias_a_real_strain_causes():
    """The measurement the reporter's ablation was trying to make.

    1000 µε of per-grain strain, frozen at the nominal lattice, biases tx by
    ~1e-3 deg; freeing the strain recovers tx exactly (this is an inverse
    crime — same forward model both ways — so the free arm is expected to be
    exact, and the number that matters is the frozen arm's bias).
    """
    tx_true = 0.30
    eps = _random_strain(5, 1000.0)
    m, obs, mt, raw, eul, pos, lat = _synth(tx_true, eps)
    tx_frozen, cost_frozen, _ = _fit_tx(m, obs, mt, raw, eul, pos, lat,
                                        refine_strain=False)
    tx_free, cost_free, u = _fit_tx(m, obs, mt, raw, eul, pos, lat,
                                    refine_strain=True)
    bias = abs(tx_frozen - tx_true)
    assert bias > 3e-4, (
        f"the frozen arm is meant to be biased here, got {bias:.2e} deg — "
        "if this fails the test has lost its power, not the code its bug")
    assert abs(tx_free - tx_true) < 1e-5 < bias
    assert cost_free < 1e-6 * cost_frozen
    # and the strain it recovers IS the strain that was put in
    got = u["grain_strain"].detach().numpy()
    assert np.abs(got - eps).max() < 5e-6      # 5 ue


def test_free_strain_does_not_absorb_tx_when_there_is_none():
    """The null the fix has to survive: 6 free parameters per grain against 2
    geometry parameters could soak up tx. With no strain in the data they do
    not — the two arms agree to 1e-9 deg."""
    eps0 = np.zeros((5, 6))
    m, obs, mt, raw, eul, pos, lat = _synth(0.30, eps0)
    tx_frozen, _, _ = _fit_tx(m, obs, mt, raw, eul, pos, lat, refine_strain=False)
    tx_free, _, u = _fit_tx(m, obs, mt, raw, eul, pos, lat, refine_strain=True)
    assert abs(tx_frozen - 0.30) < 1e-8
    assert abs(tx_free - tx_frozen) < 1e-9
    assert np.abs(u["grain_strain"].detach().numpy()).max() < 1e-7


def test_the_tx_coupling_is_carried_by_the_DEVIATORIC_strain():
    """Mechanism, not just outcome — and the decomposition matters.

    A symmetric strain carries no rigid rotation, and its frame-independent
    radial part is the HYDROSTATIC one, tr(eps)/3 * I: it scales every
    |g| and so moves every spot along R, which tx (a pure in-plane rotation)
    leaves invariant. Everything that shifts eta is deviatoric.

    Note this is NOT the same as "the shear components". e11, e22, e33 are
    normal only in the CRYSTAL frame; unless they are equal they carry
    deviatoric content in the lab, and freezing them wrongly does bias tx.
    Measured here: the hydrostatic arm biases tx ~300-500x less than the
    deviatoric arm, and the deviatoric arm alone reproduces the full bias.

    This is why the fix moves tx so little on real FF data: a per-grain
    lattice differs from the nominal one mostly by a COMMON hydrostatic
    offset (+1100 ue on shade_LSHR layer 1, a/b/c within 20 ue of each
    other), which is precisely the tx-blind part.
    """
    tx_true = 0.30
    diag = [0, 3, 5]                        # plain-Voigt e11, e22, e33
    eps = _random_strain(5, 1000.0)
    tr = eps[:, diag].sum(axis=1) / 3.0
    hydro = np.zeros_like(eps); hydro[:, diag] = tr[:, None]
    dev = eps - hydro

    def frozen_bias(e):
        m, obs, mt, raw, eul, pos, lat = _synth(tx_true, e)
        tx, _, _ = _fit_tx(m, obs, mt, raw, eul, pos, lat, refine_strain=False)
        return abs(tx - tx_true)

    b_full, b_hyd, b_dev = frozen_bias(eps), frozen_bias(hydro), frozen_bias(dev)
    assert b_hyd < 0.02 * b_dev, (
        f"hydrostatic bias {b_hyd:.2e} deg vs deviatoric {b_dev:.2e} deg — "
        "the hydrostatic part is supposed to be radial, and radial is what tx "
        "cannot see")
    assert b_dev == pytest.approx(b_full, rel=0.25), (
        f"deviatoric {b_dev:.2e} should account for the full bias {b_full:.2e}")
