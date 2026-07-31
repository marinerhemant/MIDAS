"""Tests for the paired ``OmegaRange`` / ``BoxSize`` acceptance gate.

Reference C: ``CalcDiffrSpots_Furnace`` in
``NF_HEDM/src/CalcDiffractionSpots.c:185-248`` (identical code in
``NF_HEDM/src/MakeDiffrSpots.c:216-259``)::

    RealType RingRadius = distance * tan(2 * deg2rad * Thetas[indexhkl]);  # :209
    CalcSpotPosition(RingRadius, etas[i], &yl, &zl);                       # :224
      # yl = -(sin(eta) * RingRadius);  zl = cos(eta) * RingRadius;        # :83-84
    for (OmegaRangeNo = 0; OmegaRangeNo < NOmegaRanges; OmegaRangeNo++) {  # :225
      KeepSpot = 0;
      if ((Omega > OmegaRange[i][0]) && (Omega < OmegaRange[i][1]) &&      # :227-232
          (yl > BoxSizes[i][0]) && (yl < BoxSizes[i][1]) &&
          (zl > BoxSizes[i][2]) && (zl < BoxSizes[i][3])) {
        KeepSpot = 1; break;
      }
    }
    if (KeepSpot == 1) { spots[spotnr*3 + ...] = ...; spotnr++; }          # :237-243

``distance`` is ``Lsd[0]`` -- the caller passes it explicitly
(``NF_HEDM/src/SharedFuncsFit.c:830``). A rejected spot is never written into
``TheorSpots``, so it is dropped from both the numerator and the denominator of
``CalcFracOverlap`` (``SharedFuncsFit.c:645`` increments ``TotalPixels`` only
for spots present in the list).

Run with:
    cd packages/midas_diffract
    python -m pytest tests/test_omega_box_filter.py -v
"""

import math

import pytest
import torch

from midas_diffract.forward import HEDMForwardModel, HEDMGeometry


LSD = 1_000_000.0


def _geom(**kw):
    base = dict(
        Lsd=LSD,
        y_BC=1024.0,
        z_BC=1024.0,
        px=200.0,
        omega_start=-180.0,
        omega_step=1.0,
        n_frames=360,
        n_pixels_y=2048,
        n_pixels_z=2048,
        min_eta=6.0,
        wavelength=0.172979,
        flip_y=False,          # NF convention
        multi_mode="layered",
    )
    base.update(kw)
    return HEDMGeometry(**base)


def _model(geom):
    """Gold (a = 4.08 A) reflections, matching the NF fit-orientation path."""
    hkls_int = torch.tensor(
        [[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0, 0], [2, 1, 1]],
        dtype=torch.float64,
    )
    B = torch.eye(3, dtype=torch.float64) / 4.08
    hkls_cart = hkls_int @ B.T
    g = torch.linalg.norm(hkls_cart, dim=-1)
    thetas = torch.asin(g * geom.wavelength / 2.0)
    return HEDMForwardModel(
        hkls=hkls_cart, thetas=thetas, geometry=geom,
        hkls_int=hkls_int, device=torch.device("cpu"),
    )


EUL = torch.tensor([[0.3, 0.5, 0.7]], dtype=torch.float64)
POS = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)


def _nominal_ring_coords(model, eul=EUL):
    """(yl, zl) in um exactly as the C CalcSpotPosition computes them."""
    om, eta, tt, valid = model.calc_bragg_geometry(model.euler2mat(eul))
    R = LSD * torch.tan(tt)
    return -torch.sin(eta) * R, torch.cos(eta) * R, om, valid


# ---------------------------------------------------------------------------
#  Default: no BoxSize => filter off, behaviour unchanged
# ---------------------------------------------------------------------------

class TestDefaultOff:
    def test_flag_off_and_buffers_empty(self):
        m = _model(_geom())
        assert m._has_omega_box is False
        assert m._omega_ranges.shape == (0, 2)
        assert m._box_sizes.shape == (0, 4)

    def test_valid_mask_identical_to_wide_open_box(self):
        """A box wider than any possible ring must not change ``valid``."""
        m_off = _model(_geom())
        m_open = _model(_geom(
            omega_ranges=[(-180.0, 180.0)],
            box_sizes=[(-1e9, 1e9, -1e9, 1e9)],
        ))
        s_off = m_off(EUL, POS)
        s_open = m_open(EUL, POS)
        assert torch.equal(s_off.valid, s_open.valid)
        assert float(s_off.valid.sum()) > 0     # the test is not vacuous

    def test_ff_geometry_unaffected(self):
        """FF/pf callers never pass the keys -> byte-identical forward."""
        m = _model(_geom(flip_y=True, multi_mode="panel"))
        assert m._has_omega_box is False
        s = m(EUL, POS)
        assert float(s.valid.sum()) > 0


# ---------------------------------------------------------------------------
#  Inside is kept, outside is dropped
# ---------------------------------------------------------------------------

class TestInsideOutside:
    def test_spot_inside_box_is_kept_outside_is_dropped(self):
        m_off = _model(_geom())
        yl, zl, _om, _v = _nominal_ring_coords(m_off)
        s_off = m_off(EUL, POS)
        base = s_off.valid.bool()
        n_base = int(base.sum())
        assert n_base > 2

        # Cut just above the smallest valid zl -> drop exactly those spots.
        z_cut = float(zl[base].min()) + 1.0
        n_below = int((base & (zl <= z_cut)).sum())
        assert n_below > 0

        m_box = _model(_geom(
            omega_ranges=[(-180.0, 180.0)],
            box_sizes=[(-1e9, 1e9, z_cut, 1e9)],
        ))
        kept = m_box(EUL, POS).valid.bool()

        assert int(kept.sum()) == n_base - n_below
        # Every kept spot was valid before and is inside the box.
        assert bool((kept <= base).all())
        assert bool((zl[kept] > z_cut).all())

    def test_box_is_micrometres_at_lsd0_not_pixels(self):
        """A 0..2048 box read as *pixels* would keep everything; read as um
        (the C convention) it keeps nothing for this geometry."""
        m_off = _model(_geom())
        yl, zl, _om, _v = _nominal_ring_coords(m_off)
        base = m_off(EUL, POS).valid.bool()
        # Sanity: every valid ring here is far outside a +/-2048 um window.
        assert float(zl[base].abs().min()) > 2048.0

        m_px = _model(_geom(
            omega_ranges=[(-180.0, 180.0)],
            box_sizes=[(-2048.0, 2048.0, -2048.0, 2048.0)],
        ))
        assert float(m_px(EUL, POS).valid.sum()) == 0.0

    def test_y_sign_convention(self):
        """yl = -sin(eta) * R: an eta = +90 deg spot lands at negative y."""
        m = _model(_geom(
            omega_ranges=[(-180.0, 180.0)],
            box_sizes=[(-1e9, 1e9, -1e9, 1e9)],
        ))
        two_theta = torch.tensor([[0.1]], dtype=torch.float64)
        eta = torch.tensor([[math.pi / 2]], dtype=torch.float64)
        omega = torch.zeros_like(eta)
        R = float(LSD * math.tan(0.1))
        # Positive-y half plane rejects it, negative-y half plane accepts it.
        m._box_sizes = torch.tensor([[0.0, 1e9, -1e9, 1e9]], dtype=torch.float64)
        assert not bool(m.omega_box_mask(omega, eta, two_theta).item())
        m._box_sizes = torch.tensor([[-1e9, 0.0, -1e9, 1e9]], dtype=torch.float64)
        assert bool(m.omega_box_mask(omega, eta, two_theta).item())
        assert R > 0     # ring radius is positive, so the sign came from -sin


# ---------------------------------------------------------------------------
#  Strict (exclusive) bounds -- C uses > and <, never >= / <=
# ---------------------------------------------------------------------------

class TestStrictBounds:
    @staticmethod
    def _one_spot(box):
        """Evaluate the mask for a single spot at eta = 0, omega = 0."""
        m = _model(_geom(
            omega_ranges=[(-180.0, 180.0)], box_sizes=[box],
        ))
        two_theta = torch.tensor([[0.1]], dtype=torch.float64)
        eta = torch.zeros_like(two_theta)
        omega = torch.zeros_like(two_theta)
        # zl = cos(0) * Lsd*tan(2theta); reproduce bit-for-bit.
        zl = float(
            torch.cos(eta) * (torch.tensor(LSD, dtype=torch.float64)
                              * torch.tan(two_theta))
        )
        return m, omega, eta, two_theta, zl

    def test_zmin_edge_exact_is_rejected(self):
        m, om, eta, tt, zl = self._one_spot((-1e6, 1e6, 0.0, 1e9))
        m._box_sizes = torch.tensor([[-1e6, 1e6, zl, 1e9]], dtype=torch.float64)
        assert not bool(m.omega_box_mask(om, eta, tt).item())

    def test_zmax_edge_exact_is_rejected(self):
        m, om, eta, tt, zl = self._one_spot((-1e6, 1e6, 0.0, 1e9))
        m._box_sizes = torch.tensor([[-1e6, 1e6, -1e9, zl]], dtype=torch.float64)
        assert not bool(m.omega_box_mask(om, eta, tt).item())

    def test_strictly_inside_is_accepted(self):
        m, om, eta, tt, zl = self._one_spot((-1e6, 1e6, 0.0, 1e9))
        m._box_sizes = torch.tensor(
            [[-1e6, 1e6, zl - 1e-6, zl + 1e-6]], dtype=torch.float64,
        )
        assert bool(m.omega_box_mask(om, eta, tt).item())

    def test_omega_edge_exact_is_rejected(self):
        m, om, eta, tt, zl = self._one_spot((-1e6, 1e6, -1e9, 1e9))
        # omega == 0 exactly; a range starting at 0 must reject it.
        m._omega_ranges = torch.tensor([[0.0, 180.0]], dtype=torch.float64)
        assert not bool(m.omega_box_mask(om, eta, tt).item())
        m._omega_ranges = torch.tensor([[-1e-9, 180.0]], dtype=torch.float64)
        assert bool(m.omega_box_mask(om, eta, tt).item())


# ---------------------------------------------------------------------------
#  Paired any-pair-accepts logic
# ---------------------------------------------------------------------------

class TestPairing:
    def test_any_pair_accepts(self):
        """Pair 0 accepts on omega only, pair 1 on box only; each spot needs
        BOTH halves of the SAME pair, and either pair is enough."""
        m_off = _model(_geom())
        yl, zl, om, _v = _nominal_ring_coords(m_off)
        base = m_off(EUL, POS).valid.bool()
        om_deg = om * (180.0 / math.pi)

        # Split the valid spots by omega sign.
        neg = base & (om_deg < 0)
        pos = base & (om_deg > 0)
        assert int(neg.sum()) > 0 and int(pos.sum()) > 0

        z_hi = float(zl[base].max()) + 1.0
        z_lo = float(zl[base].min()) - 1.0

        # Pair 0: negative omega, box open.  Pair 1: positive omega, box shut.
        m2 = _model(_geom(
            omega_ranges=[(-180.0, 0.0), (0.0, 180.0)],
            box_sizes=[(-1e9, 1e9, z_lo, z_hi), (-1e9, 1e9, 1e8, 1e9)],
        ))
        kept = m2(EUL, POS).valid.bool()
        assert torch.equal(kept, neg)

        # Swapping the boxes swaps which half survives -- the pairing is by
        # index, not a global OR of all omegas with all boxes.
        m3 = _model(_geom(
            omega_ranges=[(-180.0, 0.0), (0.0, 180.0)],
            box_sizes=[(-1e9, 1e9, 1e8, 1e9), (-1e9, 1e9, z_lo, z_hi)],
        ))
        assert torch.equal(m3(EUL, POS).valid.bool(), pos)

    def test_both_pairs_open_is_union(self):
        m_off = _model(_geom())
        yl, zl, om, _v = _nominal_ring_coords(m_off)
        base = m_off(EUL, POS).valid.bool()
        z_hi = float(zl[base].max()) + 1.0
        z_lo = float(zl[base].min()) - 1.0
        m2 = _model(_geom(
            omega_ranges=[(-180.0, 0.0), (0.0, 180.0)],
            box_sizes=[(-1e9, 1e9, z_lo, z_hi), (-1e9, 1e9, z_lo, z_hi)],
        ))
        assert torch.equal(m2(EUL, POS).valid.bool(), base)


# ---------------------------------------------------------------------------
#  Arity validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_box_without_omega_range_raises(self):
        with pytest.raises(ValueError, match="paired filter"):
            _model(_geom(box_sizes=[(-1e6, 1e6, -1e6, 1e6)]))

    def test_omega_range_without_box_raises(self):
        with pytest.raises(ValueError, match="paired filter"):
            _model(_geom(omega_ranges=[(-180.0, 180.0)]))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="must match box_sizes"):
            _model(_geom(
                omega_ranges=[(-180.0, 180.0)],
                box_sizes=[(-1e6, 1e6, -1e6, 1e6), (-1e6, 1e6, -1e6, 1e6)],
            ))


# ---------------------------------------------------------------------------
#  The mask reaches every downstream consumer of ``valid``
# ---------------------------------------------------------------------------

class TestPropagation:
    def _pair(self):
        m_off = _model(_geom())
        yl, zl, _om, _v = _nominal_ring_coords(m_off)
        base = m_off(EUL, POS).valid.bool()
        z_cut = float(zl[base].min()) + 1.0
        m_box = _model(_geom(
            omega_ranges=[(-180.0, 180.0)],
            box_sizes=[(-1e9, 1e9, z_cut, 1e9)],
        ))
        return m_off, m_box

    def test_layer_valid_inherits_in_multi_distance(self):
        hkls_int = torch.tensor([[1, 1, 1], [2, 0, 0]], dtype=torch.float64)
        B = torch.eye(3, dtype=torch.float64) / 4.08
        hkls_cart = hkls_int @ B.T
        g = torch.linalg.norm(hkls_cart, dim=-1)
        thetas = torch.asin(g * 0.172979 / 2.0)

        def build(**kw):
            geom = _geom(
                Lsd=[LSD, 1.05 * LSD],
                y_BC=[1024.0, 1024.0],
                z_BC=[1024.0, 1024.0],
                **kw,
            )
            return HEDMForwardModel(
                hkls=hkls_cart, thetas=thetas, geometry=geom,
                hkls_int=hkls_int, device=torch.device("cpu"),
            )

        m_off = build()
        s_off = m_off(EUL, POS)
        assert s_off.layer_valid is not None
        assert float(s_off.valid.sum()) > 0

        m_box = build(
            omega_ranges=[(-180.0, 180.0)],
            box_sizes=[(-1e9, 1e9, 1e8, 1e9)],     # shut
        )
        s_box = m_box(EUL, POS)
        assert float(s_box.valid.sum()) == 0.0
        assert float(s_box.layer_valid.sum()) == 0.0

    def test_forward_per_grain_agrees_with_forward(self):
        _m_off, m_box = self._pair()
        s_fwd = m_box(EUL, POS)
        s_pg = m_box.forward_per_grain(EUL, POS)
        assert torch.equal(s_fwd.valid.reshape(-1), s_pg.valid.reshape(-1))
