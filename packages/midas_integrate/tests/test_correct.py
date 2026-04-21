"""correct_image() — geometric rectification (Dioptas interop).

Round-trip validation (correct → re-calibrate → <5 µε) is its own test
file (``test_correct_roundtrip.py``, week 11). Here we exercise the
pure-geometry + panel plumbing on synthetic inputs so these run
everywhere without the C binaries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midas_auto_calibrate import DetectorGeometry
from midas_integrate import (
    IntegrationConfig,
    Panel,
    correct_image,
    correct_images,
    generate_panels,
    load_panel_shifts,
    write_tiff,
)
from midas_integrate.correct import _compute_forward_map, _tilt_matrix


class TestTiltMatrix:
    def test_zero_tilts_is_identity(self):
        M = _tilt_matrix(0, 0, 0)
        np.testing.assert_allclose(M, np.eye(3), atol=1e-15)

    def test_small_tilt_is_nearly_identity(self):
        M = _tilt_matrix(0.01, -0.02, 0.05)
        np.testing.assert_allclose(M, np.eye(3), atol=1e-3)


class TestGeneratePanels:
    def test_pilatus_6x8_layout(self):
        # Pilatus CdTe 6M layout — 6 columns, 8 rows, 243×195 modules, gaps.
        panels = generate_panels(
            n_panels_y=6, n_panels_z=8,
            panel_size_y=243, panel_size_z=195,
            gaps_y=[1, 7, 1, 7, 1],
            gaps_z=[17, 17, 17, 17, 17, 17, 17],
        )
        assert len(panels) == 48

        # First panel anchored at (0, 0).
        assert panels[0].y_min == 0
        assert panels[0].z_min == 0
        assert panels[0].y_max == 242
        assert panels[0].z_max == 194

        # Second panel in Z direction shifted by panel_size + first gap.
        assert panels[1].z_min == 195 + 17
        assert panels[1].y_min == 0

    def test_ids_are_sequential(self):
        panels = generate_panels(2, 3, 100, 100, [5], [5, 5])
        assert [p.id for p in panels] == [0, 1, 2, 3, 4, 5]


class TestLoadPanelShifts:
    def test_applies_shifts_in_place(self, tmp_path):
        panels = generate_panels(2, 2, 100, 100, [5], [5])
        shifts = tmp_path / "shifts.txt"
        shifts.write_text(
            "# header comment\n"
            "\n"
            "0 1.5 -2.5 0.01\n"
            "2 -0.1 0.2\n"   # no dTheta → should stay 0
        )
        load_panel_shifts(shifts, panels)
        assert panels[0].dY == pytest.approx(1.5)
        assert panels[0].dZ == pytest.approx(-2.5)
        assert panels[0].dTheta == pytest.approx(0.01)
        assert panels[2].dY == pytest.approx(-0.1)
        assert panels[2].dTheta == 0.0

    def test_six_column_layout(self, tmp_path):
        """MIDAS writes ``id dY dZ dTheta dLsd dP2`` — parse all six."""
        panels = generate_panels(1, 1, 10, 10, [], [])
        shifts = tmp_path / "shifts.txt"
        shifts.write_text("0 0.5 -0.3 0.02 100.0 1.5e-4\n")
        load_panel_shifts(shifts, panels)
        p = panels[0]
        assert p.dY == pytest.approx(0.5)
        assert p.dZ == pytest.approx(-0.3)
        assert p.dTheta == pytest.approx(0.02)
        assert p.dLsd == pytest.approx(100.0)
        assert p.dP2 == pytest.approx(1.5e-4)

    def test_unknown_id_ignored(self, tmp_path):
        panels = generate_panels(1, 1, 10, 10, [], [])
        shifts = tmp_path / "shifts.txt"
        shifts.write_text("99 1 2\n0 0.5 0.5\n")
        load_panel_shifts(shifts, panels)
        assert panels[0].dY == 0.5


class TestCorrectImageNoDistortion:
    """With zero tilts + zero distortion, correction is a no-op up to
    interpolation error."""

    def test_identity_preserves_image(self):
        geom = DetectorGeometry(
            lsd=1_000_000, ybc=256, zbc=256, tx=0, ty=0, tz=0,
            px=200.0, nr_pixels_y=512, nr_pixels_z=512,
            rhod=50_000,
        )
        rng = np.random.default_rng(0)
        img = rng.uniform(0, 100, (512, 512)).astype(np.float64)
        corrected = correct_image(img, geom)

        # Interior pixels should be near-identical; edges may differ due
        # to the tilt matrix at y=0/z=0 having boundary interpolation.
        interior = np.index_exp[50:462, 50:462]
        np.testing.assert_allclose(
            corrected[interior], img[interior], rtol=0, atol=0.5
        )


class TestCorrectImageWithTilt:
    """A pure tilt should move content radially outward at high η."""

    def test_tilt_changes_smooth_image(self):
        """A non-trivial tilt alters the corrected image vs the untilted case.

        A delta spike is a poor probe because bilinear interpolation smears
        its energy across sub-pixel inverse-map lookups; use a smooth
        radial ramp so small displacements show up in the pixel values.
        """
        # Build a smooth radial ramp centred on (256, 256).
        yy, zz = np.meshgrid(np.arange(512), np.arange(512))
        img = np.hypot(yy - 256, zz - 256).astype(np.float64)

        no_tilt = DetectorGeometry(
            lsd=500_000, ybc=256, zbc=256, tx=0, ty=0, tz=0,
            px=200.0, nr_pixels_y=512, nr_pixels_z=512, rhod=50_000,
        )
        tilted = DetectorGeometry(
            lsd=500_000, ybc=256, zbc=256, tx=0, ty=0.5, tz=0,
            px=200.0, nr_pixels_y=512, nr_pixels_z=512, rhod=50_000,
        )
        corrected_no_tilt = correct_image(img, no_tilt)
        corrected_tilt = correct_image(img, tilted)

        # The tilt-corrected ramp should differ from the no-tilt one in
        # the interior (edges may match due to boundary behaviour).
        interior = np.index_exp[50:462, 50:462]
        diff = corrected_tilt[interior] - corrected_no_tilt[interior]
        assert np.max(np.abs(diff)) > 0.1, (
            "tilt correction produced no visible change — "
            f"max(|Δ|) = {np.max(np.abs(diff)):.3g}"
        )


class TestInvalidInputs:
    def test_shape_mismatch_raises(self):
        geom = DetectorGeometry(
            lsd=1_000_000, ybc=256, zbc=256,
            px=200.0, nr_pixels_y=512, nr_pixels_z=512,
        )
        img = np.zeros((128, 128))
        with pytest.raises(ValueError, match="image shape"):
            correct_image(img, geom)

    def test_unsupported_file_format_raises(self, tmp_path):
        geom = DetectorGeometry(px=200.0, nr_pixels_y=512, nr_pixels_z=512)
        fn = tmp_path / "image.png"
        fn.write_bytes(b"not a tiff")
        with pytest.raises(ValueError, match="Unsupported image format"):
            correct_image(fn, geom)


class TestWriteTiffProvenance:
    def test_description_contains_geometry(self, tmp_path):
        import tifffile

        arr = np.arange(100, dtype=np.float32).reshape(10, 10)
        geom = DetectorGeometry(
            lsd=657_436.9, ybc=685.5, zbc=921.0, px=172.0,
        )
        path = tmp_path / "out.tif"
        write_tiff(path, arr, geometry=geom)

        with tifffile.TiffFile(path) as tf:
            desc = tf.pages[0].tags["ImageDescription"].value
        assert "midas-integrate" in desc
        assert "Lsd=657436.900" in desc
        assert "BC=(685.500, 921.000)" in desc
        assert "px=172.000" in desc

    def test_write_without_geometry_has_no_midas_prefix(self, tmp_path):
        import tifffile
        arr = np.arange(100, dtype=np.float32).reshape(10, 10)
        path = tmp_path / "bare.tif"
        write_tiff(path, arr)
        with tifffile.TiffFile(path) as tf:
            desc = tf.pages[0].tags.get("ImageDescription")
            # tifffile may emit a tag with an empty description when we pass
            # None — what we care about is that no "midas-integrate …"
            # provenance string got baked in.
            if desc is not None and desc.value:
                assert "midas-integrate" not in desc.value


class TestCorrectImagesBatch:
    def test_writes_corrected_suffix_per_input(self, tmp_path):
        geom = DetectorGeometry(
            lsd=1_000_000, ybc=64, zbc=64, px=200.0,
            nr_pixels_y=128, nr_pixels_z=128, rhod=10_000,
        )
        # Two tiny TIFFs.
        import tifffile
        paths = []
        for i, val in enumerate([1.0, 2.0]):
            p = tmp_path / f"frame_{i:03d}.tif"
            tifffile.imwrite(p, np.full((128, 128), val, dtype=np.float32))
            paths.append(p)
        written = correct_images(
            paths, geom, output_dir=tmp_path / "out", suffix="_cor",
        )
        assert len(written) == 2
        for orig, out in zip(paths, written):
            assert out.name == f"{orig.stem}_cor.tif"
            assert out.exists()


class TestForwardMapSanity:
    def test_no_distortion_is_identity_at_beam_center(self):
        # At the beam center, no math can move the pixel anywhere else —
        # every model collapses to (ybc, zbc) at (ybc, zbc).
        cfg = IntegrationConfig(
            lsd=1_000_000, ybc=100, zbc=100, tx=0, ty=0, tz=0,
            pixel_size=200.0, nr_pixels_y=200, nr_pixels_z=200,
            rho_d=20_000,
        )
        fwd_y, fwd_z = _compute_forward_map(cfg, [], None)
        # At (y=100, z=100) (i.e. at the beam center), ideal ≈ beam center.
        # fwd arrays are indexed (z, y), so [100, 100] is the BC.
        assert fwd_y[100, 100] == pytest.approx(cfg.ybc, abs=1e-6)
        assert fwd_z[100, 100] == pytest.approx(cfg.zbc, abs=1e-6)


class TestFullDistortionModel:
    """Verify each p0..p14 term contributes the expected displacement.

    Strategy: set ALL distortion to zero → fwd map is identity (up to
    tilts, which we also zero). Then turn on ONE coefficient at a time
    and verify the resulting displacement in a direction the coefficient's
    harmonic predicts.
    """

    def _flat_cfg(self, **overrides):
        defaults = dict(
            lsd=1_000_000, ybc=512, zbc=512, tx=0, ty=0, tz=0,
            pixel_size=200.0, nr_pixels_y=1024, nr_pixels_z=1024,
            rho_d=50_000,
        )
        defaults.update(overrides)
        return IntegrationConfig(**defaults)

    def test_all_zero_is_identity(self):
        """Zero distortion + zero tilt → fwd map is identity."""
        cfg = self._flat_cfg()
        fwd_y, fwd_z = _compute_forward_map(cfg, [], None)
        # Interior only — boundary has interpolation effects via tilt matrix.
        y_expected = np.arange(1024)[None, :]
        z_expected = np.arange(1024)[:, None]
        # With zero tilts and zero distortion, fwd should match identity exactly.
        np.testing.assert_allclose(fwd_y[256:768, 256:768],
                                   np.broadcast_to(y_expected, fwd_y.shape)[256:768, 256:768],
                                   atol=1e-9)
        np.testing.assert_allclose(fwd_z[256:768, 256:768],
                                   np.broadcast_to(z_expected, fwd_z.shape)[256:768, 256:768],
                                   atol=1e-9)

    def test_p4_r6_term_grows_with_radius(self):
        """p4 · R⁶ — a pure radial distortion, no η dependence."""
        cfg_off = self._flat_cfg()
        cfg_on = self._flat_cfg(p4=1.0)  # big enough to see
        fy0, fz0 = _compute_forward_map(cfg_off, [], None)
        fy1, fz1 = _compute_forward_map(cfg_on, [], None)
        # Displacement at the edge (R large) should exceed displacement
        # near the BC (R small).
        near_bc = np.hypot(fy1[520, 520] - fy0[520, 520],
                            fz1[520, 520] - fz0[520, 520])
        far = np.hypot(fy1[900, 900] - fy0[900, 900],
                        fz1[900, 900] - fz0[900, 900])
        assert far > near_bc * 10, f"p4 term not R-dependent: near={near_bc}, far={far}"

    def test_p5_r4_term_grows_with_radius(self):
        """p5 · R⁴ — pure radial, like p4 but weaker R-dependence."""
        cfg_off = self._flat_cfg()
        cfg_on = self._flat_cfg(p5=1.0)
        fy0, fz0 = _compute_forward_map(cfg_off, [], None)
        fy1, fz1 = _compute_forward_map(cfg_on, [], None)
        # At BC: no displacement. Far: significant.
        near = np.abs(fy1[520, 520] - fy0[520, 520]) + np.abs(fz1[520, 520] - fz0[520, 520])
        far = np.abs(fy1[900, 900] - fy0[900, 900]) + np.abs(fz1[900, 900] - fz0[900, 900])
        assert far > near * 5

    def test_p9_3eta_harmonic_creates_threefold_pattern(self):
        """p9 · R³·cos(3·EtaT + p10) — threefold symmetric in EtaT.

        EtaT = 90 − η, so cos(3·EtaT) peaks at EtaT ∈ {0°, 120°, 240°},
        which maps to η ∈ {90°, −30°, −150°}. At BC=(512,512), R=100 px:
            η=90°   → (y,z) = (612, 512)    (east of BC)
            η=−30°  → (y,z) = (462, 599)
            η=−150° → (y,z) = (462, 425)
        Displacement magnitude should match within interpolation noise
        at these three points.
        """
        cfg = self._flat_cfg(p9=0.01)
        fy_ref, fz_ref = _compute_forward_map(self._flat_cfg(), [], None)
        fy_on, fz_on = _compute_forward_map(cfg, [], None)

        disps = []
        for y, z in [(612, 512), (462, 599), (462, 425)]:
            dy = fy_on[z, y] - fy_ref[z, y]
            dz = fz_on[z, y] - fz_ref[z, y]
            disps.append(np.hypot(dy, dz))
        disps = np.array(disps)
        assert disps.std() / disps.mean() < 0.05, (
            f"not threefold symmetric: {disps}"
        )
        # All three should be nonzero (verifies the term is active).
        assert disps.min() > 0.01

    def test_p6_phase_shifts_p0_distortion(self):
        """p0 term has ``cos(2η + p6)`` — nonzero p6 rotates the pattern."""
        cfg_no_phase = self._flat_cfg(p0=0.01)
        cfg_phased = self._flat_cfg(p0=0.01, p6=45.0)  # 45° phase shift
        fy0, fz0 = _compute_forward_map(cfg_no_phase, [], None)
        fy1, fz1 = _compute_forward_map(cfg_phased, [], None)
        # Phase shift ≠ 0 → displacement fields differ somewhere on the detector.
        diff = np.abs(fy1 - fy0) + np.abs(fz1 - fz0)
        assert diff.max() > 0.01, (
            "p6 phase not wired: cos(2η+p6=0) and cos(2η+p6=45°) should differ"
        )

    def test_distortion_sum_matches_midas_c_scalar_form(self):
        """Compare a single pixel's distort value against the C formula."""
        cfg = IntegrationConfig(
            lsd=657_436.9, ybc=685.5, zbc=921.0,
            tx=0, ty=0, tz=0, rho_d=219964.42411013643,
            wavelength=0.172973, pixel_size=172.0,
            nr_pixels_y=1475, nr_pixels_z=1679,
            # Mix of all 15 to exercise every term.
            p0=1e-4, p1=2e-4, p2=-5e-4, p3=-13.8, p4=2e-3, p5=1e-4,
            p6=0.5, p7=-2e-3, p8=0.3, p9=5e-5, p10=0.0,
            p11=9e-5, p12=0.0, p13=2e-4, p14=0.0,
        )
        fwd_y, fwd_z = _compute_forward_map(cfg, [], None)

        # Pick a pixel well away from BC and from the detector edge.
        y_px, z_px = 900, 1200
        # Reproduce the math scalarly to sanity-check the vectorized form.
        Yc = -(y_px - cfg.ybc) * cfg.pixel_size
        Zc = (z_px - cfg.zbc) * cfg.pixel_size
        TRs = np.eye(3)
        ABCPr = TRs @ np.array([0, Yc, Zc])
        X = cfg.lsd + ABCPr[0]
        r_yz = np.hypot(ABCPr[1], ABCPr[2])
        R = (cfg.lsd / X) * r_yz
        eta = np.degrees(np.arccos(ABCPr[2] / r_yz))
        if ABCPr[1] > 0:
            eta = -eta
        r_norm = R / cfg.rho_d
        eta_t_rad = np.radians(90 - eta)
        d = (
            cfg.p0 * r_norm**2 * np.cos(2 * eta_t_rad + np.radians(cfg.p6))
            + cfg.p1 * r_norm**4 * np.cos(4 * eta_t_rad + np.radians(cfg.p3))
            + cfg.p2 * r_norm**2
            + cfg.p4 * r_norm**6
            + cfg.p5 * r_norm**4
            + cfg.p7 * r_norm**4 * np.cos(eta_t_rad + np.radians(cfg.p8))
            + cfg.p9 * r_norm**3 * np.cos(3 * eta_t_rad + np.radians(cfg.p10))
            + cfg.p11 * r_norm**5 * np.cos(5 * eta_t_rad + np.radians(cfg.p12))
            + cfg.p13 * r_norm**6 * np.cos(6 * eta_t_rad + np.radians(cfg.p14))
            + 1.0
        )
        Rt_px_expected = R * d / cfg.pixel_size
        eta_rad = np.radians(eta)
        y_expected = cfg.ybc + Rt_px_expected * np.sin(eta_rad)
        z_expected = cfg.zbc + Rt_px_expected * np.cos(eta_rad)
        assert fwd_y[z_px, y_px] == pytest.approx(y_expected, rel=1e-10)
        assert fwd_z[z_px, y_px] == pytest.approx(z_expected, rel=1e-10)


class TestPerPanelDLsdDP2:
    """Panel-level dLsd + dP2 must modify R and p2 for pixels in that panel."""

    def test_per_panel_dlsd_changes_mapping(self):
        from midas_integrate.correct import _panel_index_map
        # Two stacked panels: one with dLsd=0, one with dLsd=500 µm.
        panels = generate_panels(1, 2, 100, 100, [], [20])
        panels[0].dLsd = 0.0
        panels[1].dLsd = 500.0   # 0.5 mm stack-up on second panel

        cfg = IntegrationConfig(
            lsd=1_000_000, ybc=50, zbc=110, tx=0, ty=0, tz=0,
            pixel_size=200.0, nr_pixels_y=100, nr_pixels_z=220,
            p0=1e-4,  # a bit of distortion to amplify the effect
            rho_d=25_000,
        )
        pmap = _panel_index_map(100, 220, panels)

        fy_flat, fz_flat = _compute_forward_map(
            cfg, [Panel(id=p.id, y_min=p.y_min, y_max=p.y_max,
                        z_min=p.z_min, z_max=p.z_max)
                  for p in panels], pmap,
        )
        fy_panel, fz_panel = _compute_forward_map(cfg, panels, pmap)

        # Panel 0 (z_min=0..99) should have identical fwd values both ways.
        np.testing.assert_allclose(fy_flat[:99], fy_panel[:99], atol=1e-9)
        # Panel 1 (z_min=120..219) should differ because of dLsd.
        diff = np.abs(fy_panel[150:200] - fy_flat[150:200])
        assert diff.max() > 1e-6, (
            f"dLsd on panel 1 produced no effect; max diff {diff.max():.2e}"
        )
