"""IntegrationConfig + from_geometry + Parameters.txt rendering."""

from __future__ import annotations

import pytest

from midas_auto_calibrate import DetectorGeometry
from midas_integrate import IntegrationConfig, write_params_file


class TestFromGeometry:
    def test_copies_all_geometry_fields(self):
        geom = DetectorGeometry(
            lsd=600_000, ybc=1021.5, zbc=1029.2,
            tx=0.01, ty=-0.02, tz=0.03,
            p0=1.2e-6, p4=3.4e-9, p14=5.6e-12,
            rhod=180_000, wavelength=0.172979,
            px=172.0, nr_pixels_y=1475, nr_pixels_z=1679,
        )
        cfg = IntegrationConfig.from_geometry(geom)

        assert cfg.lsd == pytest.approx(600_000)
        assert cfg.ybc == pytest.approx(1021.5)
        assert cfg.ty == pytest.approx(-0.02)
        assert cfg.p0 == pytest.approx(1.2e-6)
        assert cfg.p4 == pytest.approx(3.4e-9)
        assert cfg.p14 == pytest.approx(5.6e-12)
        assert cfg.wavelength == pytest.approx(0.172979)
        assert cfg.pixel_size == pytest.approx(172.0)
        assert cfg.nr_pixels_y == 1475
        assert cfg.rho_d == pytest.approx(180_000)

    def test_overrides_merge(self):
        geom = DetectorGeometry(lsd=500_000, wavelength=0.1)
        cfg = IntegrationConfig.from_geometry(geom, r_max=1500, eta_bin_size=0.5)
        assert cfg.lsd == pytest.approx(500_000)
        assert cfg.r_max == 1500
        assert cfg.eta_bin_size == pytest.approx(0.5)

    def test_fallback_pixel_count_when_geometry_missing_it(self):
        # A DetectorGeometry deserialised from old JSON may have 0 pixel counts.
        geom = DetectorGeometry(lsd=100_000, px=200.0)
        cfg = IntegrationConfig.from_geometry(
            geom, nr_pixels_y=2048, nr_pixels_z=2048)
        assert cfg.nr_pixels_y == 2048
        assert cfg.nr_pixels_z == 2048


class TestToParams:
    def _cfg(self):
        return IntegrationConfig(
            lsd=1_000_000,
            ybc=1024, zbc=1024,
            tx=0.0, ty=0.02, tz=-0.01,
            p0=1.0e-5, p1=2.0e-5, p2=3.0e-6,
            wavelength=0.17, pixel_size=172.0,
            nr_pixels_y=1475, nr_pixels_z=1679,
            r_bin_size=0.25, eta_bin_size=1.0,
        )

    def test_emits_all_geometry_keys(self):
        params = self._cfg().to_params()
        for k in ("Lsd", "BC", "tx", "ty", "tz", "Wavelength", "px",
                  "NrPixelsY", "NrPixelsZ", "RhoD",
                  "RMin", "RMax", "RBinSize",
                  "EtaMin", "EtaMax", "EtaBinSize",
                  "SolidAngleCorrection"):
            assert k in params, f"missing key {k}"
        # All 15 p coefficients emitted, zero or not.
        for i in range(15):
            assert f"p{i}" in params

    def test_rho_d_auto_filled(self):
        cfg = self._cfg()
        params = cfg.to_params()
        # min(1475, 1679)/2 * 172 = 126850
        assert params["RhoD"] == pytest.approx(126_850)

    def test_rho_d_override(self):
        cfg = self._cfg()
        cfg.rho_d = 200_000
        assert cfg.to_params()["RhoD"] == pytest.approx(200_000)

    def test_r_max_auto_to_half_short_side(self):
        params = self._cfg().to_params()
        assert params["RMax"] == pytest.approx(737.5)

    def test_extra_params_overrides_defaults(self):
        cfg = self._cfg()
        cfg.extra_params = {"RhoD": 1234, "Foo": "Bar"}
        params = cfg.to_params()
        assert params["RhoD"] == 1234
        assert params["Foo"] == "Bar"

    def test_panel_params_only_emitted_when_set(self):
        cfg = self._cfg()
        assert "NPanelsY" not in cfg.to_params()

        cfg.n_panels_y = 6
        cfg.n_panels_z = 8
        cfg.panel_gaps_y = [1, 7, 1, 7, 1]
        cfg.panel_size_y = 243
        cfg.panel_size_z = 195
        params = cfg.to_params()
        assert params["NPanelsY"] == 6
        assert params["PanelGapsY"] == [1, 7, 1, 7, 1]
        assert params["PanelSizeY"] == 243

    def test_q_binning_only_emitted_when_set(self):
        cfg = self._cfg()
        assert "QMax" not in cfg.to_params()
        cfg.q_min = 0.1
        cfg.q_max = 15.0
        cfg.q_bin_size = 0.001
        params = cfg.to_params()
        assert params["QMin"] == pytest.approx(0.1)
        assert params["QMax"] == pytest.approx(15.0)
        assert params["QBinSize"] == pytest.approx(0.001)

    def test_end_to_end_writes_params_file(self, tmp_path):
        cfg = self._cfg()
        path = tmp_path / "p.txt"
        write_params_file(path, cfg.to_params())
        text = path.read_text()
        for line_start in ("Lsd ", "BC ", "Wavelength ", "RhoD ",
                           "EtaBinSize ", "p0 ", "p14 "):
            assert line_start in text, f"missing: {line_start}"
