"""DetectorGeometry dataclass — JSON + Parameters.txt round-trips."""

from __future__ import annotations

import json

import pytest

from midas_auto_calibrate import DetectorGeometry


def _example_geometry() -> DetectorGeometry:
    return DetectorGeometry(
        lsd=987_654.321,
        ybc=1021.5,
        zbc=1029.25,
        tx=0.0012,
        ty=-0.0034,
        tz=0.056,
        p0=1.234e-6, p1=-2.345e-7, p2=3.456e-8,
        p3=0.0, p4=4.567e-9, p5=0.0, p6=0.0, p7=0.0,
        p8=0.0, p9=0.0, p10=0.0, p11=0.0, p12=0.0, p13=0.0, p14=0.0,
        rhod=180_000.0,
        wavelength=0.172979,
        px=200.0,
        nr_pixels_y=2048,
        nr_pixels_z=2048,
        mean_strain=4.2,
        std_strain=1.3,
    )


class TestJsonRoundTrip:
    def test_serializes_all_fields(self, tmp_path):
        geom = _example_geometry()
        p = geom.to_json(tmp_path / "g.json")
        loaded = json.loads(p.read_text())
        # Every dataclass field present
        expected = set(vars(geom).keys())
        assert expected == set(loaded.keys())

    def test_round_trip_exact(self, tmp_path):
        geom = _example_geometry()
        geom.to_json(tmp_path / "g.json")
        back = DetectorGeometry.from_json(tmp_path / "g.json")
        assert back == geom

    def test_from_dict_ignores_extras(self):
        # Forward-compatible: unknown keys do not crash from_dict.
        data = _example_geometry().to_dict()
        data["future_field"] = "nonsense"
        reparsed = DetectorGeometry.from_dict(data)
        assert reparsed == _example_geometry()


class TestMidasParamsRoundTrip:
    def test_writes_key_value_pairs(self, tmp_path):
        geom = _example_geometry()
        p = geom.to_midas_params(tmp_path / "Parameters.txt")
        text = p.read_text()

        # BC is a single line with two values
        assert any(line.startswith("BC ") and len(line.split()) == 3
                   for line in text.splitlines()), text

        # Every p0..p14 is emitted (even zeros) for unambiguous round-trip
        for i in range(15):
            assert f"p{i} " in text, f"p{i} missing"

        # Key scalars present
        for key in ("Lsd", "tx", "ty", "tz", "RhoD", "Wavelength", "px",
                    "NrPixelsY", "NrPixelsZ"):
            assert f"{key} " in text, f"{key} missing"

    def test_round_trip_preserves_geometry(self, tmp_path):
        geom = _example_geometry()
        path = tmp_path / "Parameters.txt"
        geom.to_midas_params(path)
        back = DetectorGeometry.from_midas_params(path)

        # Fit-quality fields are not emitted by to_midas_params, so compare
        # everything else.
        for attr in ("lsd", "ybc", "zbc", "tx", "ty", "tz",
                     "rhod", "wavelength", "px", "nr_pixels_y", "nr_pixels_z"):
            assert getattr(back, attr) == pytest.approx(getattr(geom, attr), rel=1e-12)
        for i in range(15):
            assert getattr(back, f"p{i}") == pytest.approx(
                getattr(geom, f"p{i}"), rel=1e-12)

    def test_handles_comments_and_blank_lines(self, tmp_path):
        text = (
            "# comment line — ignored\n"
            "\n"
            "Lsd 500000.0  # inline comment ignored\n"
            "BC 512.0 514.5\n"
            "tx 0.01\n"
        )
        path = tmp_path / "Parameters.txt"
        path.write_text(text)

        geom = DetectorGeometry.from_midas_params(path)
        assert geom.lsd == pytest.approx(500_000.0)
        assert geom.ybc == pytest.approx(512.0)
        assert geom.zbc == pytest.approx(514.5)
        assert geom.tx == pytest.approx(0.01)

    def test_extra_kwargs_appended(self, tmp_path):
        geom = _example_geometry()
        path = tmp_path / "Parameters.txt"
        geom.to_midas_params(path, extra={
            "RMin": 10,
            "RMax": 2800,
            "ImTransOpt": [1, 3],
            "MaskFile": "/tmp/mask.tif",
        })
        text = path.read_text()
        assert "RMin 10\n" in text
        assert "RMax 2800\n" in text
        assert "ImTransOpt 1 3\n" in text
        assert "MaskFile /tmp/mask.tif\n" in text


class TestConvenienceAccessors:
    def test_tilts_tuple(self):
        geom = _example_geometry()
        assert geom.tilts == (geom.tx, geom.ty, geom.tz)

    def test_distortion_tuple(self):
        geom = _example_geometry()
        d = geom.distortion
        assert len(d) == 15
        assert d[0] == geom.p0
        assert d[14] == geom.p14
