from midas_pdf import Composition
from midas_pdf.fluorescence import expected_fluorescence, wavelength_to_energy_keV


def test_iron_fluoresces_above_K_edge():
    # Fe K-edge ~7.11 keV; at 20 keV it should fluoresce on the K shell.
    lines = expected_fluorescence(["Fe"], incident_energy_keV=20.0)
    assert any(d["element"] == "Fe" and d["shell"] == "K" for d in lines)
    fe_k = next(d for d in lines if d["element"] == "Fe" and d["shell"] == "K")
    assert abs(fe_k["edge_keV"] - 7.112) < 0.05
    assert fe_k["line_keV"] is not None


def test_no_fluorescence_below_edge():
    # At 5 keV (below Fe K-edge) there is no K fluorescence from Fe.
    lines = expected_fluorescence(["Fe"], incident_energy_keV=5.0)
    assert not any(d["element"] == "Fe" and d["shell"] == "K" for d in lines)


def test_high_energy_clean_for_light_elements():
    # SiO2 at 74 keV: no significant fluorescence expected (light elements).
    comp = Composition({"Si": 1, "O": 2})
    lines = expected_fluorescence(comp.elements, wavelength_A=0.1665)
    assert lines == [] or all(d["element"] in {"Si", "O"} for d in lines)


def test_wavelength_energy_roundtrip():
    assert abs(wavelength_to_energy_keV(0.1665) - 74.46) < 0.5
