"""C-written parameter files (FitSetup, the FF pipeline's per-layer
paramstest.txt) end value lines with ';'. parse_params used to drop every such
key silently (float('150.000000;') fails): px, NrPixels read as 0."""
from midas_integrate.params import parse_params


def test_semicolon_terminated_values_parse(tmp_path):
    f = tmp_path / "paramstest.txt"
    f.write_text("px 150.000000;\nNrPixelsY 2880;\nNrPixelsZ 2880;\n"
                 "Wavelength 0.172979;\nBC 1430.43 1342.48;\n")
    p = parse_params(f)
    assert (p.pxY, p.pxZ) == (150.0, 150.0)
    assert (p.NrPixelsY, p.NrPixelsZ) == (2880, 2880)
    assert p.Wavelength == 0.172979
    assert (p.BC_y, p.BC_z) == (1430.43, 1342.48)


def test_inline_comment_stripped(tmp_path):
    f = tmp_path / "ps.txt"
    f.write_text("px 200 # micrometres\n")
    assert parse_params(f).pxY == 200.0
