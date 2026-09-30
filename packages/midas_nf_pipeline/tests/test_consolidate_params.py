"""extract_nf_params must read leading value tokens only, like params.parse_parameters.

Regression: "GridSize 5 # demo pass; scan step is 2 um" crashed consolidate with
ValueError after a completed nf709 reconstruction (bt_20id_jul26b, 2026-09-24).
"""
import numpy as np

from midas_nf_pipeline.consolidate import extract_nf_params


def test_inline_comment_is_ignored():
    p = extract_nf_params(
        "GridSize         5            # demo pass; scan step is 2 um\n"
        "SpaceGroup 225   # fcc\n"
        "GlobalPosition 0\n"
    )
    assert p["GridSize"] == 5.0
    assert p["SpaceGroupNr"] == 225
    assert p["GlobalPosition"] == 0.0


def test_semicolon_terminated_values_still_parse():
    p = extract_nf_params(
        "LatticeConstant 3.5954 3.5954 3.5954 90 90 90;\n"
        "GridSize 2.000000;\n"
        "NumPhases 1;\n"
    )
    assert np.allclose(p["LatticeConstant"], [3.5954] * 3 + [90] * 3)
    assert p["GridSize"] == 2.0
    assert p["NumPhases"] == 1


def test_comment_lines_and_bare_keys_are_skipped():
    p = extract_nf_params("# GridSize 9\nGridSize\nGridSize 4\n")
    assert p["GridSize"] == 4.0
