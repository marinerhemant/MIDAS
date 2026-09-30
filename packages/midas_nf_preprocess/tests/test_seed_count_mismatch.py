"""A NrOrientations / seed-file mismatch stays an ERROR (wrong-seed-file guard),
but the message must name the fix.

Regression (bt_20id_jul26b, 2026-09-24): a param file copied from before the pipeline
regenerated seeds carried NrOrientations 243129 against a 251545-row file, and the
bare "expected N, found M" gave no hint which side to change.
"""
import pytest

from midas_nf_preprocess.diffr_spots.hkls import read_seed_orientations


def test_mismatch_error_names_the_fix(tmp_path):
    f = tmp_path / "seeds.csv"
    f.write_text("1,0,0,0\n0,1,0,0\n0,0,1,0\n")
    with pytest.raises(ValueError, match=r"expected 5.*found 3.*NrOrientations 3"):
        read_seed_orientations(f, nr_orientations=5)
