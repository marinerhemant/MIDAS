"""RelFitRMSE.bin: one float64 per ExtraInfo.bin row, mapped by the row's own SpotID (col 4)."""
import numpy as np
import pytest

from midas_transforms.io.csv import load_rel_fit_rmse, write_rel_fit_rmse_bin


def _run(d, *, link=True):
    # InputAll: SpotID s has OrigSpotID 100+s. Radius numbers its rows differently
    # (SpotID 1..4 with OrigSpotID reversed), so only the OrigSpotID join is right.
    cols = "YLab ZLab Omega GrainRadius SpotID" + (" OrigSpotID" if link else "")
    rows = [f"0 0 0 1 {s}" + (f" {100 + s}" if link else "") for s in range(1, 5)]
    (d / "InputAllExtraInfoFittingAll.csv").write_text(cols + "\n" + "\n".join(rows) + "\n")
    rad = ["SpotID IMax FitRMSE OrigSpotID"]
    for s in range(1, 5):
        o = 100 + (5 - s)                           # Radius row s carries InputAll spot 5-s
        rad.append(f"{s} {10.0 * o} {float(o)} {o}")  # rel = 0.1 for every spot, by construction
    rad[1] = "1 0.0 7.0 104"                         # IMax 0 for InputAll spot 4 -> NaN
    (d / "Radius_StartNr_1_EndNr_9.csv").write_text("\n".join(rad) + "\n")
    # ExtraInfo rows deliberately NOT in SpotID order: row r holds SpotID [3, 1, 4, 2][r]
    ei = np.zeros((4, 16)); ei[:, 4] = [3, 1, 4, 2]
    ei.tofile(d / "ExtraInfo.bin")


def test_rows_follow_extrainfo_spotid(tmp_path):
    _run(tmp_path)
    out, n, known = write_rel_fit_rmse_bin(tmp_path)
    v = np.fromfile(out, dtype="<f8")
    assert (n, known) == (4, 3) and v.size == 4
    # rows hold SpotIDs 3, 1, 4, 2 -> 0.1, 0.1, NaN (IMax 0), 0.1
    np.testing.assert_allclose(v[[0, 1, 3]], 0.1)
    assert np.isnan(v[2])


def test_refuses_without_link(tmp_path):
    _run(tmp_path, link=False)
    assert load_rel_fit_rmse(tmp_path) is None
    with pytest.raises(FileNotFoundError):
        write_rel_fit_rmse_bin(tmp_path)


def test_spot_weights_model_and_file(tmp_path):
    from midas_transforms.io.csv import spot_weights_from_noise_model, write_spot_weights_bin
    knots = [0.0, 0.5, 1.0]
    w = spot_weights_from_noise_model([0.0, 0.5, 2.0, np.nan], knots,
                                      [10, 20, 40], [10, 50, 100], [1.0, 0.9, 2.0])
    np.testing.assert_allclose(w[0], [1, 1, 1])
    np.testing.assert_allclose(w[1], [0.5, 0.2, 1.0])       # omega dip 0.9 -> held at 1.0 (non-decreasing)
    np.testing.assert_allclose(w[2], [0.25, 0.1, 0.5])      # beyond the last knot: held flat
    np.testing.assert_allclose(w[3], [1, 1, 1])             # unknown rel -> 1
    _run(tmp_path)
    out, n, known = write_spot_weights_bin(tmp_path, [0.0, 0.2], [1, 2], [1, 4], [1, 1])
    v = np.fromfile(out, dtype="<f8").reshape(-1, 3)
    assert v.shape == (4, 3) and (n, known) == (4, 3)
    np.testing.assert_allclose(v[0], [1 / 1.5, 1 / 2.5, 1.0])  # rel 0.1 -> sigma 1.5 / 2.5 / 1
    np.testing.assert_allclose(v[2], [1, 1, 1])             # the IMax-0 spot: rel NaN -> 1
