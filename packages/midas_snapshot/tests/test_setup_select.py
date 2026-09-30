"""select_setup picks the right (geometry, matrix) pair on synthetic frames."""
import numpy as np
import tifffile

from midas_snapshot.setup_select import select_setup

N = 256
LAM = 0.124


def _cif(a, sg, el):
    hm = {225: "F m -3 m", 229: "I m -3 m"}[sg]
    return f"""data_x
_cell_length_a {a}
_cell_length_b {a}
_cell_length_c {a}
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number {sg}
_symmetry_space_group_name_H-M '{hm}'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
X1 {el} 0 0 0 1
"""


def _geom(lsd):
    return f"Lsd {lsd}\nBC {N/2} {N/2}\npx 172\nWavelength {LAM}\nNrPixelsY {N}\nNrPixelsZ {N}\n"


def test_selects_true_geometry_and_matrix(tmp_path):
    from midas_snapshot.geometry import build_maps, matrix_lines
    (tmp_path / "gA.txt").write_text(_geom(250000))
    (tmp_path / "gB.txt").write_text(_geom(400000))
    (tmp_path / "fcc.cif").write_text(_cif(3.60, 225, "Ni"))
    (tmp_path / "bcc.cif").write_text(_cif(2.87, 229, "Fe"))
    maps = build_maps(str(tmp_path / "gA.txt"), 0.005)          # truth: geometry A, fcc matrix
    lines = matrix_lines(str(tmp_path / "fcc.cif"), LAM, float(np.nanmax(maps.tth))) * 1.01
    rng = np.random.default_rng(0)
    lam = np.full((N, N), 0.1)
    for d in lines:
        t = np.degrees(2 * np.arcsin(LAM / (2 * d)))
        lam += 4.0 * np.exp(-0.5 * ((maps.tth - t) / 0.02) ** 2)   # continuous rings, easy case
    fr = tmp_path / "frames"; fr.mkdir()
    for i in range(5):
        tifffile.imwrite(fr / f"f_{i:03d}.tif", rng.poisson(lam).astype(np.int32))
    res = select_setup(str(fr), {"A": str(tmp_path / "gA.txt"), "B": str(tmp_path / "gB.txt")},
                       {"fcc": str(tmp_path / "fcc.cif"), "bcc": str(tmp_path / "bcc.cif")},
                       flip=None, n_frames=5)
    assert res["best"]["geometry"] == "A" and res["best"]["matrix"] == "fcc"
    assert abs(res["best"]["scale"] - 1.01) < 2e-3
