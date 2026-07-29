"""Rev-15 CLI tests for midas-pdf-multiphase and midas-pdf-coreshell."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest


NI_CIF = textwrap.dedent("""\
    data_FCC_Ni
    _cell_length_a       3.524
    _cell_length_b       3.524
    _cell_length_c       3.524
    _cell_angle_alpha    90
    _cell_angle_beta     90
    _cell_angle_gamma    90
    _symmetry_Int_Tables_number   225

    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    Ni1  Ni  0.0  0.0  0.0
""")

CU_CIF = textwrap.dedent("""\
    data_FCC_Cu
    _cell_length_a       3.615
    _cell_length_b       3.615
    _cell_length_c       3.615
    _cell_angle_alpha    90
    _cell_angle_beta     90
    _cell_angle_gamma    90
    _symmetry_Int_Tables_number   225

    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    Cu1  Cu  0.0  0.0  0.0
""")


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def _synthetic_multiphase_gr(tmp_path: Path) -> Path:
    """Generate a synthetic Ni+Cu G(r) with 60/40 mixing weights."""
    import torch
    from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
    from midas_pdf.multi_phase import multi_phase_gr
    from midas_pdf.structure import build_pair_list

    ni = Crystal(lattice=Lattice(3.524, 3.524, 3.524, 90, 90, 90),
                  space_group=SpaceGroup.from_number(225),
                  atoms=[Atom(element="Ni", fract=(0, 0, 0))]).to_torch()
    cu = Crystal(lattice=Lattice(3.615, 3.615, 3.615, 90, 90, 90),
                  space_group=SpaceGroup.from_number(225),
                  atoms=[Atom(element="Cu", fract=(0, 0, 0))]).to_torch()
    r = torch.linspace(1.5, 8.0, 200, dtype=torch.float64)
    pairs_ni = build_pair_list(ni, r_max=9.0)
    pairs_cu = build_pair_list(cu, r_max=9.0)
    G = multi_phase_gr(
        [ni, cu], [pairs_ni, pairs_cu], r,
        weights=torch.tensor([0.6, 0.4], dtype=torch.float64),
        u_isos=torch.tensor([0.006, 0.008], dtype=torch.float64),
    )
    rng = torch.Generator().manual_seed(0)
    G_obs = G + 0.02 * torch.randn(G.shape, generator=rng, dtype=torch.float64)

    path = tmp_path / "nicu.gr"
    with path.open("w") as f:
        f.write("# r  G(r)  sigma\n")
        for i in range(len(r)):
            f.write(f"{float(r[i]):.4f}  {float(G_obs[i]):.6f}  0.02\n")
    return path


# ---------------------------------------------------------------------------
# midas-pdf-multiphase
# ---------------------------------------------------------------------------

def test_multiphase_recovers_lattice_and_weights(tmp_path, capsys):
    from midas_pdf.cli.multiphase_cmd import main
    ni_cif = _write(tmp_path / "ni.cif", NI_CIF)
    cu_cif = _write(tmp_path / "cu.cif", CU_CIF)
    gr = _synthetic_multiphase_gr(tmp_path)

    rc = main(["--cif", str(ni_cif), str(cu_cif),
                "--gr", str(gr),
                "--steps", "100"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_phases"] == 2
    assert abs(out["fitted"]["a_0"] - 3.524) < 0.01
    assert abs(out["fitted"]["a_1"] - 3.615) < 0.01
    w_sum = out["fitted"]["weight_0"] + out["fitted"]["weight_1"]
    assert abs(w_sum - 1.0) < 1e-6
    assert abs(out["fitted"]["weight_0"] - 0.6) < 0.1
    assert out["chi2_reduced"] < 5.0


def test_multiphase_rejects_single_cif(tmp_path):
    from midas_pdf.cli.multiphase_cmd import main
    ni_cif = _write(tmp_path / "ni.cif", NI_CIF)
    gr = _synthetic_multiphase_gr(tmp_path)
    with pytest.raises(SystemExit):
        main(["--cif", str(ni_cif), "--gr", str(gr)])


def test_multiphase_rejects_wrong_length_weights(tmp_path):
    from midas_pdf.cli.multiphase_cmd import main
    ni_cif = _write(tmp_path / "ni.cif", NI_CIF)
    cu_cif = _write(tmp_path / "cu.cif", CU_CIF)
    gr = _synthetic_multiphase_gr(tmp_path)
    with pytest.raises(SystemExit):
        main(["--cif", str(ni_cif), str(cu_cif),
              "--gr", str(gr), "--weights", "0.5", "--steps", "10"])


# ---------------------------------------------------------------------------
# midas-pdf-coreshell
# ---------------------------------------------------------------------------

def _synthetic_coreshell_gr(tmp_path: Path) -> Path:
    """Generate a synthetic Ni-core / Cu-shell G(r)."""
    import torch
    from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
    from midas_pdf.multi_phase import core_shell_pdf_gr
    from midas_pdf.structure import build_pair_list

    ni = Crystal(lattice=Lattice(3.524, 3.524, 3.524, 90, 90, 90),
                  space_group=SpaceGroup.from_number(225),
                  atoms=[Atom(element="Ni", fract=(0, 0, 0))]).to_torch()
    cu = Crystal(lattice=Lattice(3.615, 3.615, 3.615, 90, 90, 90),
                  space_group=SpaceGroup.from_number(225),
                  atoms=[Atom(element="Cu", fract=(0, 0, 0))]).to_torch()
    r = torch.linspace(1.5, 8.0, 200, dtype=torch.float64)
    ni_pairs = build_pair_list(ni, r_max=9.0)
    cu_pairs = build_pair_list(cu, r_max=9.0)
    G = core_shell_pdf_gr(
        ni, cu, r, ni_pairs, cu_pairs,
        R_core_A=torch.tensor(25.0, dtype=torch.float64),
        shell_thickness_A=torch.tensor(8.0, dtype=torch.float64),
        u_iso_core=torch.tensor(0.006, dtype=torch.float64),
        u_iso_shell=torch.tensor(0.010, dtype=torch.float64),
    )
    rng = torch.Generator().manual_seed(0)
    G_obs = G + 0.02 * torch.randn(G.shape, generator=rng, dtype=torch.float64)

    path = tmp_path / "coreshell.gr"
    with path.open("w") as f:
        f.write("# r  G(r)  sigma\n")
        for i in range(len(r)):
            f.write(f"{float(r[i]):.4f}  {float(G_obs[i]):.6f}  0.02\n")
    return path


def test_coreshell_recovers_lattices(tmp_path, capsys):
    from midas_pdf.cli.coreshell_cmd import main
    ni_cif = _write(tmp_path / "ni.cif", NI_CIF)
    cu_cif = _write(tmp_path / "cu.cif", CU_CIF)
    gr = _synthetic_coreshell_gr(tmp_path)

    rc = main(["--core-cif", str(ni_cif), "--shell-cif", str(cu_cif),
                "--gr", str(gr),
                "--r-core", "25", "--shell-thickness", "8",
                "--steps", "80"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert abs(out["fitted"]["a_core"] - 3.524) < 0.01
    assert abs(out["fitted"]["a_shell"] - 3.615) < 0.01
    assert out["chi2_reduced"] < 5.0


def test_coreshell_writes_all_fitted_geometry_keys(tmp_path, capsys):
    from midas_pdf.cli.coreshell_cmd import main
    ni_cif = _write(tmp_path / "ni.cif", NI_CIF)
    cu_cif = _write(tmp_path / "cu.cif", CU_CIF)
    gr = _synthetic_coreshell_gr(tmp_path)
    rc = main(["--core-cif", str(ni_cif), "--shell-cif", str(cu_cif),
                "--gr", str(gr), "--steps", "30"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    for k in ("a_core", "a_shell", "u_iso_core", "u_iso_shell",
              "R_core_A", "shell_thickness_A", "scale_core", "scale_shell"):
        assert k in out["fitted"], f"missing key {k}"
