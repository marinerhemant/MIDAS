"""Rev-12 tests: console-script CLI wrappers.

We test the ``main()`` entry points directly (not via subprocess) so the
tests are fast and can inspect returned status. Each CLI writes its
JSON output to stdout; we capture with ``capsys``.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest


CIF_NI = textwrap.dedent("""\
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


def _write_cif(tmp_path: Path) -> Path:
    path = tmp_path / "ni.cif"
    path.write_text(CIF_NI)
    return path


def _write_gr(tmp_path: Path, has_sigma: bool = True) -> Path:
    """Emit a synthetic FCC Ni G(r) file for the refine CLI."""
    import torch
    from midas_hkls import Crystal, Atom, Lattice, SpaceGroup
    from midas_pdf.structure import build_pair_list, pdffit_gr
    ni = Crystal(lattice=Lattice(3.524, 3.524, 3.524, 90, 90, 90),
                  space_group=SpaceGroup.from_number(225),
                  atoms=[Atom(element="Ni", fract=(0, 0, 0))],
                  name="Ni").to_torch()
    r = torch.linspace(1.5, 8.0, 150, dtype=torch.float64)
    pairs = build_pair_list(ni, r_max=9.0)
    G = pdffit_gr(ni, r, pairs, scale=1.0, u_iso=0.006)
    rng = torch.Generator().manual_seed(0)
    G_obs = G + 0.02 * torch.randn(G.shape, generator=rng, dtype=torch.float64)
    path = tmp_path / "ni.gr"
    with path.open("w") as f:
        f.write("# r  G(r)  sigma\n")
        for i in range(len(r)):
            row = f"{float(r[i]):.4f}  {float(G_obs[i]):.6f}"
            if has_sigma:
                row += "  0.02"
            f.write(row + "\n")
    return path


# ---------------------------------------------------------------------------
# midas-pdf-cif
# ---------------------------------------------------------------------------

def test_cif_info_prints_json_summary(tmp_path, capsys):
    from midas_pdf.cli.cif_cmd import main
    path = _write_cif(tmp_path)
    rc = main(["info", str(path)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["space_group"]["number"] == 225
    assert abs(out["lattice"]["a"] - 3.524) < 1e-6
    assert out["n_atoms_asu"] == 1
    assert out["atoms_asu"][0]["element"] == "Ni"


def test_cif_convert_round_trip_writes_new_file(tmp_path):
    from midas_pdf.cli.cif_cmd import main
    src = _write_cif(tmp_path)
    dst = tmp_path / "ni_rt.cif"
    rc = main(["convert", str(src), str(dst)])
    assert rc == 0
    assert dst.exists()
    # And the re-read should give the same crystal
    from midas_pdf.cif import read_cif_to_crystal
    c = read_cif_to_crystal(dst)
    assert c.space_group.number == 225


def test_cif_convert_rejects_non_cif_destination(tmp_path):
    from midas_pdf.cli.cif_cmd import main
    src = _write_cif(tmp_path)
    dst = tmp_path / "not_cif.xyz"
    with pytest.raises(SystemExit):
        main(["convert", str(src), str(dst)])


def test_cif_raw_dumps_keys_and_loops(tmp_path, capsys):
    from midas_pdf.cli.cif_cmd import main
    path = _write_cif(tmp_path)
    rc = main(["raw", str(path)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert "_cell_length_a" in out["keys"]
    assert out["n_loops"] == 1


# ---------------------------------------------------------------------------
# midas-pdf-refine
# ---------------------------------------------------------------------------

def test_refine_recovers_lattice_from_synthetic_gr(tmp_path, capsys):
    from midas_pdf.cli.refine_cmd import main
    cif = _write_cif(tmp_path)
    gr = _write_gr(tmp_path)
    rc = main(["--cif", str(cif), "--gr", str(gr),
                "--steps", "100"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert abs(out["fitted"]["a"] - 3.524) < 0.005
    assert out["chi2_reduced"] < 5.0
    assert out["n_data_points"] > 100


def test_refine_reads_sigma_column(tmp_path, capsys):
    from midas_pdf.cli.refine_cmd import main
    cif = _write_cif(tmp_path)
    gr = _write_gr(tmp_path, has_sigma=True)
    rc = main(["--cif", str(cif), "--gr", str(gr), "--steps", "50"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert "uncertainty" in out
    for k in ("a", "u_iso", "scale"):
        assert k in out["fitted"]


def test_refine_r_window_narrows_data(tmp_path, capsys):
    from midas_pdf.cli.refine_cmd import main
    cif = _write_cif(tmp_path)
    gr = _write_gr(tmp_path)
    rc = main(["--cif", str(cif), "--gr", str(gr),
                "--r-min", "2.5", "--r-max", "5.0", "--steps", "50"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_data_points"] < 100     # narrower window than default


# ---------------------------------------------------------------------------
# midas-pdf-rmc
# ---------------------------------------------------------------------------

def _write_rmc_gr(tmp_path: Path) -> Path:
    """Synthetic supercell G(r) target for the RMC CLI."""
    import torch
    from midas_hkls import Crystal, Atom, Lattice, SpaceGroup
    from midas_pdf.rmc import Supercell, supercell_G_r
    ni = Crystal(lattice=Lattice(3.524, 3.524, 3.524, 90, 90, 90),
                  space_group=SpaceGroup.from_number(225),
                  atoms=[Atom(element="Ni", fract=(0, 0, 0))],
                  name="Ni").to_torch()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    r = torch.linspace(1.5, 4.5, 100, dtype=torch.float64)
    G = supercell_G_r(sc, r, u_iso=0.005, r_max=5.5)
    path = tmp_path / "target.gr"
    with path.open("w") as f:
        f.write("# r  G(r)\n")
        for i in range(len(r)):
            f.write(f"{float(r[i]):.4f}  {float(G[i]):.6f}\n")
    return path


def test_rmc_cli_runs_and_reports_chi2(tmp_path, capsys):
    from midas_pdf.cli.rmc_cmd import main
    cif = _write_cif(tmp_path)
    gr = _write_rmc_gr(tmp_path)
    rc = main(["--cif", str(cif), "--gr", str(gr),
                "--size", "3", "--moves", "100",
                "--min-distance", "1.5", "--seed", "42"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["supercell"]["n_atoms"] == 4 * 3 ** 3     # FCC 4 × 3³
    assert out["moves"]["n_attempted"] == 100
    assert 0 <= out["moves"]["acceptance_ratio"] <= 1
    assert "initial" in out["chi2"] and "final" in out["chi2"]
    assert out["coordination_first_shell"]["Z_after_mean"] > 0


def test_rmc_cli_writes_output_cif(tmp_path, capsys):
    from midas_pdf.cli.rmc_cmd import main
    cif = _write_cif(tmp_path)
    gr = _write_rmc_gr(tmp_path)
    out_cif = tmp_path / "refined.cif"
    rc = main(["--cif", str(cif), "--gr", str(gr),
                "--size", "3", "--moves", "50",
                "--min-distance", "1.5", "--seed", "0",
                "--output", str(out_cif)])
    assert rc == 0
    assert out_cif.exists()
    # The output CIF is P1 (every atom explicit)
    from midas_pdf.cif import read_cif_to_crystal
    c = read_cif_to_crystal(out_cif)
    assert c.space_group.number == 1
    assert len(c.atoms) == 4 * 3 ** 3


def test_rmc_cli_supports_cluster_moves(tmp_path, capsys):
    from midas_pdf.cli.rmc_cmd import main
    cif = _write_cif(tmp_path)
    gr = _write_rmc_gr(tmp_path)
    rc = main(["--cif", str(cif), "--gr", str(gr),
                "--size", "3", "--moves", "50",
                "--move-types", "displace+cluster",
                "--min-distance", "1.5", "--seed", "1"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["moves"]["kind"] == "displace+cluster"


def test_rmc_cli_supports_all_moves(tmp_path, capsys):
    from midas_pdf.cli.rmc_cmd import main
    cif = _write_cif(tmp_path)
    gr = _write_rmc_gr(tmp_path)
    rc = main(["--cif", str(cif), "--gr", str(gr),
                "--size", "3", "--moves", "50",
                "--move-types", "all",
                "--min-distance", "1.5", "--seed", "2"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["moves"]["kind"] == "all"


def test_rmc_cli_rejects_unknown_move_types(tmp_path):
    from midas_pdf.cli.rmc_cmd import main
    cif = _write_cif(tmp_path)
    gr = _write_rmc_gr(tmp_path)
    with pytest.raises(SystemExit):
        # argparse rejects unknown --move-types choices before we ever run
        main(["--cif", str(cif), "--gr", str(gr),
              "--size", "3", "--moves", "10", "--move-types", "widget"])
