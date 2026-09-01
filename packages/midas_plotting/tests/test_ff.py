"""Far-field Grains.csv reading and plotting."""
from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from midas_plotting import ff                                   # noqa: E402
from midas_plotting.grains import read_grains                   # noqa: E402
from midas_plotting.ipf import (                                # noqa: E402
    direction_rgb, ipf_rgb, ipf_rgb_from_matrix,
)

# The CURRENT width is 53: the six DiffPos/Ome/Angle Pre/Post diagnostics were
# appended on 2026-08-21. The fixture used to stop at 47, which is why nothing
# here noticed the widening.
_DATA_COLS = ([f"O{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
              + ["X", "Y", "Z", "a", "b", "c", "alpha", "beta", "gamma",
                 "DiffPos", "DiffOme", "DiffAngle",
                 "GrainRadius", "Confidence"]
              + [f"eFab{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
              + [f"eKen{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
              + ["RMSErrorStrain", "PhaseNr", "Eul0", "Eul1", "Eul2",
                 "DiffPosPre", "DiffOmePre", "DiffAnglePre",
                 "DiffPosPost", "DiffOmePost", "DiffAnglePost"])
_COLS = ["ID"] + _DATA_COLS


def _write_grains(tmp_path, n=4, sg=225, break_orientation=False,
                  *, trailing_tab=False, id_token="ID", sep="\t"):
    """A Grains.csv with the real preamble shape, incl. tab-indented phase info.

    Parameters
    ----------
    trailing_tab : bool
        Terminate every data row with a tab, which is what the C
        ``ProcessGrains`` actually writes. A reader that does ``split("\t")``
        then sees a spurious empty final field and ``float("")`` raises. Every
        real file under ``midas_stress/dev/paper3/runs/*/data/*MPa/`` and the
        bundled ``GrainsSim.csv`` is in this form, and this fixture's omission
        of it is why the crash shipped.
    id_token : "ID" or "GrainID"
        Both spellings are written by live MIDAS writers.
    sep : column separator for the header and data rows.
    """
    from midas_stress.orientation import euler_to_orient_mat_batch

    rng = np.random.default_rng(0)
    eul = rng.uniform(0, 2 * np.pi, (n, 3))
    eul[:, 1] = rng.uniform(0, np.pi, n)
    om = np.asarray(euler_to_orient_mat_batch(eul)).reshape(-1, 3, 3)
    if break_orientation:
        om = om[::-1]                       # deliberately inconsistent

    lines = [
        f"%NumGrains {n}", "%BeamCenter 0.0 0.0", "%BeamThickness 0.0",
        "%GlobalPosition 0.0", "%NumPhases 1", "%PhaseInfo",
        f"%\tSpaceGroup:{sg}",
        "%\tLattice Parameter:4.078200\t4.078200\t4.078200\t"
        "90.000000\t90.000000\t90.000000",
        "%" + sep.join([id_token] + _DATA_COLS),
    ]
    for k in range(n):
        row = [str(k + 1)] + [f"{v:.6f}" for v in om[k].reshape(-1)]
        row += [f"{v:.6f}" for v in rng.uniform(-200, 200, 3)]          # X Y Z
        row += ["4.0782"] * 3 + ["90.0"] * 3                            # lattice
        row += ["150.0", "0.05", "0.08"]                                # diffs
        row += [f"{40.0 + 10 * k:.4f}", "1.0"]                          # R, conf
        row += [f"{v:.4f}" for v in rng.normal(200, 50, 9)]             # eFab
        row += [f"{v:.4f}" for v in rng.normal(200, 50, 9)]             # eKen
        row += ["400.0", "1"]
        row += [f"{v:.9f}" for v in eul[k]]
        row += ["151.0", "0.06", "0.09", "150.0", "0.05", "0.08"]       # pre/post
        assert len(row) == len(_COLS)
        lines.append(sep.join(row) + ("\t" if trailing_tab else ""))
    p = tmp_path / "Grains.csv"
    p.write_text("\n".join(lines) + "\n")
    return p, eul, om


# ── reader ──────────────────────────────────────────────────────────────────
def test_reads_columns_and_header(tmp_path):
    p, eul, om = _write_grains(tmp_path, n=5)
    g = read_grains(p)
    assert len(g) == 5 and g.n_grains == 5
    assert np.allclose(g.euler, eul, atol=1e-6)
    assert np.allclose(g.orient_mat, om, atol=1e-5)
    assert g.space_group == 225
    assert np.allclose(g.lattice_parameter, [4.0782] * 3 + [90.0] * 3)
    assert g.header["NumGrains"] == "5"
    assert np.allclose(g.radius, [40.0, 50.0, 60.0, 70.0, 80.0])


def test_tab_indented_phase_lines_do_not_crash(tmp_path):
    """`%\\tSpaceGroup:225` has an empty first tab-field -- an early version
    of the parser raised IndexError on exactly this line."""
    p, _, _ = _write_grains(tmp_path)
    g = read_grains(p)
    assert g.space_group == 225


def test_columns_looked_up_by_name_not_position(tmp_path):
    """Reordering columns must not change what is read."""
    p, _, _ = _write_grains(tmp_path, n=3)
    ref = read_grains(p)
    lines = p.read_text().splitlines()
    hdr_i = next(i for i, l in enumerate(lines) if "O11" in l)
    cols = lines[hdr_i].lstrip("%").split("\t")
    order = list(range(len(cols)))[::-1]
    lines[hdr_i] = "%" + "\t".join(cols[k] for k in order)
    for i in range(hdr_i + 1, len(lines)):
        f = lines[i].split("\t")
        lines[i] = "\t".join(f[k] for k in order)
    p.write_text("\n".join(lines) + "\n")
    got = read_grains(p)
    assert np.allclose(got.pos, ref.pos)
    assert np.allclose(got.radius, ref.radius)
    assert np.allclose(got.orient_mat, ref.orient_mat)


def test_orientation_mismatch_warns(tmp_path):
    """The guard against the 0.5.6-style column-rotation class of bug."""
    p, _, _ = _write_grains(tmp_path, n=4, break_orientation=True)
    with pytest.warns(RuntimeWarning, match="disagree"):
        read_grains(p)


def test_orientation_check_can_be_disabled(tmp_path):
    p, _, _ = _write_grains(tmp_path, n=4, break_orientation=True)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        read_grains(p, check_orientation=False)


@pytest.mark.parametrize("trailing_tab", [False, True])
@pytest.mark.parametrize("id_token", ["ID", "GrainID"])
def test_trailing_tab_and_both_id_spellings(tmp_path, trailing_tab, id_token):
    """The shipped crash: `line.split("\t")` on a tab-terminated row.

    The C ``ProcessGrains`` ends every data row with a tab, so a tab-split
    yields one extra empty field and ``float("")`` raised
    ``could not convert string to float: ''`` on every real file under
    ``midas_stress/dev/paper3/runs/*/data/*MPa/Grains.csv``.
    """
    p, eul, om = _write_grains(tmp_path, n=4, trailing_tab=trailing_tab,
                               id_token=id_token)
    g = read_grains(p)
    assert len(g) == 4
    assert len(g.columns) == 53
    np.testing.assert_array_equal(g.ids, [1, 2, 3, 4])
    assert np.allclose(g.euler, eul, atol=1e-6)
    assert np.allclose(g.orient_mat, om, atol=1e-5)
    assert np.allclose(g.radius, [40.0, 50.0, 60.0, 70.0])


def test_space_separated_file_is_readable(tmp_path):
    """NF ``mic2grains`` seed files are space-separated, not tab-separated."""
    p, eul, _ = _write_grains(tmp_path, n=3, sep=" ")
    g = read_grains(p)
    assert len(g) == 3 and len(g.columns) == 53
    assert np.allclose(g.euler, eul, atol=1e-6)


def test_current_width_exposes_pre_post_residual_columns(tmp_path):
    """The 2026-08-21 widening: 47 -> 53. The extra columns must be readable
    by name, and must not displace anything that was already there."""
    p, _, _ = _write_grains(tmp_path, n=2)
    g = read_grains(p)
    idx = {c: i for i, c in enumerate(g.columns)}
    for c in ("DiffPosPre", "DiffOmePre", "DiffAnglePre",
              "DiffPosPost", "DiffOmePost", "DiffAnglePost"):
        assert c in idx
    np.testing.assert_allclose(g.raw[:, idx["DiffPosPre"]], 151.0)
    np.testing.assert_allclose(g.raw[:, idx["DiffPosPost"]], 150.0)
    # ...and the pre-existing block is untouched.
    np.testing.assert_allclose(g.diff_pos, 150.0)
    np.testing.assert_allclose(g.diff_ome, 0.05)


def test_missing_header_is_an_error(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("1\t2\t3\n")
    with pytest.raises(ValueError, match="no column header"):
        read_grains(p)


# ── strain units (regression) ───────────────────────────────────────────────
def test_strain_is_not_rescaled(tmp_path):
    """Grains.csv already stores MICROSTRAIN.

    An early version multiplied by 1e6, turning a ~250 µε grain into 2.5e8 µε.
    """
    p, _, _ = _write_grains(tmp_path, n=4)
    g = read_grains(p)
    raw11 = g.strain_fab[:, 0, 0]
    assert np.allclose(ff.strain_scalar(g, "11", convention="fab"), raw11)
    assert np.abs(ff.strain_scalar(g, "hydrostatic")).max() < 1e4


def test_strain_scalars(tmp_path):
    p, _, _ = _write_grains(tmp_path, n=4)
    g = read_grains(p)
    e = g.strain("fab")
    assert np.allclose(ff.strain_scalar(g, "hydrostatic", convention="fab"),
                       np.trace(e, axis1=1, axis2=2) / 3.0)
    e = g.strain()          # default convention, checked below
    assert np.allclose(ff.strain_scalar(g, "hydrostatic"),
                       np.trace(e, axis1=1, axis2=2) / 3.0)
    assert (ff.strain_scalar(g, "vonmises") >= 0).all()
    assert np.allclose(ff.strain_scalar(g, "13"), e[:, 0, 2])
    with pytest.raises(ValueError):
        ff.strain_scalar(g, "nonsense")
    with pytest.raises(ValueError):
        g.strain("nonsense")


# ── IPF ─────────────────────────────────────────────────────────────────────
def test_matrix_and_euler_entry_points_agree(tmp_path):
    p, eul, om = _write_grains(tmp_path, n=6)
    assert np.allclose(ipf_rgb(eul, 225), ipf_rgb_from_matrix(om, 225))


def test_cubic_triangle_corners_are_primary_colours():
    rgb = direction_rgb(np.array([[0, 0, 1.0], [1, 0, 1.0], [1, 1, 1.0]]), 225)
    assert np.allclose(rgb[0], [1, 0, 0], atol=1e-6)
    assert np.allclose(rgb[1], [0, 1, 0], atol=1e-6)
    assert np.allclose(rgb[2], [0, 0, 1], atol=1e-6)


def test_direction_rgb_is_normalisation_invariant():
    a = direction_rgb(np.array([[1, 1, 3.0]]), 225)
    b = direction_rgb(np.array([[2, 2, 6.0]]), 225)
    assert np.allclose(a, b)


@pytest.mark.parametrize("sg", [225, 194])
def test_ipf_legend_corners_lie_on_the_triangle(sg):
    """Corners must be normalised before projecting.

    Projecting the raw index triple puts [111] at (0.5, 0.5) rather than
    (0.366, 0.366), which floated the marker and label outside the colours.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ff.ipf_legend(sg, ax)
    xs = [ln.get_xdata()[0] for ln in ax.lines]
    ys = [ln.get_ydata()[0] for ln in ax.lines]
    x0, x1 = ax.images[0].get_extent()[:2]
    y0, y1 = ax.images[0].get_extent()[2:]
    assert all(x0 <= x <= x1 for x in xs), (xs, (x0, x1))
    assert all(y0 <= y <= y1 for y in ys), (ys, (y0, y1))
    if sg == 225:
        assert np.isclose(max(xs), 1 / (1 + np.sqrt(0.5)) * np.sqrt(0.5),
                          atol=0.02) or np.isclose(max(xs), 0.5, atol=0.02)
    plt.close(fig)


# ── plots run ───────────────────────────────────────────────────────────────
def test_plots_run_and_return_axes(tmp_path):
    import matplotlib.pyplot as plt
    p, _, _ = _write_grains(tmp_path, n=8)
    g = read_grains(p)
    for call in (
        lambda: ff.grain_map(g),
        lambda: ff.grain_map(g, plane="xz", color="completeness"),
        lambda: ff.grain_map(g, color="radius"),
        lambda: ff.grain_size_distribution(g),
        lambda: ff.completeness_hist(g),
        lambda: ff.strain_map(g, kind="hydrostatic"),
        lambda: ff.strain_distribution(g),
        lambda: ff.pole_figure(g),
        lambda: ff.pole_figure(g, projection="equal_area"),
    ):
        ax = call()
        assert ax is not None
        plt.close(ax.figure)
    fig = ff.summary(g)
    assert fig is not None
    plt.close(fig)


def test_space_group_comes_from_the_file(tmp_path):
    """Must not silently default to cubic for a hexagonal sample."""
    p, _, _ = _write_grains(tmp_path, n=4, sg=194)
    g = read_grains(p)
    assert g.space_group == 194
    ax = ff.grain_map(g)                       # uses 194, no argument passed
    assert ax is not None


def test_missing_space_group_raises_rather_than_assuming(tmp_path):
    p, _, _ = _write_grains(tmp_path, n=3)
    txt = [l for l in p.read_text().splitlines() if "SpaceGroup" not in l]
    p.write_text("\n".join(txt) + "\n")
    g = read_grains(p)
    assert g.space_group is None
    with pytest.raises(ValueError, match="no SpaceGroup"):
        ff.grain_map(g)


def test_bad_plane_rejected(tmp_path):
    p, _, _ = _write_grains(tmp_path, n=3)
    with pytest.raises(ValueError, match="plane must be"):
        ff.grain_map(read_grains(p), plane="qq")


# ── CLI sniffing ────────────────────────────────────────────────────────────
def test_ff_detected_by_content_not_filename(tmp_path):
    from midas_plotting.cli import _looks_like_ff
    p, _, _ = _write_grains(tmp_path)
    renamed = tmp_path / "totally_not_grains.txt"
    renamed.write_text(p.read_text())
    assert _looks_like_ff(renamed)
    other = tmp_path / "x.mic"
    other.write_text("1 2 3 4 5\n")
    assert not _looks_like_ff(other)
    assert not _looks_like_ff(tmp_path / "missing.csv")


# ── strain convention default (regression) ──────────────────────────────────
def test_strain_default_convention_is_eken(tmp_path):
    """The default must be eKen, matching ``midas_stress.io``'s ``strain`` key.

    midas_plotting defaulted to eFab while midas_stress.io mapped its plain
    ``strain`` key to eKen and documented eKen as the recommendation. Both
    sites labelled their choice, so nothing raised and nothing was mislabelled
    -- but a strain map from here and a stress from there, on the SAME
    Grains.csv, were different tensors.
    """
    p, _, _ = _write_grains(tmp_path, n=6)
    g = read_grains(p)
    # The fixture draws eFab and eKen independently, so a wrong default shows.
    assert not np.allclose(g.strain_fab, g.strain_ken)
    np.testing.assert_allclose(g.strain(), g.strain_ken)
    for kind in ("hydrostatic", "vonmises", "11", "23"):
        np.testing.assert_allclose(ff.strain_scalar(g, kind),
                                   ff.strain_scalar(g, kind, convention="ken"))


def test_strain_default_agrees_with_midas_stress(tmp_path):
    """Same file, same tensor, across the two packages.

    midas_stress returns dimensionless strain (it divides the stored
    microstrain by 1e6); midas_plotting keeps microstrain. That factor is the
    ONLY difference the two defaults may have.
    """
    ms_io = pytest.importorskip("midas_stress.io")
    p, _, _ = _write_grains(tmp_path, n=6)
    g = read_grains(p)
    ms = ms_io.read_grains_csv(str(p))
    np.testing.assert_allclose(g.strain(), ms["strain"] * 1e6, rtol=1e-9)
