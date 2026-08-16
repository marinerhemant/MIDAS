"""Tests for the cross-CLI preflight.

Each case is a failure that actually reached a user and was reported as "your
code is broken".
"""
import os
import sys

import pytest

from midas_params.preflight import (
    ENV_DISABLE,
    MidasArgumentParser,
    check_device,
    check_environment,
    check_hdf5_group,
    check_paths,
    preflight,
)


# ─── paths ───────────────────────────────────────────────────────────────────
def test_missing_path_is_reported(tmp_path):
    out = check_paths({"--params": tmp_path / "nope.txt"})
    assert len(out) == 1 and "does not exist" in out[0]


def test_missing_path_suggests_a_near_neighbour(tmp_path):
    (tmp_path / "ps_ti7al_bh100_ol25_aug26.txt").write_text("x")
    out = check_paths({"--params": tmp_path / "ps_ti7al_bh100_ol25_aug27.txt"})
    assert "Did you mean 'ps_ti7al_bh100_ol25_aug26.txt'" in out[0]


def test_missing_parent_directory_is_called_out(tmp_path):
    out = check_paths({"--params": tmp_path / "nodir" / "ps.txt"})
    assert "directory" in out[0] and "does not exist" in out[0]


def test_none_and_empty_values_are_skipped():
    assert check_paths({"--a": None, "--b": ""}) == []


def test_empty_file_is_reported(tmp_path):
    f = tmp_path / "ps.txt"; f.write_text("")
    assert "empty" in check_paths({"--params": f})[0]


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores the read bit")
def test_unreadable_file_is_distinguished_from_missing(tmp_path):
    """Permission denied and not-found send you to completely different places,
    so they must not collapse into one message."""
    f = tmp_path / "secret.h5"; f.write_text("x"); f.chmod(0o000)
    try:
        out = check_paths({"--image": f})
        assert len(out) == 1 and "not readable" in out[0]
    finally:
        f.chmod(0o644)


def test_existing_readable_file_is_silent(tmp_path):
    f = tmp_path / "ps.txt"; f.write_text("Lsd 1000000\n")
    assert check_paths({"--params": f}) == []


# ─── hdf5 groups ─────────────────────────────────────────────────────────────
def test_wrong_hdf5_group_lists_the_real_keys(tmp_path):
    h5py = pytest.importorskip("h5py")
    import numpy as np
    p = tmp_path / "c.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("exchange/data", data=np.zeros((2, 2, 2)))
        f.create_group("WM")
    out = check_hdf5_group(p, "exchange/nope")
    assert len(out) == 1
    assert "exchange" in out[0] and "WM" in out[0]


def test_right_hdf5_group_is_silent(tmp_path):
    h5py = pytest.importorskip("h5py")
    import numpy as np
    p = tmp_path / "c.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("exchange/data", data=np.zeros((2, 2, 2)))
    assert check_hdf5_group(p, "exchange/data") == []


# ─── environment ─────────────────────────────────────────────────────────────
def test_environment_reports_prefix_and_version():
    rep = check_environment("python", package="pytest")
    assert rep.prefix == sys.prefix
    assert rep.version is not None
    assert "pytest" in rep.one_line()


def test_unknown_package_is_noted_not_raised():
    rep = check_environment("python", package="definitely-not-installed-xyz")
    assert rep.version is None
    assert any("not installed" in n for n in rep.notes)


def test_device_check_is_silent_unless_cuda_requested():
    assert check_device(None) == []
    assert check_device("cpu") == []


# ─── argparse ────────────────────────────────────────────────────────────────
def _parser():
    p = MidasArgumentParser("demo", package="pytest")
    p.add_argument("paramsfile")
    p.add_argument("--mode", choices=("single", "ff"))
    p.add_argument("--dark-group")
    return p


def test_typo_in_flag_suggests_the_real_one(capsys):
    """argparse alone says only 'unrecognized arguments'."""
    with pytest.raises(SystemExit):
        _parser().parse_args(["ps.txt", "--dark-groupp", "x"])
    err = capsys.readouterr().err
    assert "Did you mean --dark-group?" in err


def test_invalid_choice_reports_the_build(capsys):
    """The real cause of `invalid choice: 'ff'` was an older build on PATH, so
    the version and prefix belong in the error."""
    p = MidasArgumentParser("demo", package="pytest")
    p.add_argument("--mode", choices=("single",))
    with pytest.raises(SystemExit):
        p.parse_args(["--mode", "ff"])
    err = capsys.readouterr().err
    assert "invalid choice" in err
    assert "pytest" in err and sys.prefix in err


def test_missing_positional_says_it_is_positional(capsys):
    with pytest.raises(SystemExit):
        _parser().parse_args(["--mode", "single"])
    err = capsys.readouterr().err
    assert "positional" in err


def test_valid_command_line_is_untouched():
    args = _parser().parse_args(["ps.txt", "--mode", "ff", "--dark-group", "g"])
    assert args.paramsfile == "ps.txt" and args.mode == "ff"


# ─── the one call ────────────────────────────────────────────────────────────
def test_preflight_passes_on_a_good_run(tmp_path):
    f = tmp_path / "ps.txt"; f.write_text("Lsd 1000000\n")
    r = preflight(tool="python", package="pytest", paths={"--params": f},
                  verbose=False)
    assert r.ok and bool(r) and r.problems == []


def test_preflight_flags_a_bad_path(tmp_path):
    r = preflight(tool="python", paths={"--params": tmp_path / "nope.txt"},
                  verbose=False)
    assert not r.ok and len(r.problems) == 1


def test_preflight_strict_exits_before_doing_work(tmp_path):
    with pytest.raises(SystemExit):
        preflight(tool="python", paths={"--params": tmp_path / "nope.txt"},
                  strict=True, verbose=False)


def test_preflight_can_be_disabled_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_DISABLE, "1")
    r = preflight(tool="python", paths={"--params": tmp_path / "nope.txt"},
                  strict=True, verbose=False)
    assert r.ok


def test_preflight_skip_flag(tmp_path):
    r = preflight(tool="python", paths={"--params": tmp_path / "nope.txt"},
                  skip=True, strict=True, verbose=False)
    assert r.ok
