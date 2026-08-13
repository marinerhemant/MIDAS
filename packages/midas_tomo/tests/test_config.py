"""TomoConfig: validation, derived shapes, and parameter-file round-trip.

None of these need the compiled binary, so they run everywhere.
"""

from __future__ import annotations

import pytest

from midas_tomo.config import (
    FILTERS,
    TomoConfig,
    next_power_of_2,
    parse_shift_arg,
)


def _minimal(tmp_path, **kw):
    data = tmp_path / "in.bin"
    data.write_bytes(b"\0" * 16)
    thetas = tmp_path / "thetas.txt"
    thetas.write_text("0\n1\n")
    base = dict(
        data_file=data,
        recon_file=tmp_path / "out",
        det_xdim=128,
        det_ydim=8,
        theta_file=thetas,
    )
    base.update(kw)
    return TomoConfig(**base)


# ----------------------------------------------------------------- helpers
@pytest.mark.parametrize(
    "n,expected",
    [(1, 1), (2, 2), (3, 4), (128, 128), (129, 256), (1475, 2048), (0, 1), (-5, 1)],
)
def test_next_power_of_2(n, expected):
    assert next_power_of_2(n) == expected


def test_parse_shift_scalar():
    assert parse_shift_arg(2.5) == (2.5, 2.5, 1.0, 1)


def test_parse_shift_range_matches_c_arithmetic():
    # The C computes round(|end-start|/step) + 1.
    assert parse_shift_arg([-2, 2, 0.5])[3] == 9
    assert parse_shift_arg([0, 0, 0.5])[3] == 1


def test_parse_shift_rejects_zero_step():
    with pytest.raises(ValueError, match="non-zero"):
        parse_shift_arg([0, 1, 0])


def test_parse_shift_rejects_wrong_length():
    with pytest.raises(ValueError, match=r"scalar or \[start, end, step\]"):
        parse_shift_arg([0, 1])


# -------------------------------------------------------------- validation
def test_valid_config_has_no_problems(tmp_path):
    assert _minimal(tmp_path).validate() == []


def test_requires_exactly_one_theta_source(tmp_path):
    cfg = _minimal(tmp_path, theta_range=(0.0, 180.0, 1.0))  # plus theta_file
    assert any("exactly one of theta_file" in p for p in cfg.validate())

    cfg = _minimal(tmp_path)
    cfg.theta_file = None
    assert any("exactly one of theta_file" in p for p in cfg.validate())


def test_rejects_unknown_filter(tmp_path):
    problems = _minimal(tmp_path, filter_nr=99).validate()
    assert any("filter_nr must be one of" in p for p in problems)
    # every documented filter is accepted
    for k in FILTERS:
        assert _minimal(tmp_path, filter_nr=k).validate() == []


def test_rejects_nonpositive_dims(tmp_path):
    assert any("det_xdim" in p for p in _minimal(tmp_path, det_xdim=0).validate())
    assert any("det_ydim" in p for p in _minimal(tmp_path, det_ydim=-1).validate())


def test_rejects_even_vo_window(tmp_path):
    # The Vo median filters assume an odd, centred window; the C does not check.
    problems = _minimal(
        tmp_path, do_stripe_removal=True, stripe_la_size=60
    ).validate()
    assert any("stripe_la_size must be odd" in p for p in problems)


def test_even_vo_window_ignored_when_stripe_removal_off(tmp_path):
    assert _minimal(tmp_path, stripe_la_size=60).validate() == []


def test_check_raises_with_all_problems(tmp_path):
    cfg = _minimal(tmp_path, det_xdim=0, filter_nr=99)
    with pytest.raises(ValueError) as exc:
        cfg.check()
    assert "det_xdim" in str(exc.value)
    assert "filter_nr" in str(exc.value)


# ----------------------------------------------------------------- derived
def test_recon_xdim_padding(tmp_path):
    assert _minimal(tmp_path, det_xdim=1475).recon_xdim == 2048
    assert _minimal(tmp_path, det_xdim=1475, extra_pad=True).recon_xdim == 4096


def test_n_shifts(tmp_path):
    assert _minimal(tmp_path).n_shifts == 1
    assert _minimal(tmp_path, shift_values=(-2.0, 2.0, 0.5)).n_shifts == 9


def test_rejects_odd_shift_count(tmp_path):
    # The engine reconstructs shifts in pairs; an odd count makes it exit
    # non-zero with no useful message, so catch it here instead.
    problems = _minimal(tmp_path, shift_values=(-2.0, 2.0, 1.0)).validate()  # 5
    assert any("reconstructs shifts in pairs" in p for p in problems)


def test_accepts_even_and_single_shift_counts(tmp_path):
    assert _minimal(tmp_path, shift_values=(-2.0, 3.0, 1.0)).validate() == []  # 6
    assert _minimal(tmp_path, shift_values=(1.5, 1.5, 1.0)).validate() == []   # 1


def test_output_path_encodes_shape(tmp_path):
    cfg = _minimal(tmp_path, det_xdim=128, shift_values=(-1.0, 1.0, 1.0))
    p = cfg.output_path(6)
    assert p.name.endswith("_NrShifts_003_NrSlices_00006_XDim_000128_YDim_000128_float32.bin")


def test_output_path_sweep_mode_has_cleanup_prefix(tmp_path):
    cfg = _minimal(tmp_path, det_xdim=128)
    assert "_NrCleanup_004_NrShifts_" in cfg.output_path(4, n_cleanup=4).name


def test_wisdom_paths_are_double_recon_xdim(tmp_path):
    cfg = _minimal(tmp_path, det_xdim=128)
    names = [p.name for p in cfg.wisdom_paths(tmp_path)]
    assert names == ["fftwf_wisdom_1d_256.txt", "fftwf_wisdom_2d_256.txt"]


# ---------------------------------------------------------------- emission
def test_to_param_file_writes_absolute_paths(tmp_path, monkeypatch):
    # Relative paths in the parameter file are a silent-wrong-answer trap:
    # the engine does not resolve them against its own cwd in every path.
    monkeypatch.chdir(tmp_path)
    cfg = _minimal(tmp_path, data_file="in.bin", theta_file="thetas.txt")
    lines = cfg.to_param_file(tmp_path / "p.par").read_text().splitlines()
    for key in ("dataFileName", "reconFileName", "thetaFileName"):
        value = next(l.split(maxsplit=1)[1] for l in lines if l.startswith(key))
        assert value.startswith("/"), f"{key} is not absolute: {value}"


def test_ring_removal_omitted_when_zero(tmp_path):
    # Presence-is-enable in the C: writing `ringRemovalCoeff 0` would still
    # set use_ring_removal = 1.
    lines = _minimal(tmp_path, ring_removal_coeff=None).to_lines()
    assert not any(l.startswith("ringRemovalCoeff") for l in lines)

    lines = _minimal(tmp_path, ring_removal_coeff=1.0).to_lines()
    assert any(l.startswith("ringRemovalCoeff 1.0") for l in lines)


def test_stripe_config_file_supersedes_single_config(tmp_path):
    grid = tmp_path / "grid.txt"
    grid.write_text("3.0 31 11\n")
    lines = _minimal(tmp_path, stripe_config_file=grid, do_stripe_removal=True).to_lines()
    assert any(l.startswith("stripeConfigFile") for l in lines)
    # The single-config keys must not also be present, or the C would read
    # both and the sweep would silently use the scalar values.
    assert not any(l.startswith("stripeSnr") for l in lines)


def test_deterministic_is_not_a_param_file_key(tmp_path):
    # It is a command-line flag; emitting it here would be silently ignored.
    # Compare keywords only -- a substring search matches the tmp_path baked
    # into dataFileName, which is how this assertion first failed.
    keys = {l.split(maxsplit=1)[0].lower() for l in _minimal(tmp_path, deterministic=True).to_lines()}
    assert "deterministic" not in keys


# --------------------------------------------------------------- round-trip
def test_param_file_round_trip(tmp_path):
    cfg = _minimal(
        tmp_path,
        det_xdim=1475,
        det_ydim=64,
        filter_nr=4,
        shift_values=(-2.0, 2.25, 0.25),   # 18 shifts (even)
        do_log=False,
        extra_pad=True,
        auto_centering=False,
        ring_removal_coeff=1.0,
        do_stripe_removal=True,
        stripe_snr=1.5,
        stripe_la_size=61,
        stripe_sm_size=21,
    )
    back = TomoConfig.from_param_file(cfg.to_param_file(tmp_path / "p.par"))

    for attr in (
        "are_sinos", "det_xdim", "det_ydim", "filter_nr", "shift_values",
        "do_log", "extra_pad", "auto_centering", "ring_removal_coeff",
        "do_stripe_removal", "stripe_snr", "stripe_la_size", "stripe_sm_size",
    ):
        assert getattr(back, attr) == getattr(cfg, attr), attr


def test_from_param_file_accepts_legacy_ring_spelling(tmp_path):
    # The C matches with strncmp("ringRemovalCoeff"), so the longer legacy
    # spelling `ringRemovalCoefficient` also matches. The 2023 DT configs use it.
    p = tmp_path / "legacy.par"
    p.write_text(
        "dataFileName /tmp/sinos.bin\n"
        "reconFileName /tmp/recon\n"
        "areSinos 1\n"
        "detXdim 55\n"
        "detYdim 164\n"
        "filter 2\n"
        "thetaRange 180.25 -179.75 -0.25\n"
        "slicesToProcess -1\n"
        "shiftValues 0.000000 0.000000 0.500000\n"
        "ringRemovalCoefficient 1.0\n"
        "doLog 0\n"
        "ExtraPad 1\n"
    )
    cfg = TomoConfig.from_param_file(p)
    assert cfg.ring_removal_coeff == 1.0
    assert cfg.are_sinos is True
    assert cfg.theta_range == (180.25, -179.75, -0.25)
    assert cfg.theta_file is None
    assert cfg.extra_pad is True
    assert cfg.do_log is False
    # 55 translations, ExtraPad -> 2 * next_pow2(55) = 128, matching the
    # reconSize the 2023 DT scripts computed.
    assert cfg.recon_xdim == 128
