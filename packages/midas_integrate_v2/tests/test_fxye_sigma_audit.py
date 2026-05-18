"""Item 14 — FXYE σ audit + ESG writer.

Confirms:
- write_fxye column 3 is σ (one stddev), not σ². ESD == σ per GSAS conv.
- write_fxye accepts both centidegree and degree input via x_unit.
- BANK statement is properly formed.
- write_esg emits a CIF-like loop_ block MAUD/MILK can ingest.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from midas_integrate_v2.io import write_esg, write_fxye


def _make_profile(n=512, two_theta_min=1.0, two_theta_step=0.01):
    tth_deg = two_theta_min + np.arange(n, dtype=np.float64) * two_theta_step
    intensity = 1000.0 + 100.0 * np.sin(2.0 * np.pi * tth_deg / 5.0)
    sigma = np.sqrt(np.maximum(intensity, 1.0))
    return tth_deg, intensity, sigma


def test_fxye_sigma_is_stddev_not_variance(tmp_path: Path):
    tth_deg, I, sig = _make_profile()
    out = tmp_path / "ceo2.fxye"
    write_fxye(
        out,
        r_axis=tth_deg,
        intensity=I,
        sigma=sig,
        x_unit="degrees_2theta",
    )
    raw = out.read_text().splitlines()
    # Skip 80-char title + BANK statement + any '# ' header lines
    data_lines = [ln for ln in raw[2:] if not ln.startswith("#")]
    assert len(data_lines) == len(I)
    # Parse last line, assert third column matches sigma (not sigma**2)
    last = data_lines[-1].split()
    x_centideg = float(last[0])
    y = float(last[1])
    e = float(last[2])
    assert x_centideg == pytest.approx(tth_deg[-1] * 100.0, rel=1e-5)
    assert y == pytest.approx(I[-1], rel=1e-5)
    assert e == pytest.approx(sig[-1], rel=1e-5)
    # The σ written must match input σ; if a buggy implementation wrote
    # variance we'd see e ≈ sig**2 which differs by ~30× for I ~ 1000.
    assert abs(e - sig[-1] ** 2) > 1.0


def test_fxye_centideg_default(tmp_path: Path):
    tth_centideg = np.linspace(100.0, 5000.0, 256)
    I = np.full_like(tth_centideg, 1.0)
    sig = np.full_like(tth_centideg, 0.1)
    out = tmp_path / "default.fxye"
    write_fxye(
        out, r_axis=tth_centideg, intensity=I, sigma=sig
    )
    bank_line = out.read_text().splitlines()[1]
    # BANK 1 256 256 CONST 100.00000 ... FXYE
    assert bank_line.startswith("BANK 1 256 256 CONST")
    assert bank_line.rstrip().endswith("FXYE")
    m = re.search(r"CONST\s+([\d.\-]+)\s+([\d.\-]+)", bank_line)
    assert m is not None
    start = float(m.group(1))
    step = float(m.group(2))
    assert start == pytest.approx(100.0, rel=1e-5)
    assert step == pytest.approx(tth_centideg[1] - tth_centideg[0], rel=1e-5)


def test_fxye_rejects_unknown_x_unit(tmp_path: Path):
    tth_deg, I, sig = _make_profile()
    with pytest.raises(ValueError):
        write_fxye(
            tmp_path / "bad.fxye",
            r_axis=tth_deg, intensity=I, sigma=sig,
            x_unit="Q_invA",
        )


def test_esg_writes_cif_like_loop(tmp_path: Path):
    tth_deg, I, sig = _make_profile()
    out = tmp_path / "ceo2.esg"
    write_esg(
        out,
        two_theta_deg=tth_deg,
        intensity=I,
        sigma=sig,
        wavelength_A=0.1839,
        bank_id=1,
    )
    text = out.read_text()
    # Required CIF-like keys
    assert "_pd_block_id" in text
    assert "_diffrn_radiation_wavelength 0.183900" in text
    assert "_pd_meas_2theta_range_min" in text
    assert "_pd_meas_2theta_range_max" in text
    assert "_pd_meas_2theta_range_inc" in text
    assert "loop_" in text
    assert "_pd_proc_intensity_total" in text
    assert "_pd_proc_intensity_total_su" in text
    # Three-column data follows the loop_
    lines = text.splitlines()
    loop_idx = next(i for i, ln in enumerate(lines) if ln.strip() == "loop_")
    # Skip the three column declarations after loop_
    data_lines = lines[loop_idx + 4:]
    data_lines = [ln for ln in data_lines if ln.strip() and not ln.startswith("#")]
    assert len(data_lines) == len(I)
    last = data_lines[-1].split()
    assert float(last[0]) == pytest.approx(tth_deg[-1], rel=1e-5)
    assert float(last[1]) == pytest.approx(I[-1], rel=1e-5)
    assert float(last[2]) == pytest.approx(sig[-1], rel=1e-5)


def test_esg_warns_on_nonuniform_spacing(tmp_path: Path):
    # Non-uniform 2θ axis (e.g., log-spaced) → expect comment warning
    tth_deg = np.geomspace(1.0, 50.0, 64)
    I = np.ones_like(tth_deg)
    sig = np.full_like(tth_deg, 0.1)
    out = tmp_path / "nonuniform.esg"
    write_esg(
        out,
        two_theta_deg=tth_deg,
        intensity=I,
        sigma=sig,
        wavelength_A=0.5,
    )
    text = out.read_text()
    assert "WARNING" in text and "uniformly spaced" in text


def test_esg_rejects_mismatched_shapes(tmp_path: Path):
    tth = np.linspace(1.0, 50.0, 100)
    I = np.ones(100)
    sig = np.ones(99)
    with pytest.raises(ValueError):
        write_esg(
            tmp_path / "bad.esg",
            two_theta_deg=tth, intensity=I, sigma=sig,
            wavelength_A=0.5,
        )
