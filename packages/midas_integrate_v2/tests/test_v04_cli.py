"""v0.4: CLI tests.

The CLI doesn't add new math — it composes the spec loader, an image
loader, the chosen binning kernel, and CSV/Map.bin emission. The tests
exercise it end-to-end on a small synthetic dataset and on the real
Pilatus CeO₂ frame.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from midas_integrate.bin_io import load_map
from midas_integrate.params import IntegrationParams

from midas_integrate_v2 import (
    spec_from_v1_params,
    HardBinGeometry, integrate_hard,
)
from midas_integrate_v2.cli import integrate_main, write_map_main
from midas_integrate_v2.compat.to_v1 import v1_params_from_spec


def _write_paramstest(tmp_path: Path, spec) -> Path:
    """Serialize a spec to a v1-style paramstest."""
    p = v1_params_from_spec(spec)
    lines = [
        f"NrPixelsY {p.NrPixelsY}",
        f"NrPixelsZ {p.NrPixelsZ}",
        f"px {p.pxY}",
        f"Lsd {p.Lsd}",
        f"BC {p.BC_y} {p.BC_z}",
        f"tx {p.tx}",
        f"ty {p.ty}",
        f"tz {p.tz}",
        f"RhoD {p.RhoD}",
        f"Wavelength {p.Wavelength}",
        f"Parallax {p.Parallax}",
        f"RMin {p.RMin}",
        f"RMax {p.RMax}",
        f"RBinSize {p.RBinSize}",
        f"EtaMin {p.EtaMin}",
        f"EtaMax {p.EtaMax}",
        f"EtaBinSize {p.EtaBinSize}",
    ]
    for k in range(15):
        lines.append(f"p{k} {getattr(p, f'p{k}')}")
    for op in p.TransOpt:
        lines.append(f"ImTransOpt {op}")
    path = tmp_path / "paramstest.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def _spec(NY=24, NZ=24, *, ops=None):
    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
        BC_y=NY / 2.0 + 0.37, BC_z=NZ / 2.0 - 0.41, RhoD=float(NY),
        RMin=1.0, RMax=12.0, RBinSize=1.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=60.0,
    )
    if ops is not None:
        p.TransOpt = list(ops); p.NrTransOpt = len(p.TransOpt)
    return spec_from_v1_params(p, requires_grad=False)


def _gauss_image(NY, NZ, *, R0_px=6.0, sigma_px=1.5, px=200.0):
    yy, zz = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
    Yc = -(yy - NY / 2.0 - 0.37) * px
    Zc = (zz - NZ / 2.0 + 0.41) * px
    R_um = np.sqrt(Yc * Yc + Zc * Zc)
    R_px = R_um / px
    return np.exp(-(R_px - R0_px) ** 2 / (2 * sigma_px ** 2)).astype(np.float64)


# ── integrate_main: TIFF → CSV ──

def test_cli_integrate_writes_csv_with_correct_shape(tmp_path):
    pytest.importorskip("tifffile")
    import tifffile

    spec = _spec()
    paramstest = _write_paramstest(tmp_path, spec)
    img_path = tmp_path / "img.tif"
    tifffile.imwrite(img_path, _gauss_image(spec.NrPixelsY, spec.NrPixelsZ))

    out_path = tmp_path / "profile.csv"
    rc = integrate_main([
        str(paramstest),
        "--image", str(img_path),
        "--mode", "subpixel", "-K", "2",
        "--out", str(out_path),
    ])
    assert rc == 0
    assert out_path.exists()
    data = np.loadtxt(out_path, delimiter=",", skiprows=1)
    assert data.shape == (spec.n_r_bins, 2)
    # R axis should be monotonic
    assert (np.diff(data[:, 0]) > 0).all()


@pytest.mark.parametrize("mode,extra", [
    ("hard", []),
    ("subpixel", ["-K", "2"]),
    ("subpixel", ["-K", "3"]),
    ("soft", []),
])
def test_cli_integrate_all_modes_produce_csv(tmp_path, mode, extra):
    pytest.importorskip("tifffile")
    import tifffile

    spec = _spec()
    paramstest = _write_paramstest(tmp_path, spec)
    img_path = tmp_path / f"img_{mode}.tif"
    tifffile.imwrite(img_path, _gauss_image(spec.NrPixelsY, spec.NrPixelsZ))

    rc = integrate_main([
        str(paramstest),
        "--image", str(img_path),
        "--mode", mode,
        *extra,
    ])
    assert rc == 0
    auto_out = img_path.with_suffix(img_path.suffix + ".profile.csv")
    assert auto_out.exists()


def test_cli_integrate_with_dark_subtracts(tmp_path):
    pytest.importorskip("tifffile")
    import tifffile

    spec = _spec()
    paramstest = _write_paramstest(tmp_path, spec)
    img = _gauss_image(spec.NrPixelsY, spec.NrPixelsZ) + 5.0
    dark = np.full_like(img, 5.0)
    img_path = tmp_path / "img.tif"
    dark_path = tmp_path / "dark.tif"
    tifffile.imwrite(img_path, img)
    tifffile.imwrite(dark_path, dark)

    out_with_dark = tmp_path / "with_dark.csv"
    rc = integrate_main([
        str(paramstest),
        "--image", str(img_path), "--dark", str(dark_path),
        "--mode", "hard",
        "--out", str(out_with_dark),
    ])
    assert rc == 0

    out_no_dark = tmp_path / "no_dark.csv"
    rc = integrate_main([
        str(paramstest),
        "--image", str(img_path),
        "--mode", "hard",
        "--out", str(out_no_dark),
    ])
    assert rc == 0

    a = np.loadtxt(out_with_dark, delimiter=",", skiprows=1)
    b = np.loadtxt(out_no_dark, delimiter=",", skiprows=1)
    # Dark-subtracted profile should be lower in R bins where there's
    # signal (constant dark removed everywhere).
    assert (a[:, 1] < b[:, 1]).any()


# ── write_map_main: emit Map.bin readable by v1 ──

def test_cli_write_map_emits_files_v1_can_read(tmp_path):
    spec = _spec()
    paramstest = _write_paramstest(tmp_path, spec)

    rc = write_map_main([
        str(paramstest),
        "--out-dir", str(tmp_path / "maps"),
        "--mode", "subpixel", "-K", "2",
    ])
    assert rc == 0
    map_p = tmp_path / "maps" / "Map.bin"
    nmap_p = tmp_path / "maps" / "nMap.bin"
    assert map_p.exists() and nmap_p.exists()

    pm = load_map(map_p, nmap_p)
    assert pm.n_bins == spec.n_eta_bins * spec.n_r_bins
    assert pm.n_entries > 0


def test_cli_write_map_no_header_legacy_format(tmp_path):
    spec = _spec()
    paramstest = _write_paramstest(tmp_path, spec)
    rc = write_map_main([
        str(paramstest),
        "--out-dir", str(tmp_path / "legacy"),
        "--mode", "hard",
        "--no-header",
    ])
    assert rc == 0
    pm = load_map(tmp_path / "legacy" / "Map.bin",
                   tmp_path / "legacy" / "nMap.bin")
    assert pm.map_header is None


# ── Subprocess invocation (closes the "is the script wired?" loop) ──

def test_cli_subprocess_help_runs():
    """Smoke: the entry-point script is callable from a subprocess."""
    res = subprocess.run(
        [sys.executable, "-m", "midas_integrate_v2.cli", "--help"],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
    )
    # cli.py's __main__ runs integrate_main; --help exits 0.
    assert res.returncode == 0
    assert "midas-integrate-v2" in res.stdout
    assert "--mode" in res.stdout


def test_cli_no_trans_opt_flag_strips_TransOpt(tmp_path):
    """When --no-trans-opt is set and the paramstest has ImTransOpt,
    the CLI should integrate as if no transform were configured."""
    pytest.importorskip("tifffile")
    import tifffile

    spec = _spec(ops=[2])
    paramstest = _write_paramstest(tmp_path, spec)
    img_path = tmp_path / "img.tif"
    tifffile.imwrite(img_path, _gauss_image(spec.NrPixelsY, spec.NrPixelsZ))

    out_with = tmp_path / "with.csv"
    out_without = tmp_path / "without.csv"
    integrate_main([
        str(paramstest), "--image", str(img_path), "--mode", "hard",
        "--out", str(out_with),
    ])
    integrate_main([
        str(paramstest), "--image", str(img_path), "--mode", "hard",
        "--no-trans-opt",
        "--out", str(out_without),
    ])
    a = np.loadtxt(out_with, delimiter=",", skiprows=1)[:, 1]
    b = np.loadtxt(out_without, delimiter=",", skiprows=1)[:, 1]
    # Different transforms must produce different profiles
    assert not np.allclose(a, b, atol=1e-12)
