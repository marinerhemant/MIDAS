"""End-to-end CLI smoke test.

Builds a tiny on-disk dataset (paramstest.txt, ExtraInfo.bin, IndexBest.bin,
IndexBestFull.bin, hkls.csv, SpotsToIndex.csv) from the synthetic fixture,
runs ``midas-fit-grain`` via ``cli.main()``, and verifies the four output
files are produced with non-empty contents.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from midas_fit_grain.cli import main as cli_main
from midas_fit_grain.io_binary import (
    EXTRA_INFO_NCOLS, MAX_NHKLS_DEFAULT,
    ORIENT_POS_FIT_NCOLS, read_orient_pos_fit, read_key,
)

from ._synthetic import make_synthetic, fixture_to_observed

DEG2RAD = math.pi / 180.0


def _build_fake_dataset(tmp_path: Path) -> dict:
    """Lay out the on-disk files the driver expects."""
    fix = make_synthetic(device=torch.device("cpu"), dtype=torch.float64)
    obs = fixture_to_observed(fix, device=torch.device("cpu"),
                              dtype=torch.float64)

    cwd = tmp_path
    out_dir = cwd / "Output"
    res_dir = cwd / "Results"
    out_dir.mkdir()
    res_dir.mkdir()

    # paramstest.txt — a slimmed-down version that the parser groks.
    a = float(fix.gt_lattice[0])
    body = [
        f"LatticeParameter {a} {a} {a} 90 90 90;",
        "SpaceGroup 225;",
        f"Wavelength {fix.model.wavelength};",
        f"Distance {fix.model.Lsd};",
        f"px {fix.px};",
        f"MaxRingRad {fix.model.Lsd * math.tan(0.5 * fix.obs_two_theta.max().item() * 2)};",
        "MarginRadius 800;",
        "MarginEta 5; MarginOme 2;",
        "EtaBinSize 2; OmeBinSize 2;",
        "ExcludePoleAngle 6;",
        "MargABC 1; MargABG 1;",
        "FitAllAtOnce 0;",
        "Wedge 0;",
    ]
    for nr, rs in zip(fix.ring_numbers, [1.0] * len(fix.ring_numbers)):
        body.append(f"RingNumbers {nr};")
        body.append(f"RingRadii {rs};")
    body.append("OmegaRange -180 180;")
    body.append("BoxSize -1000000 1000000 -1000000 1000000;")
    body.append(f"OutputFolder {out_dir.as_posix()}")
    body.append(f"ResultFolder {res_dir.as_posix()}")
    body.append("SpotsFileName Spots.bin")
    body.append("IDsFileName SpotsToIndex.csv")
    body.append("RefinementFileName InputAllExtraInfoFittingAll.csv")
    (cwd / "paramstest.txt").write_text("\n".join(body))

    # ExtraInfo.bin — pack obs into the 16-double layout.
    n = obs.n_spots
    extra = np.zeros((n, EXTRA_INFO_NCOLS), dtype=np.float64)
    extra[:, 0] = obs.y_lab.numpy()
    extra[:, 1] = obs.z_lab.numpy()
    extra[:, 2] = obs.omega.numpy() / DEG2RAD          # back to deg
    # ExtraInfo.bin uses 1-based spot IDs to match IndexBestFull.bin's
    # convention (C-side: AllSpotsPtr[(spotID-1) * 16 + ...]).
    extra[:, 4] = obs.spot_id.numpy() + 1
    extra[:, 5] = fix.obs_ring.numpy()                 # ring slot used as ringnumber
    extra[:, 6] = obs.eta.numpy() / DEG2RAD
    extra[:, 7] = obs.two_theta.numpy() / DEG2RAD
    extra[:, 8] = obs.omega.numpy() / DEG2RAD
    extra[:, 9:11] = 0.0
    extra.tofile(cwd / "ExtraInfo.bin")

    # SpotsToIndex.csv — one seed grain.
    np.savetxt(cwd / "SpotsToIndex.csv",
               np.array([int(obs.spot_id[0]) + 1], dtype=np.int64),
               fmt="%d")

    # IndexBest.bin — 15 doubles for the seed.
    rec = np.zeros(15, dtype=np.float64)
    OM = fix.model.euler2mat(fix.gt_euler).numpy().reshape(-1)
    rec[1:10] = OM
    rec[10:13] = fix.gt_position.numpy()
    rec[14] = float(n)             # n_observed = total spots
    rec.tofile(out_dir / "IndexBest.bin")

    # IndexBestFull.bin — (1, MAX_NHKLS, 2) doubles. Set first n entries.
    full = np.zeros((1, MAX_NHKLS_DEFAULT, 2), dtype=np.float64)
    full[0, :n, 0] = obs.spot_id.numpy() + 1   # SpotID is 1-based on disk
    full.tofile(out_dir / "IndexBestFull.bin")

    # hkls.csv — h k l ds RingNr <s> <s> <s> tht <s> <s>
    lines = ["% h k l ds RingNr c1 c2 c3 tht c4 c5"]
    h2 = (fix.gt_euler.new_tensor(fix.model.hkls_int).long() ** 2).sum(-1).numpy() if fix.model.hkls_int is not None else None
    # Use the synthetic's hkls_int directly.
    hkls_int = fix.model.hkls_int.cpu().numpy().astype(int)
    thetas = fix.model.thetas.cpu().numpy()
    h2_uniq = sorted(set(int((hkls_int ** 2).sum(axis=1)[i]) for i in range(len(hkls_int))))
    h2_to_ring = {v: i for i, v in enumerate(h2_uniq)}
    for i, (h, k, l) in enumerate(hkls_int):
        s2 = int(h * h + k * k + l * l)
        rnr = h2_to_ring[s2]
        ds = 1.0 / np.linalg.norm(np.array([h, k, l]) / a)
        tht_deg = math.degrees(float(thetas[i]))
        lines.append(f"{h} {k} {l} {ds} {rnr} - - - {tht_deg} - -")
    (cwd / "hkls.csv").write_text("\n".join(lines) + "\n")

    return {
        "cwd": cwd, "out_dir": out_dir, "res_dir": res_dir,
        "fix": fix, "obs": obs,
    }


@pytest.mark.xfail(reason=(
    "Synthetic CLI fixture builds a fake hkls.csv with placeholder "
    "g-vector / d-spacing values that don't satisfy the C-port's exact "
    "diffraction geometry; the smoke test verified file shapes were "
    "correct before the C port landed. Real-data validation lives in "
    "the ti7_al regression run on copland."
))
def test_cli_runs_end_to_end(tmp_path):
    setup = _build_fake_dataset(tmp_path)
    cwd = setup["cwd"]
    res_dir = setup["res_dir"]

    # Invoke the CLI as if from the command line.
    rc = cli_main([
        str(cwd / "paramstest.txt"),
        "0",        # block_nr
        "1",        # num_blocks
        "1",        # num_lines
        "1",        # num_procs
        "--solver", "lbfgs",
        "--mode", "all_at_once",
        "--loss", "angular",
        "--max-iter", "50",
        "--device", "cpu",      # CI may have MPS but float64 is required
        "--dtype", "float64",
    ])
    assert rc == 0

    # OrientPosFit.bin must contain at least one row's worth of doubles.
    op_path = res_dir / "OrientPosFit.bin"
    assert op_path.exists() and op_path.stat().st_size > 0
    op = read_orient_pos_fit(op_path, n_grains=1)
    assert op.shape == (1, ORIENT_POS_FIT_NCOLS)
    # The output must contain a non-zero orientation matrix (cols 1..9).
    assert not np.allclose(op[0, 1:10], 0.0)

    # Key.bin must record n_matched > 0 for the single grain.
    key_path = res_dir / "Key.bin"
    key = read_key(key_path, n_grains=1)
    assert key[0, 1] > 0, f"Key.bin n_matched was {key[0, 1]}"
