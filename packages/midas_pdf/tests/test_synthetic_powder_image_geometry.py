"""Fast (not slow-marked, so it runs in CI) pin of ``validate.synthetic_powder_image`` on a non-square, non-symmetric grid.

It used to build the transposed image with a single pitch; that went unnoticed while NY == NZ, BC_y == BC_z and pxY == pxZ.
"""
from __future__ import annotations

import numpy as np
import torch
from midas_integrate_v2 import IntegrationSpec

from midas_pdf.validate import synthetic_powder_image


def test_synthetic_powder_image_is_row_z_col_y_with_physical_radius():
    s = IntegrationSpec()
    s.NrPixelsY, s.NrPixelsZ = 60, 40
    s.pxY, s.pxZ = 200.0, 100.0
    s.Lsd = torch.tensor(50_000.0, dtype=torch.float64)
    s.BC_y, s.BC_z = torch.tensor(22.5, dtype=torch.float64), torch.tensor(11.0, dtype=torch.float64)
    s.Wavelength = torch.tensor(0.17, dtype=torch.float64)
    q = torch.linspace(0.01, 30.0, 4000, dtype=torch.float64)
    img = synthetic_powder_image(s, q, q.clone(), counts=1e13, seed=0)         # I(Q) = Q, negligible Poisson noise
    assert tuple(img.shape) == (40, 60)                                         # (NZ rows, NY cols)
    zz, yy = np.meshgrid(np.arange(40), np.arange(60), indexing="ij")
    r = np.stack([(yy - 22.5) * 200.0, (zz - 11.0) * 100.0, np.full(yy.shape, 50_000.0)])
    two_theta = np.arccos(r[2] / np.sqrt((r * r).sum(axis=0)))                  # lab-frame angle from the beam, independent of the module
    q_true = 4.0 * np.pi / 0.17 * np.sin(0.5 * two_theta)
    np.testing.assert_allclose(img.numpy(), q_true, rtol=2e-4, atol=1e-4)


# ---- source scans: no radius conversion may use a single axis pitch ----------------------------
def _pdf_source_files():
    from pathlib import Path
    return sorted((Path(__file__).resolve().parents[1] / "midas_pdf").rglob("*.py"))


def test_no_pdf_radius_conversion_passes_pxY_as_the_pitch():
    import re
    bad = re.compile(r"px_um\s*=\s*\w+(\[\w+\])?\.pxY|\*\s*\w+(\[\w+\])?\.pxY\s*/|\.pxY\s*\*\s*R")
    hits = [f"{f.name}:{i}: {line.strip()}" for f in _pdf_source_files()
            for i, line in enumerate(f.read_text().splitlines(), 1)
            if bad.search(line) and not line.lstrip().startswith("#")]
    assert not hits, "radius converted with pxY instead of spec.mean_pixel_um():\n" + "\n".join(hits)


def test_pdf_source_scan_second_family_of_slip_patterns():
    import re
    bad = re.compile(
        r"\bpx(_um)?\s*=\s*(float\()?(torch\.as_tensor\()?(self\.)?spec\.pxY\b"
        r"|px_um\s*=\s*(float\()?(self\.)?spec\.pxY\b"
        r"|px_um\s*=\s*\w*\.?effective_pxYZ\(\)\[0\]"
        r"|\*\s*(float\()?(self\.)?spec\.pxY\)?\s*/\s*(L|Lsd|float)"
        r"|/\s*(float\()?(self\.)?spec\.pxY\b"
    )
    hits = [f"{f.name}:{i}: {line.strip()}" for f in _pdf_source_files()
            for i, line in enumerate(f.read_text().splitlines(), 1)
            if bad.search(line) and not line.lstrip().startswith("#")]
    assert not hits, "possible mean-pixel radius converted with a single axis pitch:\n" + "\n".join(hits)
