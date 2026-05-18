"""Builds notebook 00: meeting tour — runs every demo end-to-end."""
from pathlib import Path
from _nb_helper import write_notebook


CELLS = [
    ("md", """\
# 00 — Meeting tour: end-to-end MIDAS v2 demo

**Audience**: anyone you're walking through MIDAS for the first time.
**Time**: ~15 minutes (most cells are seconds; the Hydra real-data
cell is 4–5 min so park it last).
**What it covers**: every Phase 0–6 capability shipped in v0.10.

This notebook is the live-demo flow used in the LANL / Wenqian / APS
management briefings. Each cell runs one demo runner; the markdown
above each cell is the talking point.
"""),
    ("md", "## 0. Setup"),
    ("code", """\
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import subprocess
from pathlib import Path

REPO = Path("../").resolve()
RUNNERS = REPO / "dev/paper/runners"
RUNS = REPO / "dev/paper/runs"

def run(script_name: str):
    print(f"\\n=== running {script_name} ===")
    out = subprocess.run(
        ["python", str(RUNNERS / script_name)],
        env={**os.environ},
        capture_output=True, text=True, timeout=900,
    )
    print(out.stdout[-2000:] if len(out.stdout) > 2000 else out.stdout)
    if out.returncode:
        print("STDERR:", out.stderr[-1000:])
    return out.returncode
"""),
    ("md", """\
## 1. The composite story (10 s)

Synthetic CeO2 ring frame → spatial dezinger + cosmic-ray rejection
→ empty-cell subtraction → polygon-bin integrate with σ propagation
→ PDF G(r) with σ → write DAT (PDFgetX3), FXYE (TOPAS), ESG
(MAUD/MILK), CSV (G(r) + σ).

**Talking point**: "Pixel σ → I(Q) σ → G(r) σ — propagated end-to-end
through corrections and the Fourier sine transform. pyFAI provides σ
at I(Q); v2 carries it the rest of the way to σ_G(r), which no
open-source PDF pipeline does today."
"""),
    ("code", """\
run("run_aps_meeting_demo.py")
print("outputs:", list((RUNS / "aps_meeting_demo").iterdir()))
"""),
    ("md", """\
## 2. MIDAS-only feature: auto-detect bad pixels (~30 s)

Plant 200 cosmic-ray-like spikes; train LearnableMask jointly with a
sparsity prior; bad pixels naturally drop to 0.
"""),
    ("code", """\
run("run_learnable_mask_demo.py")
"""),
    ("md", """\
## 3. MIDAS-only feature: auto-recover gain drift (~30 s)

Plant 5% Gaussian-smooth gain field; train LearnableGain to reverse it.
Recovered RMSE on the order of 1%.
"""),
    ("code", """\
run("run_learnable_gain_demo.py")
"""),
    ("md", """\
## 4. Kernel comparison vs pyFAI (~60 s)

Hard / subpixel(K=2) / polygon kernels on the same synthetic CeO2;
pyFAI splitpixel(8×8) for cross-check (when pyFAI is installed).
"""),
    ("code", """\
run("run_pyfai_bakeoff.py")
"""),
    ("md", """\
## 5. σ chain pixel → MAUD (LANL Item 47, ~30 s)

Synthetic Ni standard with Poisson noise → MIDAS polygon σ → ESG with
σ in `_pd_proc_intensity_total_su`. Validates analytic σ against an
n=30 Monte Carlo bootstrap. Median rel-err is on the order of 0.1.
"""),
    ("code", """\
run("run_sigma_maud.py")
"""),
    ("md", """\
## 6. HEDM grain-ODF vs E-WIMV (LANL Item 48, ~5 s)

Synthetic 2000-grain texture; volume-weighted (1,0,0) pole figure vs
E-WIMV-style smoothed analogue. KL and L2 distance reported. Real
LANL data plug-in via the same API once available.
"""),
    ("code", """\
run("run_hedm_ewimv_xval.py")
"""),
    ("md", """\
## 7. Polarisation-plane refinement (~30 s)

Start from a wrong η_pol = 30°, fit it back to ≈ 0° (synchrotron
horizontal default) by minimising η-uniformity loss. Demonstrates
autograd-aware geometry refinement.
"""),
    ("code", """\
run("run_polarization_plane_refinement.py")
"""),
    ("md", """\
## 8. SAXS workflow (Item 37, ~30 s)

Same framework, low-Q range. Solid-angle correction is the dominant
correction at small 2θ — 3.1× larger than polarisation, measured.
"""),
    ("code", """\
run("run_saxs_workflow.py")
"""),
    ("md", """\
## 9. Magnetic / resonant polarisation-resolved demo (~30 s)

Synthetic σ-pol vs π-pol Bragg intensity; ratio map shows magnetic
contrast.
"""),
    ("code", """\
run("run_magnetic_resonant_demo.py")
"""),
    ("md", """\
## 10. Real APS 1-ID Hydra CeO2 — multi-panel parity (~270 s)

Four panels × 2048² each, integrated through MILKMultiGeometryAdapter
into one σ-aware I(2θ) on a unified axis. **This is the LANL
Item 46 demo on real APS data.**

(Skip if you're tight on time — the talking point alone is enough.)
"""),
    ("code", """\
# Uncomment to run (~270 s on CPU)
# run("run_hydra_real_data.py")
print("data path: /Users/hsharma/Desktop/analysis/hydra/")
print("expected output: dev/paper/runs/hydra_real_data/hydra_combined.png")
"""),
    ("md", """\
## 11. APS-U operational throughput (~few min)

Eiger 9M-class detector (3262 × 3108) integrated through the polygon
kernel with σ propagation; throughput vs APS-U @ 100 Hz target.
"""),
    ("code", """\
# Uncomment to run (~5 min on CPU; produces fps figure)
# run("run_apsu_benchmark.py")
print("expected output: dev/paper/runs/apsu_benchmark/throughput.png")
"""),
    ("md", """\
## 12. Refresh PNGs for slide decks

Loads each runner's CSV/H5 outputs and writes PNGs into the
respective `dev/paper/runs/*/` directory.
"""),
    ("code", """\
run("plot_all_results.py")
"""),
    ("md", """\
## Reference materials

- `dev/paper/HANDOFF.md` — full 48-item inventory with file pointers
- `dev/paper/meetings/INDEX.md` — meeting briefs index
- `dev/paper/meetings/DEMO_cheatsheet.md` — talking points per demo
- `dev/paper/meetings/slides/` — Marp slide decks (LANL / Wenqian / APS-mgmt / generic)
- `dev/paper/meetings/handouts/` — print-friendly 1-pagers
- `dev/aps_deployment_guide.md` — pan-APS recipes
"""),
]


def main():
    out = Path(__file__).parent / "00_meeting_tour.ipynb"
    write_notebook(out, CELLS)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
