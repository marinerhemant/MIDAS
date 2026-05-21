"""Backend-aware paramstest for the unified C (``c-omp``) indexer.

The unified C ``midas_indexer`` locates its binned inputs (``Spots.bin``,
``Data.bin``, ``nData.bin``, …) via ``dirname(OutputFolder)`` and writes its
``IndexBest*_all.bin`` family into ``OutputFolder``. The pipeline writes a bare
``OutputFolder <layer_dir>`` (which the in-process python backend reads fine),
so the C reader would look one level *too high* for the inputs.

For the c-omp backend we therefore hand the binary — and the downstream
``midas-fit-grain`` / ``midas-process-grains`` steps, which then read the C
outputs from the same folders — a paramstest whose ``OutputFolder`` is
``<layer_dir>/Output`` and ``ResultFolder`` is ``<layer_dir>/Results``.
"""
from __future__ import annotations

from pathlib import Path


def comp_backend_paramstest(paramstest: Path, layer_dir: Path) -> Path:
    """Write ``paramstest_comp.txt`` next to *paramstest* with OutputFolder/
    ResultFolder pointed at ``<layer_dir>/Output`` and ``<layer_dir>/Results``.

    Returns the path to the new file. The binned inputs stay in *layer_dir*
    (= ``dirname(OutputFolder)``), so the C binary finds them and emits into
    ``Output/``; refinement + process-grains read from the same folders.
    """
    out_dir = layer_dir / "Output"
    res_dir = layer_dir / "Results"
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    lines, seen_out, seen_res = [], False, False
    for ln in Path(paramstest).read_text().splitlines():
        key = ln.strip().split(" ")[0] if ln.strip() else ""
        if key == "OutputFolder":
            lines.append(f"OutputFolder {out_dir}"); seen_out = True
        elif key == "ResultFolder":
            lines.append(f"ResultFolder {res_dir}"); seen_res = True
        else:
            lines.append(ln)
    if not seen_out:
        lines.append(f"OutputFolder {out_dir}")
    if not seen_res:
        lines.append(f"ResultFolder {res_dir}")

    dst = Path(layer_dir) / "paramstest_comp.txt"
    dst.write_text("\n".join(lines) + "\n")
    return dst
