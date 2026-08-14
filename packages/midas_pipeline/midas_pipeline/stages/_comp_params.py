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


def comp_backend_paramstest(
    paramstest: Path, layer_dir: Path, result_folder: Path | None = None,
) -> Path:
    """Write ``paramstest_comp.txt`` next to *paramstest* with OutputFolder/
    ResultFolder pointed at ``<layer_dir>/Output`` and ``<layer_dir>/Results``.

    Returns the path to the new file. The binned inputs stay in *layer_dir*
    (= ``dirname(OutputFolder)``), so the C binary finds them and emits into
    ``Output/``; refinement + process-grains read from the same folders.

    ``result_folder`` overrides where the C binary writes its per-seed
    ``FitBest_*.csv`` — the PF c-omp refine path points it at a dedicated
    dir so those files don't collide (double-count) with the adapted
    ``Result_OrientPos_voxel_*.csv`` that consolidation also globs from
    ``Results/``.
    """
    out_dir = layer_dir / "Output"
    res_dir = Path(result_folder) if result_folder is not None else layer_dir / "Results"
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


# Grain-selection / sample-geometry keys that process-grains consumes and that
# ``paramstest.txt`` does not carry. FitSetup writes paramstest for the indexer
# and refiner, which have no use for them, so they are simply absent -- and
# every consumer then falls back to its own default. Measured on the datasetA
# Ni layer: the user's file says ``MinNrSpots 3`` / ``Completeness 0.5``, the
# classical chain (handed the zarr archive, which carries them) returns 6132
# grains, and process-grains driven off paramstest.txt returns 23710 from the
# SAME refiner output.
_PG_SELECTION_KEYS = (
    "MinNrSpots",
    "Completeness",
    "Vsample",
    "BeamThickness",
    "GlobalPosition",
    "Width",
    "MinEta",
    "Twin",
    "PhaseNr",
    # Read by midas_process_grains, midas_fit_grain and midas_index, and not
    # written into paramstest by FitSetup. A multi-phase run that reached
    # process-grains through the pipeline silently saw NumPhases = 1.
    "NumPhases",
)


def selection_paramstest(paramstest: Path, params_file: Path | str | None,
                         layer_dir: Path) -> Path:
    """Return a paramstest carrying the user's grain-selection thresholds.

    Copies any key in :data:`_PG_SELECTION_KEYS` that the user's own parameter
    file sets and *paramstest* lacks. Returns *paramstest* unchanged when there
    is nothing to add, so a well-formed run gains no file and no new path.

    The user's value always wins over a downstream default; a value already
    present in *paramstest* is left alone.
    """
    from .._logging import LOG

    if not params_file:
        return Path(paramstest)
    src = Path(params_file)
    if not src.exists():
        return Path(paramstest)

    def _keys(text: str) -> set:
        out = set()
        for ln in text.splitlines():
            ln = ln.split("#", 1)[0].strip()
            if ln:
                out.add(ln.split()[0].rstrip(";"))
        return out

    ps_text = Path(paramstest).read_text()
    have = _keys(ps_text)

    add = []
    for ln in src.read_text().splitlines():
        stripped = ln.split("#", 1)[0].strip()
        if not stripped:
            continue
        key = stripped.split()[0].rstrip(";")
        if key in _PG_SELECTION_KEYS and key not in have:
            add.append(stripped)
            have.add(key)          # first occurrence wins, as MIDAS parsers do

    if not add:
        return Path(paramstest)

    LOG.info("process_grains(FF): propagating %s from %s (absent from %s)",
             ", ".join(a.split()[0] for a in add), src.name,
             Path(paramstest).name)
    dst = Path(layer_dir) / "paramstest_pg.txt"
    dst.write_text(ps_text.rstrip("\n") + "\n" + "\n".join(add) + "\n")
    return dst


def localised_paramstest(paramstest: Path, layer_dir: Path) -> Path:
    """Return a paramstest whose folder keys name *layer_dir*.

    ``paramstest.txt`` inherits ``OutputFolder`` / ``ResultFolder`` from the
    source parameter file, which is embedded in the zarr archive at the machine
    where the archive was BUILT. Analysing that archive anywhere else leaves
    those keys pointing at a directory that does not exist:

        OutputFolder /Users/hsharma/Desktop/analysis/datasetA/.../LayerNr_1/

    read on a Linux cluster. The c-omp backends never notice, because
    :func:`comp_backend_paramstest` rewrites both keys before invoking the
    binary. The python backends are handed the file as-is and fail --
    ``midas_index`` raises ``FileNotFoundError: Spots.bin not found at
    <the other machine's path>``. Measured on the datasetA Ni layer, where
    ``--indexer-backend python`` could not run at all while c-omp completed.

    Returns *paramstest* unchanged when its keys are already correct, so the
    common case adds no file. Uses the bare ``layer_dir`` for both keys, which
    is the convention the python backends read.
    """
    layer_dir = Path(layer_dir)
    text = Path(paramstest).read_text()
    lines, seen_out, seen_res, needs_fix = [], False, False, False
    for ln in text.splitlines():
        key = ln.strip().split(" ")[0] if ln.strip() else ""
        if key == "OutputFolder":
            seen_out = True
            if Path(ln.split(None, 1)[1].strip()).resolve() != layer_dir.resolve():
                needs_fix = True
            lines.append(f"OutputFolder {layer_dir}")
        elif key == "ResultFolder":
            seen_res = True
            if Path(ln.split(None, 1)[1].strip()).resolve() != layer_dir.resolve():
                needs_fix = True
            lines.append(f"ResultFolder {layer_dir}")
        else:
            lines.append(ln)
    # A MISSING key is not a bug: the stage runs the backend with
    # cwd=layer_dir, so an absent OutputFolder already resolves to the right
    # place. Only a key that NAMES SOMEWHERE ELSE is the failure this repairs.
    # Materialising absent keys would change the command line for every
    # well-formed run, which is how this was first written and what
    # test_stage_ff_dispatch caught.
    if not needs_fix:
        return Path(paramstest)
    if not seen_out:
        lines.append(f"OutputFolder {layer_dir}")
    if not seen_res:
        lines.append(f"ResultFolder {layer_dir}")
    dst = layer_dir / "paramstest_local.txt"
    dst.write_text("\n".join(lines) + "\n")
    return dst
