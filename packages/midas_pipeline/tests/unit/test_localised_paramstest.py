"""paramstest folder keys must name the layer being analysed.

They are inherited from the parameter file embedded in the zarr, i.e. from the
machine that BUILT the archive. Analyse that archive anywhere else and the keys
point at a directory which does not exist. The c-omp backends never notice
because comp_backend_paramstest rewrites them; the python backends are handed
the file as-is and die with FileNotFoundError on Spots.bin.

Observed on the datasetA Ni layer: OutputFolder read
/Users/hsharma/Desktop/analysis/... on a Linux cluster, so
--indexer-backend python could not run at all while c-omp completed.
"""

from pathlib import Path

from midas_pipeline.stages._comp_params import localised_paramstest


def _write(p: Path, out: str, res: str, extra="Lsd 958874.75\nSpaceGroup 225\n"):
    p.write_text(f"OutputFolder {out}\nResultFolder {res}\n{extra}")


def _keys(p: Path):
    d = {}
    for ln in p.read_text().splitlines():
        if ln.startswith(("OutputFolder", "ResultFolder")):
            k, v = ln.split(None, 1)
            d[k] = v.strip()
    return d


def test_rewrites_a_foreign_path(tmp_path):
    layer = tmp_path / "LayerNr_1"; layer.mkdir()
    ps = layer / "paramstest.txt"
    _write(ps, "/Users/someone/Desktop/analysis/x/LayerNr_1/",
               "/Users/someone/Desktop/analysis/x/LayerNr_1/")
    out = localised_paramstest(ps, layer)
    assert out != ps, "a foreign path must produce a corrected file"
    assert out.name == "paramstest_local.txt"
    assert _keys(out) == {"OutputFolder": str(layer), "ResultFolder": str(layer)}
    assert "SpaceGroup 225" in out.read_text(), "other keys must survive"


def test_leaves_a_correct_file_untouched(tmp_path):
    """The common case must not litter the run dir with duplicates."""
    layer = tmp_path / "LayerNr_1"; layer.mkdir()
    ps = layer / "paramstest.txt"
    _write(ps, str(layer), str(layer))
    assert localised_paramstest(ps, layer) == ps
    assert not (layer / "paramstest_local.txt").exists()


def test_absent_keys_are_left_alone(tmp_path):
    """A MISSING folder key is not the bug this repairs.

    The stage runs the backend with ``cwd=layer_dir``, so an absent
    OutputFolder already resolves to the right place. Materialising one would
    change the command line for every well-formed run — which is how this was
    first written, and what ``test_stage_ff_dispatch`` caught.
    """
    layer = tmp_path / "LayerNr_1"; layer.mkdir()
    ps = layer / "paramstest.txt"
    ps.write_text("Lsd 958874.75\nSpaceGroup 225\n")
    assert localised_paramstest(ps, layer) == ps
    assert not (layer / "paramstest_local.txt").exists()


def test_trailing_slash_is_not_a_difference(tmp_path):
    """`/a/b/` and `/a/b` are the same directory; do not rewrite for that."""
    layer = tmp_path / "LayerNr_1"; layer.mkdir()
    ps = layer / "paramstest.txt"
    _write(ps, str(layer) + "/", str(layer) + "/")
    assert localised_paramstest(ps, layer) == ps


def test_one_wrong_key_is_enough_to_trigger(tmp_path):
    layer = tmp_path / "LayerNr_1"; layer.mkdir()
    ps = layer / "paramstest.txt"
    _write(ps, str(layer), "/elsewhere/LayerNr_1")
    out = localised_paramstest(ps, layer)
    assert out != ps
    assert _keys(out)["ResultFolder"] == str(layer)
