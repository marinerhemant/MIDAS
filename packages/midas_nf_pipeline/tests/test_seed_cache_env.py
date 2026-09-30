"""$MIDAS_NF_SEED_DIR must be honoured by the pipeline's seed stage.

Regression: stages.run_seed_orientations_from_cache always passed DEFAULT_SEED_DIR,
and a non-None seed_dir makes from_cache search ONLY that directory, so the env
var was never read and installs without a source tree regenerated 251 545 seeds
instead of loading the 243 129-seed cache (bt_20id_jul26b, 2026-09-24).
"""
import numpy as np

from midas_nf_pipeline import stages
from midas_nf_preprocess.seed_orientations.from_cache import space_group_to_lookup_type


def _write_cache(d, n=7):
    lookup = space_group_to_lookup_type(225)
    q = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    np.savetxt(d / f"seed_{lookup}.csv", q, delimiter=",")
    return n


def test_env_seed_dir_is_used(tmp_path, monkeypatch):
    cache = tmp_path / "cache"; cache.mkdir()
    n = _write_cache(cache)
    monkeypatch.setenv("MIDAS_NF_SEED_DIR", str(cache))
    out = tmp_path / "seedOrientations.csv"
    stages.run_seed_orientations_from_cache({"SpaceGroup": 225, "SeedOrientations": str(out)})
    rows = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(rows) == n        # loaded from the env cache, not regenerated (251 545)


def test_install_dir_cache_still_wins(tmp_path, monkeypatch):
    inst = tmp_path / "inst" / "NF_HEDM" / "seedOrientations"; inst.mkdir(parents=True)
    n = _write_cache(inst, n=5)
    env = tmp_path / "env"; env.mkdir(); _write_cache(env, n=9)
    monkeypatch.setenv("MIDAS_NF_SEED_DIR", str(env))
    out = tmp_path / "s.csv"
    stages.run_seed_orientations_from_cache({"SpaceGroup": 225, "SeedOrientations": str(out)},
                                            install_dir=str(tmp_path / "inst"))
    assert len([l for l in out.read_text().splitlines() if l.strip()]) == n
