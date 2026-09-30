"""sum_frames skips a truncated frame instead of failing."""
import numpy as np
import pytest
import tifffile

from midas_snapshot.io import list_frames, sum_frames


def test_truncated_frame_is_skipped(tmp_path):
    for i in range(3):
        tifffile.imwrite(tmp_path / f"f_{i:03d}.tif", np.full((8, 9), i + 1, np.int32))
    (tmp_path / "f_003.tif").write_bytes(b"")                       # truncated write
    with pytest.warns(UserWarning):
        s = sum_frames(list_frames(str(tmp_path)), None)
    assert s.shape == (8, 9) and np.all(s == 6)



def test_single_frame_static_series_runs_to_report(tmp_path):
    """A one-frame still: run, analyse --static and report must all work (one window)."""
    import json, os
    from midas_snapshot.cli import main
    from midas_snapshot.config import SnapshotConfig
    n = 64
    (tmp_path / "g.txt").write_text(f"Lsd 250000\nBC {n/2} {n/2}\npx 172\nWavelength 0.124\nNrPixelsY {n}\nNrPixelsZ {n}\n")
    fr = tmp_path / "fr"; fr.mkdir()
    tifffile.imwrite(fr / "s_000.tif", np.random.default_rng(0).poisson(0.2, (n, n)).astype(np.int32))
    cfg = SnapshotConfig(frames=str(fr), geometry=str(tmp_path / "g.txt"), out=str(tmp_path / "o"), flip=None,
                         window_sizes=[1], margin_px=4, local_box=9, nproc=1)
    os.makedirs(cfg.out); cp = os.path.join(cfg.out, "snapshot_config.json"); cfg.save(cp)
    main(["run", cp]); main(["analyse", cp, "--static"]); main(["report", cp])
    rep = json.load(open(os.path.join(cfg.out, "report.json")))
    assert rep["windows"]["W1"]["n_windows"] == 1


def test_injection_at_read(tmp_path, monkeypatch):
    """SNAPSHOT_INJECT adds Poisson spots at read, only inside [first, last], never on invalid
    pixels, reproducibly."""
    import json
    import tifffile
    from midas_snapshot import io as sio
    fr = tmp_path / "fr"; fr.mkdir()
    for i in range(6):
        a = np.zeros((64, 64), np.int32)
        a[:, 40] = -1                                           # gap column
        tifffile.imwrite(fr / f"x_{i:03d}.tif", a)
    spec = dict(frames=str(fr), seed=3, spots=[dict(row=20.3, col=20.6, flux=500.0, sigma=1.2, first=1, last=3),
                                                dict(row=30.0, col=40.0, flux=500.0, sigma=1.0, first=0, last=5)])
    sp = tmp_path / "inj.json"; sp.write_text(json.dumps(spec))
    monkeypatch.setenv("SNAPSHOT_INJECT", str(sp))
    files = sio.list_frames(str(fr))
    tot = [sio.read_frame(f, None)[10:31, 10:31].clip(0).sum() for f in files]
    assert tot[0] == 0 and tot[4] == 0 and tot[5] == 0
    assert all(abs(t - 500) < 100 for t in tot[1:4])
    g = sio.read_frame(files[2], None)
    assert np.all(g[:, 40] == -1)                                # gap stays invalid
    assert np.array_equal(g, sio.read_frame(files[2], None))    # reproducible
    monkeypatch.delenv("SNAPSHOT_INJECT")
    assert sio.read_frame(files[2], None)[10:31, 10:31].sum() == 0
