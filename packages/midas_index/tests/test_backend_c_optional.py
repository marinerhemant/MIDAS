"""Optional-binary contract tests for midas_index.backend_c.

These tests don't require the C binary to be installed — they verify
:func:`backend_c.available` reports the correct presence and that
:func:`backend_c.run_indexer` fails loudly with
:class:`CBackendUnavailableError` when the binary is missing.

The actual end-to-end parity tests against the C binary live in
``test_unified_c_parity.py`` and skip when the binary IS missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from midas_index import backend_c
from midas_index.backend_c import CBackendUnavailableError


def test_available_returns_bool():
    """available() always returns a bool, never raises."""
    result = backend_c.available()
    assert isinstance(result, bool)


def test_binary_path_returns_path():
    """binary_path() always returns a Path, even when the file is absent."""
    p = backend_c.binary_path()
    assert isinstance(p, Path)
    # Path components include "midas_index" and "bin" + the binary name.
    assert "midas_index" in p.parts
    assert p.name == "midas_indexer"


def test_run_indexer_raises_when_binary_missing(tmp_path, monkeypatch):
    """When the binary isn't on disk, run_indexer raises CBackendUnavailableError
    with a clear "re-install with OpenMP / use backend='python'" message."""
    fake_binary = tmp_path / "missing" / "midas_indexer"
    assert not fake_binary.exists()
    monkeypatch.setattr(backend_c, "binary_path", lambda: fake_binary)

    paramstest = tmp_path / "paramstest.txt"
    paramstest.write_text("LatticeParameter 4.08 4.08 4.08 90 90 90\n")

    with pytest.raises(CBackendUnavailableError) as exc:
        backend_c.run_indexer(paramstest, n_work=10, num_procs=1)
    # The message must mention 'OpenMP' (re-install hint) and 'python'
    # (the fallback backend) — both are critical for the user to recover.
    msg = str(exc.value)
    assert "OpenMP" in msg
    assert "backend='python'" in msg
