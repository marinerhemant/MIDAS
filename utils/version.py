"""
MIDAS Version Information

Provides version and git commit info for provenance tracking in output files.
"""

import subprocess
import os
import datetime
import functools


MIDAS_VERSION = "9.2"


@functools.lru_cache(maxsize=1)
def get_midas_version():
    """Return dict with version, git_commit, git_short, git_dirty, git_date, timestamp."""
    info = {"version": MIDAS_VERSION}
    midas_home = os.environ.get(
        "MIDAS_HOME",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=midas_home,
            stderr=subprocess.DEVNULL).decode().strip()
        info["git_short"] = info["git_commit"][:8]
        info["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=midas_home,
            stderr=subprocess.DEVNULL).decode().strip())
        info["git_date"] = subprocess.check_output(
            ["git", "log", "-1", "--format=%ci"], cwd=midas_home,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        info.update(git_commit="unknown", git_short="unknown",
                    git_dirty=False, git_date="unknown")
    info["timestamp"] = datetime.datetime.now().isoformat()
    return info


def version_string():
    """Return a human-readable version string, e.g. 'MIDAS v9.2 (a1b2c3d4)'."""
    v = get_midas_version()
    s = f"MIDAS v{v['version']} ({v['git_short']})"
    if v.get("git_dirty"):
        s += " [modified]"
    return s


def stamp_h5(h5_root):
    """Write version attrs to an h5py File or Group root.

    Args:
        h5_root: An h5py.File or h5py.Group object.
    """
    v = get_midas_version()
    h5_root.attrs['midas_version'] = v['version']
    h5_root.attrs['midas_git_commit'] = v['git_commit']
    h5_root.attrs['midas_git_date'] = v['git_date']
    h5_root.attrs['creation_timestamp'] = v['timestamp']


def stamp_zarr(zarr_root):
    """Write version attrs to a zarr Group root.

    Args:
        zarr_root: A zarr.Group object.
    """
    v = get_midas_version()
    zarr_root.attrs['midas_version'] = v['version']
    zarr_root.attrs['midas_git_commit'] = v['git_commit']
    zarr_root.attrs['midas_git_date'] = v['git_date']
    zarr_root.attrs['creation_timestamp'] = v['timestamp']
