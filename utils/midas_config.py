
"""
midas_config.py - Centralized configuration and path resolution for MIDAS.

This module resolves the MIDAS root directory and provides standard paths
for binaries, GUIs, and utility scripts. It eliminates the need for 
hardcoded paths (e.g. ~/opt/MIDAS) in individual scripts.

Usage:
    from midas_utils import midas_config
    bin_dir = midas_config.MIDAS_BIN_DIR
"""

import os
import sys
import glob
import time

def get_midas_root():
    """
    Determine the MIDAS root directory.
    Priority:
    1. MIDAS_ROOT environment variable
    2. MIDAS_HOME environment variable
    3. Relative to this file (assuming this file is in <root>/utils)
    4. Fallback to ~/opt/MIDAS (with a warning)
    """
    # 1. Check Env Vars
    if 'MIDAS_INSTALL_DIR' in os.environ:
        return os.environ['MIDAS_INSTALL_DIR']
    
    if 'MIDAS_ROOT' in os.environ:
        return os.environ['MIDAS_ROOT']
    
    if 'MIDAS_HOME' in os.environ:
        return os.environ['MIDAS_HOME']
    
    # 2. Derive from script location (assuming <ROOT>/utils/midas_config.py)
    # This file is in utils/, so root is one level up.
    try:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        derived_root = os.path.dirname(this_dir)
        # minimal check: does FF_HEDM exist there?
        if os.path.isdir(os.path.join(derived_root, 'FF_HEDM')):
            return derived_root
    except Exception:
        pass

    # 3. Fallback (Legacy) - Warn user
    legacy_path = os.path.expanduser('~/opt/MIDAS')
    if os.path.isdir(legacy_path):
        print("WARNING: MIDAS_ROOT/MIDAS_HOME not set. Falling back to default: " + legacy_path, file=sys.stderr)
        return legacy_path
    
    # 4. Fail
    print("ERROR: Could not determine MIDAS_ROOT. Please set MIDAS_ROOT environment variable.", file=sys.stderr)
    return None

MIDAS_ROOT = get_midas_root()

if MIDAS_ROOT:
    MIDAS_BIN_DIR = os.path.join(MIDAS_ROOT, 'FF_HEDM', 'bin')
    MIDAS_NF_BIN_DIR = os.path.join(MIDAS_ROOT, 'NF_HEDM', 'bin')
    MIDAS_TOMO_BIN_DIR = os.path.join(MIDAS_ROOT, 'TOMO', 'bin')
    MIDAS_GUI_DIR = os.path.join(MIDAS_ROOT, 'gui')
    MIDAS_UTILS_DIR = os.path.join(MIDAS_ROOT, 'utils')
    MIDAS_FF_DIR = os.path.join(MIDAS_ROOT, 'FF_HEDM')
    MIDAS_NF_DIR = os.path.join(MIDAS_ROOT, 'NF_HEDM')
else:
    # Set to empty strings to avoid crashes on import, but logic will fail later
    MIDAS_BIN_DIR = ""
    MIDAS_NF_BIN_DIR = ""
    MIDAS_TOMO_BIN_DIR = ""
    MIDAS_GUI_DIR = ""
    MIDAS_UTILS_DIR = ""
    MIDAS_FF_DIR = ""
    MIDAS_NF_DIR = ""

# ─── Build Notification System ────────────────────────────────────────────────
# Guards to ensure checks run at most once per process
_startup_checks_done = False

def check_build_staleness():
    """
    Compare the newest C source file mtime against the oldest binary mtime.
    If any source is newer than the oldest binary, warn the user to rebuild.
    """
    if not MIDAS_ROOT:
        return
    src_dirs = [
        os.path.join(MIDAS_ROOT, 'FF_HEDM', 'src'),
        os.path.join(MIDAS_ROOT, 'NF_HEDM', 'src'),
    ]
    bin_dirs = [MIDAS_BIN_DIR, MIDAS_NF_BIN_DIR]

    # Collect all .c source file mtimes
    src_files = []
    for d in src_dirs:
        if os.path.isdir(d):
            src_files.extend(glob.glob(os.path.join(d, '*.c')))
    if not src_files:
        return

    # Collect all binary file mtimes (exclude .py, .txt, etc.)
    bin_files = []
    for d in bin_dirs:
        if os.path.isdir(d):
            for f in os.listdir(d):
                fp = os.path.join(d, f)
                if os.path.isfile(fp) and not f.endswith(('.py', '.txt', '.csv', '.md')):
                    bin_files.append(fp)
    if not bin_files:
        # No binaries found at all — user hasn't built yet
        print(
            "\n"
            "  ╔══════════════════════════════════════════════════════════════╗\n"
            "  ║  ⚠️  MIDAS binaries not found!                              ║\n"
            "  ║  Please build first:                                        ║\n"
            "  ║     cd MIDAS && ./build.sh                                  ║\n"
            "  ╚══════════════════════════════════════════════════════════════╝\n",
            file=sys.stderr
        )
        return

    newest_src_time = max(os.path.getmtime(f) for f in src_files)
    oldest_bin_time = min(os.path.getmtime(f) for f in bin_files)

    if newest_src_time > oldest_bin_time:
        # Find which source changed
        changed_sources = [os.path.basename(f) for f in src_files
                           if os.path.getmtime(f) > oldest_bin_time]
        n = len(changed_sources)
        example = changed_sources[0] if changed_sources else "unknown"
        print(
            "\n"
            "  ╔══════════════════════════════════════════════════════════════╗\n"
            "  ║  ⚠️  MIDAS binaries may be out of date!                     ║\n"
           f"  ║  {n} source file(s) are newer than binaries"
           f"{' ' * max(0, 21 - len(str(n)))}║\n"
           f"  ║  (e.g. {example})"
           f"{' ' * max(0, 48 - len(example))}║\n"
            "  ║  Please rebuild:  cd MIDAS/build && cmake --build .         ║\n"
            "  ╚══════════════════════════════════════════════════════════════╝\n",
            file=sys.stderr
        )


def check_update_reminder():
    """
    Check if it has been more than 14 days since the last build/update.
    Uses build/.last_update_check as a timestamp file.
    """
    if not MIDAS_ROOT:
        return
    stamp_file = os.path.join(MIDAS_ROOT, 'build', '.last_update_check')
    if not os.path.exists(stamp_file):
        return  # Never built via build.sh — staleness check handles this

    try:
        age_seconds = time.time() - os.path.getmtime(stamp_file)
        age_days = int(age_seconds / 86400)
        if age_days > 14:
            print(
                "\n"
                "  ╔══════════════════════════════════════════════════════════════╗\n"
               f"  ║  ℹ️  It has been {age_days} days since your last MIDAS build."
               f"{' ' * max(0, 22 - len(str(age_days)))}║\n"
                "  ║  Consider updating:                                         ║\n"
                "  ║     cd MIDAS && git pull && cd build && cmake --build .      ║\n"
                "  ╚══════════════════════════════════════════════════════════════╝\n",
                file=sys.stderr
            )
            # Touch the file so we don't nag on every single run
            os.utime(stamp_file, None)
    except OSError:
        pass  # File system issues — don't crash the workflow


def run_startup_checks():
    """
    Run all build notification checks. Safe to call from any entry point.
    Runs at most once per process.
    """
    global _startup_checks_done
    if _startup_checks_done:
        return
    _startup_checks_done = True

    try:
        check_build_staleness()
        check_update_reminder()
    except Exception:
        pass  # Never let notification checks crash a workflow

