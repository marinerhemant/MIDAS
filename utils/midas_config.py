
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
    MIDAS_GUI_DIR = os.path.join(MIDAS_ROOT, 'gui')
    MIDAS_UTILS_DIR = os.path.join(MIDAS_ROOT, 'utils')
    MIDAS_FF_DIR = os.path.join(MIDAS_ROOT, 'FF_HEDM')
    MIDAS_NF_DIR = os.path.join(MIDAS_ROOT, 'NF_HEDM')
else:
    # Set to empty strings to avoid crashes on import, but logic will fail later
    MIDAS_BIN_DIR = ""
    MIDAS_GUI_DIR = ""
    MIDAS_UTILS_DIR = ""
    MIDAS_FF_DIR = ""
    MIDAS_NF_DIR = ""
