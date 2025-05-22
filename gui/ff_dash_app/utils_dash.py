# utils_dash.py
import numpy as np
import math
from math import sin, cos, sqrt

deg2rad = 0.0174532925199433
rad2deg = 57.2957795130823

# Extended list of distinct colors
colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
    '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7',
    '#dbdb8d', '#9edae5'
] * 5 # Repeat colors if more rings are needed

def CalcEtaAngleRad(y_physical, z_physical):
    """
    Calculates eta angle (degrees) and radius from physical coordinates.
    Args:
        y_physical: Horizontal offset from Beam Center. Positive to the LEFT.
        z_physical: Vertical offset from Beam Center. Positive UP.
    """
    if y_physical == 0 and z_physical == 0:
        return 0.0, 0.0
    Rad = sqrt(y_physical*y_physical + z_physical*z_physical)
    if Rad == 0: # Should be caught by above, but for safety
        return 0.0, 0.0

    # Ensure z_physical / Rad is within [-1, 1] due to potential floating point issues
    val_for_acos = np.clip(z_physical / Rad, -1.0, 1.0)
    alpha = rad2deg * math.acos(val_for_acos) # alpha is angle from positive Z axis towards positive Y axis

    # Correct eta based on y_physical sign
    # If y_physical > 0 (to the left of BC), eta is negative.
    # If y_physical < 0 (to the right of BC), eta is positive.
    if y_physical > 0: # Left of BC
        alpha = -alpha
    # if y_physical == 0 and z_physical < 0: alpha = 180.0 # Directly below BC, handled by acos

    return alpha, Rad


def YZ4mREta(R_physical, Eta_deg):
    """
    Calculates physical Y, Z coordinates from Radius (physical units, e.g., um)
    and Eta angle (degrees). Eta_deg can be a scalar or a NumPy array.
    Args:
        R_physical: Radius in physical units (scalar).
        Eta_deg: Eta angle in degrees (scalar or NumPy array).
    Returns:
        (y_coord, z_coord): Physical coordinates (y positive LEFT, z positive UP)
                            in same units as R_physical. If Eta_deg was an array,
                            y_coord and z_coord will also be arrays.
    """
    # Use np.deg2rad for element-wise conversion if Eta_deg is an array
    eta_rad = np.deg2rad(Eta_deg)
    
    # Use np.sin and np.cos for element-wise operations
    y_coord = -R_physical * np.sin(eta_rad) # Physical Y (positive to the LEFT of BC)
    z_coord =  R_physical * np.cos(eta_rad) # Physical Z (positive UP from BC)
    return y_coord, z_coord

def get_transform_matrix(tx_deg, ty_deg, tz_deg):
    """Computes the 3x3 rotation matrix for detector tilts."""
    txr = tx_deg * deg2rad
    tyr = ty_deg * deg2rad
    tzr = tz_deg * deg2rad
    Rx = np.array([[1, 0, 0], [0, cos(txr), -sin(txr)], [0, sin(txr), cos(txr)]])
    Ry = np.array([[cos(tyr), 0, sin(tyr)], [0, 1, 0], [-sin(tyr), 0, cos(tyr)]])
    Rz = np.array([[cos(tzr), -sin(tzr), 0], [sin(tzr), cos(tzr), 0], [0, 0, 1]])
    # Original code used Rx * Ry * Rz convention
    transform_matrix = np.dot(Rx, np.dot(Ry, Rz))
    return transform_matrix

def parse_param_line(line, keyword, dtype=str, num_values=1, default=None):
    """Helper to parse a specific line from the parameter file."""
    stripped_line = line.strip()
    if stripped_line.startswith(keyword + ' '):
        parts = stripped_line.split()
        if len(parts) > num_values:
            try:
                if num_values == 1:
                    return dtype(parts[1])
                else:
                    return [dtype(p) for p in parts[1:1+num_values]]
            except (ValueError, IndexError):
                print(f"Warning: Could not parse {keyword} from line: {stripped_line}")
                return default
    return default

def try_parse_float(value, default=0.0):
    """Safely parse a string to float."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def try_parse_int(value, default=0):
    """Safely parse a string to int."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default