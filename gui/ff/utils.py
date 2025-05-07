# utils.py
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

def CalcEtaAngleRad(y, z):
    """Calculates eta angle (degrees) and radius from y, z coordinates."""
    if y == 0 and z == 0:
        return 0.0, 0.0
    Rad = sqrt(y*y + z*z)
    # Handle potential division by zero or values slightly outside [-1, 1] due to precision
    if Rad == 0:
        return 0.0, 0.0
    z_over_rad = np.clip(z / Rad, -1.0, 1.0)
    alpha = rad2deg * math.acos(z_over_rad)
    if y > 0:
        alpha = -alpha
    return alpha, Rad

def YZ4mREta(R, Eta):
    """Calculates Y, Z coordinates from Radius (um) and Eta angle (degrees)."""
    eta_rad = Eta * deg2rad
    # Returns physical Y (left is positive), Z (up is positive) in micrometers
    y_coord = -R * sin(eta_rad)
    z_coord = R * cos(eta_rad)
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