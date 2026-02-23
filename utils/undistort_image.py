#!/usr/bin/env python3
"""
Distortion correction for HEDM detector images.

Reads a raw TIFF image and MIDAS calibration parameters, outputs a corrected
TIFF where simple radial integration (using just beam center and Lsd) gives
correct 2-theta values. Corrects for tilts, spatial distortion (p-values),
and per-panel shifts/rotations.

Usage:
    python undistort_image.py parameters.txt input.tif output.tif

The corrected image uses bilinear interpolation from the original pixels.

Author: MIDAS / Hemant Sharma
"""

import argparse
import math
import os
import struct
import sys
import time

import numpy as np
from scipy.ndimage import map_coordinates

DEG2RAD = 0.0174532925199433
RAD2DEG = 57.2957795130823


# ──────────────────── TIFF I/O ────────────────────


def read_tif(filename):
    """Read a TIFF file. Tries tifffile, PIL, then fabio."""
    for loader in [_read_tifffile, _read_pil, _read_fabio]:
        try:
            return loader(filename)
        except (ImportError, Exception):
            continue
    raise RuntimeError(f"Cannot read {filename}. Install tifffile, Pillow, or fabio.")


def _read_tifffile(fn):
    import tifffile
    return tifffile.imread(fn).astype(np.float64)


def _read_pil(fn):
    from PIL import Image
    return np.array(Image.open(fn), dtype=np.float64)


def _read_fabio(fn):
    import fabio
    return fabio.open(fn).data.astype(np.float64)


def write_tif(filename, data):
    """Write a float32 TIFF. Tries tifffile first, then PIL."""
    arr = data.astype(np.float32)
    try:
        import tifffile
        tifffile.imwrite(filename, arr)
        return
    except ImportError:
        pass
    try:
        from PIL import Image
        Image.fromarray(arr).save(filename)
        return
    except ImportError:
        pass
    # Fallback: raw TIFF writer
    _write_raw_tiff(filename, arr)


def _write_raw_tiff(filename, arr):
    """Minimal uncompressed float32 TIFF writer (little-endian)."""
    h, w = arr.shape
    n_pixels = h * w
    with open(filename, "wb") as f:
        # Header: little-endian, magic 42, IFD at offset 8
        f.write(struct.pack("<2sHI", b"II", 42, 8))
        # IFD: 10 tags
        n_tags = 10
        f.write(struct.pack("<H", n_tags))
        strip_offset = 8 + 2 + n_tags * 12 + 4
        data_bytes = n_pixels * 4
        tags = [
            (256, 3, 1, w),             # ImageWidth
            (257, 3, 1, h),             # ImageLength
            (258, 3, 1, 32),            # BitsPerSample
            (259, 3, 1, 1),             # Compression (none)
            (262, 3, 1, 1),             # PhotometricInterp
            (273, 4, 1, strip_offset),  # StripOffsets
            (277, 3, 1, 1),             # SamplesPerPixel
            (278, 4, 1, h),             # RowsPerStrip
            (279, 4, 1, data_bytes),    # StripByteCounts
            (339, 3, 1, 3),             # SampleFormat (IEEEFP)
        ]
        for tag, typ, cnt, val in tags:
            f.write(struct.pack("<HHII", tag, typ, cnt, val))
        f.write(struct.pack("<I", 0))  # next IFD
        f.write(arr.tobytes())


# ──────────────────── Panel geometry ────────────────────


def generate_panels(npy, npz, psy, psz, gaps_y, gaps_z):
    """Generate panel definitions matching MIDAS C GeneratePanels."""
    panels = []
    idx = 0
    y_start = 0
    for iy in range(npy):
        z_start = 0
        for iz in range(npz):
            y_end = y_start + psy - 1
            z_end = z_start + psz - 1
            panels.append({
                "id": idx,
                "yMin": y_start, "yMax": y_end,
                "zMin": z_start, "zMax": z_end,
                "dY": 0.0, "dZ": 0.0, "dTheta": 0.0,
                "centerY": (y_start + y_end) / 2.0,
                "centerZ": (z_start + z_end) / 2.0,
            })
            idx += 1
            z_start = z_end + 1
            if iz < npz - 1 and iz < len(gaps_z):
                z_start += gaps_z[iz]
        y_start = y_start + psy
        if iy < npy - 1 and iy < len(gaps_y):
            y_start += gaps_y[iy]
    return panels


def load_panel_shifts(filename, panels):
    """Load panel shifts file. Format: ID dY dZ [dTheta]"""
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            pid = int(parts[0])
            for p in panels:
                if p["id"] == pid:
                    p["dY"] = float(parts[1])
                    p["dZ"] = float(parts[2])
                    if len(parts) >= 4:
                        p["dTheta"] = float(parts[3])
                    break
    return panels


def get_panel_index_map(npy, npz, panels):
    """Build a 2D panel index map for vectorized lookup."""
    if not panels:
        return None
    max_y = max(p["yMax"] for p in panels) + 1
    max_z = max(p["zMax"] for p in panels) + 1
    pmap = np.full((max_y, max_z), -1, dtype=np.int32)
    for p in panels:
        pmap[p["yMin"]:p["yMax"]+1, p["zMin"]:p["zMax"]+1] = p["id"]
    return pmap


# ──────────────────── Parameter parsing ────────────────────


def parse_parameters(param_file):
    """Parse MIDAS parameter file."""
    params = {
        "tx": 0.0, "ty": 0.0, "tz": 0.0,
        "Lsd": 1e6, "ybc": 1024.0, "zbc": 1024.0,
        "px": 200.0,
        "p0": 0.0, "p1": 0.0, "p2": 0.0, "p3": 0.0,
        "RhoD": 200000.0,
        "NrPixelsY": 2048, "NrPixelsZ": 2048,
        "NPanelsY": 0, "NPanelsZ": 0,
        "PanelSizeY": 0, "PanelSizeZ": 0,
        "PanelGapsY": [], "PanelGapsZ": [],
        "PanelShiftsFile": "",
        "Folder": "",
    }
    with open(param_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            key = parts[0]
            if key == "tx":
                params["tx"] = float(parts[1])
            elif key == "ty":
                params["ty"] = float(parts[1])
            elif key == "tz":
                params["tz"] = float(parts[1])
            elif key == "Lsd":
                params["Lsd"] = float(parts[1])
            elif key == "BC":
                params["ybc"] = float(parts[1])
                params["zbc"] = float(parts[2])
            elif key == "px":
                params["px"] = float(parts[1])
            elif key == "pxY":
                params["px"] = float(parts[1])  # Use Y pixel size
            elif key == "p0":
                params["p0"] = float(parts[1])
            elif key == "p1":
                params["p1"] = float(parts[1])
            elif key == "p2":
                params["p2"] = float(parts[1])
            elif key == "p3":
                params["p3"] = float(parts[1])
            elif key == "RhoD":
                params["RhoD"] = float(parts[1])
            elif key == "NrPixels":
                params["NrPixelsY"] = int(parts[1])
                params["NrPixelsZ"] = int(parts[1])
            elif key == "NrPixelsY":
                params["NrPixelsY"] = int(parts[1])
            elif key == "NrPixelsZ":
                params["NrPixelsZ"] = int(parts[1])
            elif key == "NPanelsY":
                params["NPanelsY"] = int(parts[1])
            elif key == "NPanelsZ":
                params["NPanelsZ"] = int(parts[1])
            elif key == "PanelSizeY":
                params["PanelSizeY"] = int(parts[1])
            elif key == "PanelSizeZ":
                params["PanelSizeZ"] = int(parts[1])
            elif key == "PanelGapsY":
                params["PanelGapsY"] = [int(x) for x in parts[1:]]
            elif key == "PanelGapsZ":
                params["PanelGapsZ"] = [int(x) for x in parts[1:]]
            elif key == "PanelShiftsFile":
                params["PanelShiftsFile"] = parts[1]
            elif key == "Folder":
                params["Folder"] = parts[1]
    return params


# ──────────────────── Core geometry ────────────────────


def build_tilt_matrix(tx, ty, tz):
    """Build the combined tilt rotation matrix Rx @ Ry @ Rz."""
    txr, tyr, tzr = tx * DEG2RAD, ty * DEG2RAD, tz * DEG2RAD
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(txr), -math.sin(txr)],
        [0, math.sin(txr),  math.cos(txr)],
    ])
    Ry = np.array([
        [ math.cos(tyr), 0, math.sin(tyr)],
        [0, 1, 0],
        [-math.sin(tyr), 0, math.cos(tyr)],
    ])
    Rz = np.array([
        [math.cos(tzr), -math.sin(tzr), 0],
        [math.sin(tzr),  math.cos(tzr), 0],
        [0, 0, 1],
    ])
    return Rx @ (Ry @ Rz)


def compute_forward_map(params, panels, panel_map):
    """
    For each raw pixel (y, z), compute its ideal output position (y_ideal, z_ideal).

    The ideal position is where this pixel would appear on a flat, untilted,
    undistorted detector at the same Lsd and beam center.

    Returns:
        map_y: (NrPixelsZ, NrPixelsY) - ideal Y pixel coord for each raw pixel
        map_z: (NrPixelsZ, NrPixelsY) - ideal Z pixel coord for each raw pixel
    """
    npy = params["NrPixelsY"]  # image columns
    npz = params["NrPixelsZ"]  # image rows
    ybc = params["ybc"]
    zbc = params["zbc"]
    px = params["px"]
    Lsd = params["Lsd"]
    RhoD = params["RhoD"]
    p0, p1, p2, p3 = params["p0"], params["p1"], params["p2"], params["p3"]

    TRs = build_tilt_matrix(params["tx"], params["ty"], params["tz"])

    # Build coordinate grids: y is column index, z is row index
    y_grid, z_grid = np.meshgrid(np.arange(npy, dtype=np.float64),
                                  np.arange(npz, dtype=np.float64))

    # 1. Apply panel corrections (shift + rotation)
    y_corr = y_grid.copy()
    z_corr = z_grid.copy()

    if panels and panel_map is not None:
        # Clip to panel_map bounds
        y_clip = np.clip(y_grid.astype(int), 0, panel_map.shape[0] - 1)
        z_clip = np.clip(z_grid.astype(int), 0, panel_map.shape[1] - 1)
        pids = panel_map[y_clip, z_clip]

        for p in panels:
            pid = p["id"]
            mask = pids == pid
            if not np.any(mask):
                continue
            dY, dZ, dTheta = p["dY"], p["dZ"], p["dTheta"]
            cY, cZ = p["centerY"], p["centerZ"]

            ym = y_grid[mask]
            zm = z_grid[mask]

            if abs(dTheta) > 1e-12:
                rad = DEG2RAD * dTheta
                cosT, sinT = math.cos(rad), math.sin(rad)
                dy_c = ym - cY
                dz_c = zm - cZ
                y_corr[mask] = cY + dy_c * cosT - dz_c * sinT + dY
                z_corr[mask] = cZ + dy_c * sinT + dz_c * cosT + dZ
            else:
                y_corr[mask] = ym + dY
                z_corr[mask] = zm + dZ

    # 2. Convert to physical coordinates (microns)
    Yc = -(y_corr - ybc) * px   # physical Y
    Zc =  (z_corr - zbc) * px   # physical Z

    # 3. Apply tilt rotation
    # Stack as [0, Yc, Zc] and multiply by TRs
    ABCPr_0 = TRs[0, 1] * Yc + TRs[0, 2] * Zc
    ABCPr_1 = TRs[1, 1] * Yc + TRs[1, 2] * Zc
    ABCPr_2 = TRs[2, 1] * Yc + TRs[2, 2] * Zc

    XYZ_0 = Lsd + ABCPr_0
    XYZ_1 = ABCPr_1
    XYZ_2 = ABCPr_2

    # 4. Compute R and Eta
    R = (Lsd / XYZ_0) * np.sqrt(XYZ_1**2 + XYZ_2**2)
    # CalcEtaAngle: alpha = acos(z/r), negate if y > 0
    r_yz = np.sqrt(XYZ_1**2 + XYZ_2**2)
    r_yz = np.maximum(r_yz, 1e-30)  # avoid division by zero
    Eta_deg = RAD2DEG * np.arccos(np.clip(XYZ_2 / r_yz, -1, 1))
    Eta_deg = np.where(XYZ_1 > 0, -Eta_deg, Eta_deg)

    # 5. Apply distortion function
    RNorm = R / RhoD
    EtaT = 90.0 - Eta_deg
    DistortFunc = (
        p0 * (RNorm**2) * np.cos(DEG2RAD * (2 * EtaT)) +
        p1 * (RNorm**4) * np.cos(DEG2RAD * (4 * EtaT + p3)) +
        p2 * (RNorm**2) +
        1.0
    )
    Rcorr = R * DistortFunc  # physical units (microns)

    # 6. Convert to ideal pixel position
    Eta_rad = Eta_deg * DEG2RAD
    Rcorr_px = Rcorr / px
    y_ideal = ybc + Rcorr_px * np.sin(Eta_rad)
    z_ideal = zbc + Rcorr_px * np.cos(Eta_rad)

    return y_ideal, z_ideal


def undistort_image(raw_image, params, panels, panel_map):
    """
    Produce an undistorted image.

    Strategy:
    1. Compute forward map: raw pixel (y, z) -> ideal pixel (y_ideal, z_ideal)
    2. Build inverse map: for each ideal pixel, find the raw pixel that maps to it
    3. Interpolate using map_coordinates
    """
    npy = params["NrPixelsY"]
    npz = params["NrPixelsZ"]
    t0 = time.time()

    # Compute forward map
    print("  Computing distortion map...")
    fwd_y, fwd_z = compute_forward_map(params, panels, panel_map)

    # Build inverse map using griddata
    # For each output pixel (y_out, z_out), we need input (y_in, z_in)
    # such that fwd_y[y_in, z_in] ≈ y_out and fwd_z[y_in, z_in] ≈ z_out
    #
    # Strategy: the displacement is small and smooth, so we approximate
    # the inverse by: input = output - displacement(output)
    # where displacement = forward(input) - input
    #
    # For better accuracy, we iterate once.

    print("  Building inverse map...")
    dy = fwd_y - np.arange(npy, dtype=np.float64)[np.newaxis, :]
    dz = fwd_z - np.arange(npz, dtype=np.float64)[:, np.newaxis]

    # First approximation of inverse
    y_out_grid, z_out_grid = np.meshgrid(
        np.arange(npy, dtype=np.float64),
        np.arange(npz, dtype=np.float64),
    )

    # Inverse approximation: y_in ≈ y_out - dy(y_out, z_out)
    # Sample dy at (y_out, z_out) using interpolation
    inv_y = y_out_grid - map_coordinates(
        dy, [z_out_grid.ravel(), y_out_grid.ravel()],
        order=1, mode="nearest"
    ).reshape(npz, npy)
    inv_z = z_out_grid - map_coordinates(
        dz, [z_out_grid.ravel(), y_out_grid.ravel()],
        order=1, mode="nearest"
    ).reshape(npz, npy)

    # One iteration of refinement:
    # Evaluate forward map at estimated input position
    dy_at_inv = map_coordinates(dy, [inv_z.ravel(), inv_y.ravel()],
                                 order=1, mode="nearest").reshape(npz, npy)
    dz_at_inv = map_coordinates(dz, [inv_z.ravel(), inv_y.ravel()],
                                 order=1, mode="nearest").reshape(npz, npy)
    inv_y = y_out_grid - dy_at_inv
    inv_z = z_out_grid - dz_at_inv

    # Apply the inverse map to the raw image
    print("  Interpolating corrected image...")
    # map_coordinates expects [row, col] = [z, y]
    corrected = map_coordinates(
        raw_image, [inv_z.ravel(), inv_y.ravel()],
        order=1, mode="constant", cval=0.0
    ).reshape(npz, npy)

    # Mark gap pixels as 0
    if panels and panel_map is not None:
        y_int = np.clip(inv_y.astype(int), 0, panel_map.shape[0] - 1)
        z_int = np.clip(inv_z.astype(int), 0, panel_map.shape[1] - 1)
        gap_mask = panel_map[y_int, z_int] < 0
        corrected[gap_mask] = 0

    dt = time.time() - t0
    print(f"  Done in {dt:.2f}s")
    return corrected


# ──────────────────── Main ────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Correct detector image distortion using MIDAS calibration parameters.",
        epilog="After correction, use just BC and Lsd for radial integration.",
    )
    parser.add_argument("parameters", help="MIDAS parameter file (.txt)")
    parser.add_argument("input", help="Input raw TIFF image")
    parser.add_argument("output", help="Output corrected TIFF image")
    args = parser.parse_args()

    print(f"Reading parameters from {args.parameters}")
    params = parse_parameters(args.parameters)

    print(f"  Image size: {params['NrPixelsY']} x {params['NrPixelsZ']} (Y x Z)")
    print(f"  Lsd: {params['Lsd']:.2f}  BC: ({params['ybc']:.2f}, {params['zbc']:.2f})")
    print(f"  Tilts: tx={params['tx']:.4f} ty={params['ty']:.4f} tz={params['tz']:.4f}")
    print(f"  p-values: {params['p0']:.6f} {params['p1']:.6f} {params['p2']:.6f} {params['p3']:.6f}")

    # Generate panels
    panels = []
    panel_map = None
    if params["NPanelsY"] > 0 and params["NPanelsZ"] > 0:
        panels = generate_panels(
            params["NPanelsY"], params["NPanelsZ"],
            params["PanelSizeY"], params["PanelSizeZ"],
            params["PanelGapsY"], params["PanelGapsZ"],
        )
        print(f"  Generated {len(panels)} panels ({params['NPanelsY']}x{params['NPanelsZ']})")

        # Load panel shifts
        shifts_file = params["PanelShiftsFile"]
        if shifts_file:
            if not os.path.isabs(shifts_file) and params["Folder"]:
                shifts_file = os.path.join(params["Folder"], shifts_file)
            if not os.path.isabs(shifts_file):
                # Try relative to parameter file
                shifts_file = os.path.join(
                    os.path.dirname(os.path.abspath(args.parameters)), shifts_file
                )
            if os.path.exists(shifts_file):
                load_panel_shifts(shifts_file, panels)
                print(f"  Loaded panel shifts from {shifts_file}")
            else:
                print(f"  Warning: Panel shifts file not found: {shifts_file}")

        panel_map = get_panel_index_map(
            params["NPanelsY"], params["NPanelsZ"], panels
        )

    # Read input image
    print(f"Reading image from {args.input}")
    raw_image = read_tif(args.input)
    print(f"  Shape: {raw_image.shape}")

    # Verify dimensions
    expected_shape = (params["NrPixelsZ"], params["NrPixelsY"])
    if raw_image.shape != expected_shape:
        print(f"  Warning: Image shape {raw_image.shape} != expected {expected_shape}")
        print(f"  Using actual image dimensions.")
        params["NrPixelsZ"] = raw_image.shape[0]
        params["NrPixelsY"] = raw_image.shape[1]
        # Regenerate panel map if needed
        if panels:
            panel_map = get_panel_index_map(
                params["NPanelsY"], params["NPanelsZ"], panels
            )

    # Undistort
    print("Correcting distortion...")
    corrected = undistort_image(raw_image, params, panels, panel_map)

    # Write output
    print(f"Writing corrected image to {args.output}")
    write_tif(args.output, corrected)
    print("Done.")


if __name__ == "__main__":
    main()
