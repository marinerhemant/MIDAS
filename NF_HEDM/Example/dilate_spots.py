#!/usr/bin/env python3
"""
Dilate SpotsInfo.bin to broaden simulated diffraction spots.

Simulated spots (from simulateNF) are single-pixel sharp. Real experiments
produce spots with Gaussian-like profiles spanning several pixels. This script
applies binary dilation to SpotsInfo.bin, expanding each spot by a given radius
in the (y, z) detector directions. This makes the simulation more realistic
and allows nearby seed orientations (not just the exact ones) to match.

The SpotsInfo.bin format is a packed bit array:
  bit_index = layer * nY * nZ * nF + y * nZ * nF + z * nF + omeBin
where nY=nrPixelsY, nZ=nrPixelsZ, nF=nrFiles.

Since nF is always divisible by 8, shifts in z (by nF bits = nF/8 bytes) and
y (by nZ*nF bits = nZ*nF/8 bytes) are byte-aligned. This allows very fast
dilation using numpy byte-array OR operations — no bit unpacking needed.

Usage:
  python3 dilate_spots.py <param_file> [--radius N] [--backup]
"""
import argparse
import os
import sys
import shutil
import numpy as np


def parse_param_file(param_path):
    """Extract nrPixelsY, nrPixelsZ, nrFiles, nLayers, DataDirectory from param file."""
    params = {
        'NrPixels': 2048,  # default
        'NrPixelsY': None,
        'NrPixelsZ': None,
        'StartNr': 0,
        'EndNr': 0,
        'DataDirectory': '.',
    }
    lsd_count = 0

    with open(param_path) as f:
        for line in f:
            stripped = line.split('#', 1)[0].strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            key, val = parts[0], parts[1]
            if key in params:
                params[key] = val
            elif key == 'Lsd':
                lsd_count += 1

    nrPixelsY = int(params['NrPixelsY']) if params['NrPixelsY'] else int(params['NrPixels'])
    nrPixelsZ = int(params['NrPixelsZ']) if params['NrPixelsZ'] else int(params['NrPixels'])
    nrFiles = int(params['EndNr']) - int(params['StartNr']) + 1
    nLayers = max(lsd_count, 1)
    data_dir = params['DataDirectory']

    return nrPixelsY, nrPixelsZ, nrFiles, nLayers, data_dir


def dilate_spotsinfo(spots_path, nrPixelsY, nrPixelsZ, nrFiles, nLayers, radius, omega_radius):
    """Dilate SpotsInfo.bin in-place using byte-level OR shifts."""
    nF = nrFiles
    nZ = nrPixelsZ
    nY = nrPixelsY

    # Byte strides for shifting
    shift_z = nF // 8          # bytes per 1 pixel in z direction
    shift_y = nZ * nF // 8     # bytes per 1 pixel in y direction
    bytes_per_layer = nY * nZ * nF // 8

    if nF % 8 != 0:
        print(f"WARNING: nrFiles={nF} is not divisible by 8. Dilation may have artifacts.")

    file_size = os.path.getsize(spots_path)
    expected_size = nLayers * bytes_per_layer
    if file_size < expected_size:
        print(f"ERROR: SpotsInfo.bin is {file_size} bytes, expected at least {expected_size}")
        sys.exit(1)

    print(f"Dilating SpotsInfo.bin:")
    print(f"  Detector: {nY} x {nZ}, {nF} frames, {nLayers} layers")
    print(f"  Spatial radius: {radius} pixels (y,z)")
    print(f"  Omega radius: {omega_radius} frames")
    print(f"  File size: {file_size} bytes ({file_size/1e6:.1f} MB)")

    # Read the entire file
    data = np.fromfile(spots_path, dtype=np.uint8)

    for layer in range(nLayers):
        start = layer * bytes_per_layer
        end = start + bytes_per_layer
        chunk = data[start:end].copy()
        result = chunk.copy()

        # Sample density from middle of layer (edges may be empty)
        mid = len(chunk) // 2
        sample = chunk[mid:mid+10000]
        bits_before = sum(bin(b).count('1') for b in sample)

        # --- Spatial dilation (y, z) ---
        for r in range(radius):
            new = result.copy()
            if shift_z < len(result):
                new[shift_z:] |= result[:-shift_z]      # z+1
                new[:-shift_z] |= result[shift_z:]       # z-1
            if shift_y < len(result):
                new[shift_y:] |= result[:-shift_y]      # y+1
                new[:-shift_y] |= result[shift_y:]       # y-1
            result = new

        # --- Omega dilation (bit-level shifts within packed bytes) ---
        # Omega is the fastest-varying index, packed LSB-first in bytes.
        # Shift +1 omega = left-shift bits by 1 with carry between bytes.
        # Shift -1 omega = right-shift bits by 1 with carry between bytes.
        for r in range(omega_radius):
            new = result.copy()
            # Omega +1: left-shift by 1 bit
            left = ((result.astype(np.uint16) << 1) & 0xFF).astype(np.uint8)
            carry = result >> 7  # bit 7 carries to bit 0 of next byte
            left[1:] |= carry[:-1]
            new |= left
            # Omega -1: right-shift by 1 bit
            right = result >> 1
            carry = (result & 1) << 7  # bit 0 carries to bit 7 of prev byte
            right[:-1] |= carry[1:]
            new |= right
            result = new

        data[start:end] = result

        # Count spots after (sample)
        bits_after = sum(bin(b).count('1') for b in result[mid:mid+10000])
        ratio = bits_after / max(bits_before, 1)
        print(f"  Layer {layer}: spot density increased ~{ratio:.1f}x (sampled from middle)")

    # Write back
    data.tofile(spots_path)
    print(f"  Written: {spots_path}")


def main():
    parser = argparse.ArgumentParser(description='Dilate SpotsInfo.bin for realistic spot sizes')
    parser.add_argument('param_file', help='MIDAS parameter file')
    parser.add_argument('--radius', type=int, default=3,
                        help='Spatial dilation radius in pixels (default: 3)')
    parser.add_argument('--omega-radius', type=int, default=0,
                        help='Omega dilation radius in frames (default: 0)')
    parser.add_argument('--backup', action='store_true',
                        help='Create SpotsInfo.bin.orig backup before dilation')
    args = parser.parse_args()

    nY, nZ, nF, nLayers, data_dir = parse_param_file(args.param_file)
    spots_path = os.path.join(data_dir, 'SpotsInfo.bin')

    if not os.path.exists(spots_path):
        print(f"ERROR: {spots_path} not found")
        sys.exit(1)

    if args.backup:
        backup = spots_path + '.orig'
        if os.path.exists(backup):
            print(f"Restoring from backup: {backup}")
            shutil.copy2(backup, spots_path)
        else:
            print(f"Backing up to {backup}")
            shutil.copy2(spots_path, backup)

    dilate_spotsinfo(spots_path, nY, nZ, nF, nLayers, args.radius, args.omega_radius)
    print("Done.")


if __name__ == '__main__':
    main()
