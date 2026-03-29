#!/usr/bin/env python3
"""
view_peaks_binary.py - View contents of AllPeaks_PS.bin and AllPeaks_PX.bin

Usage:
    python3 view_peaks_binary.py AllPeaks_PS.bin [--frame FRAME_IDX] [--csv output.csv]
    python3 view_peaks_binary.py AllPeaks_PX.bin [--frame FRAME_IDX]
    python3 view_peaks_binary.py AllPeaks_PS.bin --summary

Author: Hemant Sharma
"""

import argparse
import struct
import sys
import os
import numpy as np

N_PEAK_COLS = 29
PEAK_COL_NAMES = [
    "SpotID", "IntegratedIntensity", "Omega", "YCen", "ZCen",
    "IMax", "Radius", "Eta", "SigmaR", "SigmaEta",
    "NrPixels", "NrPxTot", "nPeaks", "maxY", "maxZ",
    "diffY", "diffZ", "rawIMax", "returnCode", "retVal",
    "BG", "SigmaGR", "SigmaLR", "SigmaGEta", "SigmaLEta",
    "MU", "RawSumIntensity", "maskTouched", "FitRMSE",
]


def read_ps_binary(filename):
    """Read AllPeaks_PS.bin and return (nFrames, nPeaksArr, peak_data_per_frame)."""
    with open(filename, 'rb') as f:
        nFrames = struct.unpack('i', f.read(4))[0]
        nPeaksArr = list(struct.unpack(f'{nFrames}i', f.read(4 * nFrames)))
        offsets = list(struct.unpack(f'{nFrames}q', f.read(8 * nFrames)))
        remaining = f.read()
    
    header_size = 4 + 4 * nFrames + 8 * nFrames
    frames = {}
    for i in range(nFrames):
        nPk = nPeaksArr[i]
        if nPk == 0:
            continue
        byte_off = offsets[i] - header_size
        data = np.frombuffer(remaining, dtype=np.float64,
                             count=nPk * N_PEAK_COLS,
                             offset=byte_off).reshape(nPk, N_PEAK_COLS)
        frames[i] = data
    
    return nFrames, nPeaksArr, frames


def read_px_binary(filename):
    """Read AllPeaks_PX.bin and return (nFrames, nrPixels, nPeaksArr, pixel_data)."""
    with open(filename, 'rb') as f:
        nFrames = struct.unpack('i', f.read(4))[0]
        nrPixels = struct.unpack('i', f.read(4))[0]
        nPeaksArr = list(struct.unpack(f'{nFrames}i', f.read(4 * nFrames)))
        offsets = list(struct.unpack(f'{nFrames}q', f.read(8 * nFrames)))
        remaining = f.read()
    
    header_size = 4 + 4 + 4 * nFrames + 8 * nFrames
    frames = {}
    for i in range(nFrames):
        nPk = nPeaksArr[i]
        if nPk == 0:
            continue
        byte_off = offsets[i] - header_size
        pos = byte_off
        peaks = []
        for pk in range(nPk):
            nPx = struct.unpack_from('i', remaining, pos)[0]
            pos += 4
            coords = np.frombuffer(remaining, dtype=np.int16,
                                   count=nPx * 2, offset=pos).reshape(nPx, 2)
            pos += nPx * 4  # 2 int16 per pixel = 4 bytes
            peaks.append({'nPixels': nPx, 'y': coords[:, 0], 'z': coords[:, 1]})
        frames[i] = peaks
    
    return nFrames, nrPixels, nPeaksArr, frames


def main():
    parser = argparse.ArgumentParser(description="View MIDAS consolidated peak binary files")
    parser.add_argument("filename", help="Path to AllPeaks_PS.bin or AllPeaks_PX.bin")
    parser.add_argument("--frame", type=int, default=None,
                        help="Show data for specific frame (0-based index)")
    parser.add_argument("--summary", action="store_true",
                        help="Show summary statistics only")
    parser.add_argument("--csv", type=str, default=None,
                        help="Export peak data to CSV file (PS.bin only)")
    parser.add_argument("--head", type=int, default=20,
                        help="Max rows to display per frame (default: 20)")
    args = parser.parse_args()

    basename = os.path.basename(args.filename)

    if 'PS' in basename:
        nFrames, nPeaksArr, frames = read_ps_binary(args.filename)
        total_peaks = sum(nPeaksArr)
        frames_with_peaks = sum(1 for n in nPeaksArr if n > 0)
        
        print(f"AllPeaks_PS.bin Summary")
        print(f"  Total frames: {nFrames}")
        print(f"  Frames with peaks: {frames_with_peaks}")
        print(f"  Total peaks: {total_peaks}")
        if frames_with_peaks > 0:
            non_zero = [n for n in nPeaksArr if n > 0]
            print(f"  Peaks/frame: min={min(non_zero)}, max={max(non_zero)}, "
                  f"mean={np.mean(non_zero):.1f}")
        
        if args.summary:
            return
        
        if args.csv:
            # Export all data to CSV
            rows = []
            for fi, data in sorted(frames.items()):
                for row in data:
                    rows.append(np.concatenate([[fi + 1], row]))
            if rows:
                header = "FrameNr\t" + "\t".join(PEAK_COL_NAMES)
                arr = np.array(rows)
                np.savetxt(args.csv, arr, delimiter='\t', header=header,
                           comments='', fmt='%.6f')
                print(f"\nExported {len(rows)} peaks to {args.csv}")
            return
        
        if args.frame is not None:
            if args.frame in frames:
                data = frames[args.frame]
                print(f"\nFrame {args.frame} ({data.shape[0]} peaks):")
                print("\t".join(PEAK_COL_NAMES))
                for row in data[:args.head]:
                    print("\t".join(f"{v:.4f}" for v in row))
                if data.shape[0] > args.head:
                    print(f"  ... ({data.shape[0] - args.head} more rows)")
            else:
                print(f"\nFrame {args.frame}: no peaks")
        else:
            # Show first few frames
            shown = 0
            for fi in sorted(frames.keys()):
                if shown >= 5:
                    print(f"\n... {len(frames) - shown} more frames with peaks")
                    break
                data = frames[fi]
                print(f"\nFrame {fi} ({data.shape[0]} peaks):")
                print("  " + "\t".join(PEAK_COL_NAMES[:8]) + "\t...")
                for row in data[:3]:
                    print("  " + "\t".join(f"{v:.2f}" for v in row[:8]) + "\t...")
                if data.shape[0] > 3:
                    print(f"  ... ({data.shape[0] - 3} more peaks)")
                shown += 1

    elif 'PX' in basename:
        nFrames, nrPixels, nPeaksArr, frames = read_px_binary(args.filename)
        total_peaks = sum(nPeaksArr)
        frames_with_peaks = sum(1 for n in nPeaksArr if n > 0)
        
        print(f"AllPeaks_PX.bin Summary")
        print(f"  Total frames: {nFrames}")
        print(f"  NrPixels: {nrPixels}")
        print(f"  Frames with peaks: {frames_with_peaks}")
        print(f"  Total peaks: {total_peaks}")
        
        if args.summary:
            return
        
        if args.frame is not None:
            if args.frame in frames:
                peaks = frames[args.frame]
                print(f"\nFrame {args.frame} ({len(peaks)} peaks):")
                for pi, pk in enumerate(peaks[:args.head]):
                    print(f"  Peak {pi}: {pk['nPixels']} pixels, "
                          f"y=[{pk['y'][0]}..{pk['y'][-1]}], "
                          f"z=[{pk['z'][0]}..{pk['z'][-1]}]")
            else:
                print(f"\nFrame {args.frame}: no peaks")
        else:
            shown = 0
            for fi in sorted(frames.keys()):
                if shown >= 5:
                    print(f"\n... {len(frames) - shown} more frames")
                    break
                peaks = frames[fi]
                total_px = sum(p['nPixels'] for p in peaks)
                print(f"\n  Frame {fi}: {len(peaks)} peaks, {total_px} total pixels")
                shown += 1
    else:
        print(f"Unrecognized file: {basename}")
        print("Expected AllPeaks_PS.bin or AllPeaks_PX.bin")
        sys.exit(1)


if __name__ == '__main__':
    main()
