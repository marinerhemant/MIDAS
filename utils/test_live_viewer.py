#!/usr/bin/env python3
"""
Generate synthetic binary data to test live_viewer.py

Writes fake lineout.bin and fit.bin at ~30 fps, simulating
IntegratorFitPeaksGPUStream output with drifting peaks.

Usage:
    python test_live_viewer.py --nRBins 500 --nPeaks 3 --fps 30
"""
import argparse
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Generate test data for live_viewer')
    parser.add_argument('--nRBins', type=int, default=500)
    parser.add_argument('--nPeaks', type=int, default=3)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--nFrames', type=int, default=1000)
    args = parser.parse_args()

    dt = 1.0 / args.fps
    r_values = np.linspace(100, 600, args.nRBins)

    # Peak positions (will drift slowly)
    peak_centers = np.linspace(200, 500, args.nPeaks)
    peak_widths = np.full(args.nPeaks, 8.0)
    peak_amps = np.linspace(1000, 500, args.nPeaks)

    print(f"Writing lineout.bin + fit.bin ({args.nRBins} bins, {args.nPeaks} peaks, {args.fps} fps)")
    print(f"Test with: python live_viewer.py --lineout lineout.bin --fit fit.bin "
          f"--nRBins {args.nRBins} --nPeaks {args.nPeaks}")

    with open('lineout.bin', 'wb') as f_lineout, open('fit.bin', 'wb') as f_fit:
        for frame in range(args.nFrames):
            # Background + noise
            bg = 10.0
            intensity = np.random.normal(bg, 2.0, args.nRBins)

            # Add peaks with slow drift
            drift = 5.0 * np.sin(2 * np.pi * frame / 200)
            fit_record = []

            for p in range(args.nPeaks):
                c = peak_centers[p] + drift * (p + 1) / args.nPeaks
                sigma = peak_widths[p] + 0.5 * np.sin(frame / 50)
                amp = peak_amps[p] * (1 + 0.1 * np.sin(frame / 30 + p))
                eta = 0.3 + 0.2 * np.sin(frame / 100)

                # Add peak to lineout
                gauss = np.exp(-0.5 * ((r_values - c) / sigma) ** 2)
                lorentz = 1.0 / (1.0 + ((r_values - c) / sigma) ** 2)
                peak = amp * (eta * lorentz + (1 - eta) * gauss)
                intensity += peak

                # Fit record: [Imax, BG, η, Center, σ, GoF, Area]
                gamma = sigma * 2.355
                area = amp * gamma / 2.0 * (eta * np.pi + (1 - eta) * np.sqrt(np.pi / np.log(2)))
                fit_record.extend([amp, bg, eta, c, sigma, 0.01, area])

            # Write lineout: interleaved [R0, I0, R1, I1, ...]
            record = np.empty(args.nRBins * 2)
            record[0::2] = r_values
            record[1::2] = intensity
            f_lineout.write(record.astype(np.float64).tobytes())
            f_lineout.flush()

            # Write fit
            f_fit.write(np.array(fit_record, dtype=np.float64).tobytes())
            f_fit.flush()

            if (frame + 1) % 100 == 0:
                print(f"  Frame {frame + 1}/{args.nFrames}")

            time.sleep(dt)

    print("Done.")


if __name__ == '__main__':
    main()
