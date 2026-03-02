#!/usr/bin/env python3
"""
Generate synthetic binary data to test live_viewer.py

Always generates visible peaks in the lineout, but does NOT write fit.bin
initially — simulating a GPU integrator with no PeakLocation configured.
When peak_update.txt is received (from live_viewer's Send Peaks button),
the script starts writing fit.bin for the selected peaks, just like the
real IntegratorFitPeaksGPUStream would.

Usage:
    # Terminal 1: start the data generator (always generates 3 peaks)
    python test_live_viewer.py --fps 10

    # Terminal 2: start the viewer with no initial peaks
    python live_viewer.py --lineout lineout.bin --fit fit.bin \
        --nRBins 500 --nPeaks 0 --params test_params.txt --theme dark

    # In the viewer: click Pick → click on peaks → Send (Replace)
    # The generator will read peak_update.txt and start writing fit.bin
"""
import argparse
import os
import time
import numpy as np


def write_mock_params(path, lsd=1_000_000.0, px=200.0, wavelength=0.172979):
    """Write a minimal MIDAS parameter file with geometry."""
    with open(path, 'w') as f:
        f.write(f"# Mock parameter file for live_viewer testing\n")
        f.write(f"Lsd {lsd}\n")
        f.write(f"px {px}\n")
        f.write(f"Wavelength {wavelength}\n")
        f.write(f"NrPixels 2048\n")
        f.write(f"RMin 100\n")
        f.write(f"RMax 600\n")
        f.write(f"RBinSize 1.0\n")
    print(f"  Wrote {path}  (Lsd={lsd}, px={px}, λ={wavelength})")


def pseudo_voigt(r, center, sigma, amp, eta):
    """Generate a pseudo-Voigt peak."""
    gauss = np.exp(-0.5 * ((r - center) / sigma) ** 2)
    lorentz = 1.0 / (1.0 + ((r - center) / sigma) ** 2)
    return amp * (eta * lorentz + (1 - eta) * gauss)


def main():
    parser = argparse.ArgumentParser(
        description='Generate test data for live_viewer (interactive peak selection)')
    parser.add_argument('--nRBins', type=int, default=500)
    parser.add_argument('--fps', type=float, default=10)
    parser.add_argument('--nFrames', type=int, default=5000)
    parser.add_argument('--outdir', default='.', help='Output directory')
    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    dt = 1.0 / args.fps
    r_values = np.linspace(100, 600, args.nRBins)

    # Peaks always present in the DATA (visible in lineout)
    data_peaks = [
        {'center': 200.0, 'sigma': 8.0, 'amp': 800.0},
        {'center': 350.0, 'sigma': 10.0, 'amp': 600.0},
        {'center': 480.0, 'sigma': 7.0, 'amp': 500.0},
    ]

    # Write mock parameter file
    params_path = os.path.join(outdir, 'test_params.txt')
    write_mock_params(params_path)

    lineout_path = os.path.join(outdir, 'lineout.bin')
    fit_path = os.path.join(outdir, 'fit.bin')
    update_path = os.path.join(outdir, 'peak_update.txt')
    ap_path = os.path.join(outdir, 'active_peaks.txt')

    # Clean up old files
    for p in [lineout_path, fit_path, ap_path, update_path]:
        if os.path.exists(p):
            os.remove(p)

    print(f"Generating lineout data with {len(data_peaks)} visible peaks")
    print(f"  Peak centers: {[p['center'] for p in data_peaks]}")
    print(f"  NO fit.bin initially — waiting for peak_update.txt from viewer")
    print(f"\nRun the viewer in another terminal:")
    print(f"  python live_viewer.py --lineout {lineout_path} --fit {fit_path} "
          f"--nRBins {args.nRBins} --nPeaks 0 "
          f"--params {params_path} --theme dark")
    print(f"\nThen: click 🎯 Pick → click on the 3 peaks → 📤 Send (Replace)\n")

    # Fitting state — starts disabled
    fitting_enabled = False
    fit_peaks = []  # R values being fit (from peak_update.txt)

    with open(lineout_path, 'wb') as f_lineout:
        f_fit = None  # opened lazily

        for frame in range(args.nFrames):
            # --- Check for peak_update.txt (simulates GPU check_peak_update) ---
            if os.path.exists(update_path):
                try:
                    with open(update_path) as uf:
                        lines = uf.readlines()
                    os.remove(update_path)

                    mode = 'replace'
                    new_peaks = []
                    for line in lines:
                        line = line.strip()
                        if line.startswith('mode'):
                            mode = line.split()[1]
                        elif line:
                            new_peaks.append(float(line))

                    if new_peaks:
                        if mode == 'replace':
                            fit_peaks = new_peaks
                        else:
                            fit_peaks.extend(new_peaks)

                        # Open fit.bin lazily
                        if f_fit is None:
                            f_fit = open(fit_path, 'wb')
                            print(f"  [Frame {frame}] Opened fit.bin for writing")

                        fitting_enabled = True

                        # Write active_peaks.txt
                        with open(ap_path, 'w') as apf:
                            for r in fit_peaks:
                                apf.write(f'{r:.4f}\n')

                        print(f"  [Frame {frame}] PeakUpdate: mode={mode}, "
                              f"nPeaks={len(fit_peaks)}, R={fit_peaks}")
                except Exception as e:
                    print(f"  Warning: failed to read peak_update.txt: {e}")

            # --- Generate lineout (always has visible peaks) ---
            bg = 10.0
            drift = 3.0 * np.sin(2 * np.pi * frame / 300)
            intensity = np.random.normal(bg, 1.5, args.nRBins)

            for pk in data_peaks:
                c = pk['center'] + drift
                sigma = pk['sigma'] + 0.3 * np.sin(frame / 80)
                amp = pk['amp'] * (1 + 0.08 * np.sin(frame / 40))
                eta = 0.35
                intensity += pseudo_voigt(r_values, c, sigma, amp, eta)

            # Write lineout: interleaved [R0, I0, R1, I1, ...]
            record = np.empty(args.nRBins * 2)
            record[0::2] = r_values
            record[1::2] = intensity
            f_lineout.write(record.astype(np.float64).tobytes())
            f_lineout.flush()

            # --- Write fit.bin (only if fitting is enabled) ---
            if fitting_enabled and f_fit is not None:
                fit_record = []
                for r_fit in fit_peaks:
                    # Find the closest data peak to simulate a fit result
                    c = r_fit + drift
                    sigma = 8.0 + 0.3 * np.sin(frame / 80)
                    amp = 600.0 * (1 + 0.08 * np.sin(frame / 40))
                    eta = 0.35
                    gamma = sigma * 2.355
                    area = amp * gamma / 2.0 * (eta * np.pi +
                           (1 - eta) * np.sqrt(np.pi / np.log(2)))
                    # [Imax, BG, η, Center, σ, GoF, Area]
                    fit_record.extend([amp, bg, eta, c, sigma, 0.01, area])

                f_fit.write(np.array(fit_record, dtype=np.float64).tobytes())
                f_fit.flush()

            if (frame + 1) % 100 == 0:
                status = f"fitting {len(fit_peaks)} peaks" if fitting_enabled else "no fitting"
                print(f"  Frame {frame + 1}/{args.nFrames} ({status})")

            time.sleep(dt)

        if f_fit:
            f_fit.close()

    print("Done.")


if __name__ == '__main__':
    main()
