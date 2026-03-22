#!/usr/bin/env python3
"""
Compare PeakFitMode 0 (pV) vs PeakFitMode 1 (TCH) on 4 calibrant datasets.

Uses the same dataset configs as benchmark_pyfai_vs_midas.py.
Runs ACZ twice per dataset (once per mode), then compares results.
"""

import os
import sys
import time
import subprocess
import re
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MIDAS_HOME = os.environ.get('MIDAS_HOME',
    os.path.join(os.path.expanduser('~'), 'opt', 'MIDAS'))

ACZ_SCRIPT = os.path.join(MIDAS_HOME, 'utils', 'AutoCalibrateZarr.py')

# Dataset configs (from benchmark_pyfai_vs_midas.py)
DATASETS = {
    'pilatus': {
        'label': 'Pilatus CeO₂ (172µm, 71.7keV)',
        'data_dir': os.path.join(MIDAS_HOME, 'FF_HEDM', 'Example', 'Calibration'),
        'data_file': 'CeO2_Pil_100x100_att000_650mm_71p676keV_001956.tif',
        'dark_file': 'dark_CeO2_Pil_100x100_att000_650mm_71p676keV_001975.tif',
        'mask_file': 'mask_upd.tif',
        'wavelength_A': 0.17297,
        'im_trans': 2,
        'n_iterations': 30,
    },
    'varex': {
        'label': 'Varex CeO₂ (150µm, 63keV)',
        'data_dir': os.path.join(MIDAS_HOME, 'FF_HEDM', 'Example', 'Calibration'),
        'data_file': 'Ceria_63keV_900mm_100x100_0p5s_aero_0_001137.tif',
        'dark_file': None,
        'mask_file': None,
        'wavelength_A': 0.19582,
        'im_trans': 2,
        'n_iterations': 30,
    },
    'ge5': {
        'label': 'GE5 CeO₂ non-summed (200µm, 71.7keV)',
        'data_dir': os.path.expanduser(
            '~/Desktop/analysis/lieghanne/LCG_MIDAS_TEST_03092026-selected_no_sum'),
        'data_file': 'ceria_1dfocusbeam_0deg_10f0p2s_000417.ge5.h5',
        'dark_file': 'dark_ceria_1dfocusbeam_0deg_10f0p2s_000418.ge5.h5',
        'mask_file': None,
        'wavelength_A': 0.173058,
        'im_trans': 0,
        'n_iterations': 30,
        'data_loc': '/exchange/data',
        'dark_loc': '/exchange/data',
    },
    'ge5_summed': {
        'label': 'GE5 CeO₂ summed (200µm, 71.7keV)',
        'data_dir': os.path.expanduser(
            '~/Desktop/analysis/lieghanne/LCG_MIDAS_TEST_03092026-selected_sum'),
        'data_file': 'summed_ceria_1dfocusbeam_0deg_10f0p2s_000417.ge5.h5',
        'dark_file': 'summed_dark_ceria_1dfocusbeam_0deg_10f0p2s_000418.ge5.h5',
        'mask_file': None,
        'wavelength_A': 0.173058,
        'im_trans': 0,
        'n_iterations': 30,
        'data_loc': '/exchange/data',
        'dark_loc': '/exchange/data',
    },
}

MODE_NAMES = {0: 'pV (pseudo-Voigt)', 1: 'TCH (GSAS-II)'}


def parse_acz_output(output_text):
    """Parse ACZ output for calibration results.
    
    Looks for the 'Converged / Best' block and per-ring statistics table.
    """
    result = {}
    
    # Parse Converged block
    in_converged = False
    for line in output_text.split('\n'):
        line = line.strip()
        if 'Converged' in line and 'Best' in line:
            in_converged = True
            continue
        if in_converged:
            if line.startswith('Lsd'):
                try: result['lsd_um'] = float(line.split()[1])
                except: pass
            elif line.startswith('BC'):
                try:
                    parts = line.split()
                    result['bc_y'] = float(parts[1])
                    result['bc_z'] = float(parts[2])
                except: pass
            elif line.startswith('ty'):
                try: result['ty'] = float(line.split()[1])
                except: pass
            elif line.startswith('tz'):
                try: result['tz'] = float(line.split()[1])
                except: pass
            elif line.startswith('p0'):
                try: result['p0'] = float(line.split()[1])
                except: pass
            elif line.startswith('p1'):
                try: result['p1'] = float(line.split()[1])
                except: pass
            elif line.startswith('p2'):
                try: result['p2'] = float(line.split()[1])
                except: pass
            elif line.startswith('Mean Strain') or line.startswith('MeanStrain'):
                try: result['mean_strain'] = float(line.split()[-1])
                except: pass
            elif line.startswith('Std Strain') or line.startswith('StdStrain'):
                try: result['std_strain'] = float(line.split()[-1])
                except: pass
            elif line.startswith('===='):
                if result.get('lsd_um') is not None:
                    in_converged = False

    # Fallback: parse INFO lines
    if 'lsd_um' not in result:
        for line in output_text.split('\n'):
            if 'INFO - Lsd' in line and 'lsd_um' not in result:
                try: result['lsd_um'] = float(line.strip().split()[-1])
                except: pass
            elif 'INFO - BC' in line and 'bc_y' not in result:
                try:
                    parts = line.strip().split()
                    result['bc_y'] = float(parts[-2])
                    result['bc_z'] = float(parts[-1])
                except: pass

    # Parse per-ring statistics from last iteration
    ring_stats = []
    for line in output_text.split('\n'):
        # Look for ring lines: "    1      360        16.035  ..."
        m = re.match(r'\s+(\d+)\s+(\d+)\s+([\d.+-]+)\s+([\d.+-]+)\s+([\d.+-]+)\s+([\d.+-]+)\s+([\d.+-]+)', line)
        if m:
            ring_stats.append({
                'ring': int(m.group(1)),
                'npts': int(m.group(2)),
                'mean_dr': float(m.group(3)),
                'med_dr': float(m.group(4)),
                'mean_abs_dr': float(m.group(5)),
                'med_abs_dr': float(m.group(6)),
                'mean_snr': float(m.group(7)),
            })
    
    # Keep only the last set of ring stats (final iteration)
    if ring_stats:
        # Find where the last repeated ring=1 starts
        last_start = 0
        for i, rs in enumerate(ring_stats):
            if rs['ring'] == 1 and i > 0:
                last_start = i
        result['ring_stats'] = ring_stats[last_start:]

    return result


def run_acz(dataset_cfg, peak_fit_mode=0, timeout=600):
    """Run ACZ on a dataset with the given PeakFitMode."""
    data_dir = dataset_cfg['data_dir']
    data_file = dataset_cfg['data_file']

    cmd = [
        sys.executable, ACZ_SCRIPT,
        '--data', data_file,
        '--material', 'ceo2',
        '--wavelength', str(dataset_cfg['wavelength_A']),
        '--im-trans', str(dataset_cfg['im_trans']),
        '--n-iterations', str(dataset_cfg['n_iterations']),
        '--peak-fit-mode', str(peak_fit_mode),
    ]
    if dataset_cfg.get('dark_file'):
        cmd.extend(['--dark', dataset_cfg['dark_file']])
    if dataset_cfg.get('mask_file'):
        cmd.extend(['--mask', dataset_cfg['mask_file']])
    if dataset_cfg.get('data_loc'):
        cmd.extend(['--data-loc', dataset_cfg['data_loc']])
    if dataset_cfg.get('dark_loc'):
        cmd.extend(['--dark-loc', dataset_cfg['dark_loc']])

    print(f"    Running ACZ with PeakFitMode={peak_fit_mode} ({MODE_NAMES[peak_fit_mode]})...")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=data_dir, capture_output=True, text=True,
                          timeout=timeout)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"    FAILED (exit {proc.returncode})")
        if proc.stderr:
            # Print last 300 chars of stderr
            print(f"    stderr: ...{proc.stderr[-300:]}")
        return None

    output = proc.stdout + '\n' + proc.stderr
    result = parse_acz_output(output)
    result['runtime_s'] = elapsed
    result['mode'] = peak_fit_mode
    
    if 'lsd_um' in result:
        strain_ue = result.get('mean_strain', 0) * 1e6
        print(f"    OK ({elapsed:.1f}s)  Lsd={result['lsd_um']:.1f} µm  "
              f"BC=({result.get('bc_y', 0):.2f}, {result.get('bc_z', 0):.2f})  "
              f"MeanStrain={strain_ue:.1f} µε")
    else:
        print(f"    WARNING: Could not parse output ({elapsed:.1f}s)")

    return result


def compare_two_modes(result_pv, result_tch, label):
    """Print comparison table for two modes."""
    if result_pv is None or result_tch is None:
        print(f"  Cannot compare — one or both modes failed")
        return

    print(f"\n  {'Parameter':<20s}  {'pV (mode 0)':>14s}  {'TCH (mode 1)':>14s}  {'Δ(TCH-pV)':>14s}")
    print(f"  {'-'*68}")

    lsd_pv = result_pv.get('lsd_um', 0)
    lsd_tch = result_tch.get('lsd_um', 0)
    print(f"  {'Lsd (µm)':<20s}  {lsd_pv:>14.1f}  {lsd_tch:>14.1f}  {lsd_tch-lsd_pv:>+14.1f}")

    bc_y_pv = result_pv.get('bc_y', 0)
    bc_y_tch = result_tch.get('bc_y', 0)
    print(f"  {'BC_Y (px)':<20s}  {bc_y_pv:>14.2f}  {bc_y_tch:>14.2f}  {bc_y_tch-bc_y_pv:>+14.3f}")

    bc_z_pv = result_pv.get('bc_z', 0)
    bc_z_tch = result_tch.get('bc_z', 0)
    print(f"  {'BC_Z (px)':<20s}  {bc_z_pv:>14.2f}  {bc_z_tch:>14.2f}  {bc_z_tch-bc_z_pv:>+14.3f}")

    ty_pv = result_pv.get('ty', 0)
    ty_tch = result_tch.get('ty', 0)
    print(f"  {'ty (°)':<20s}  {ty_pv:>14.4f}  {ty_tch:>14.4f}  {ty_tch-ty_pv:>+14.5f}")

    tz_pv = result_pv.get('tz', 0)
    tz_tch = result_tch.get('tz', 0)
    print(f"  {'tz (°)':<20s}  {tz_pv:>14.4f}  {tz_tch:>14.4f}  {tz_tch-tz_pv:>+14.5f}")

    ms_pv = result_pv.get('mean_strain', 0) * 1e6
    ms_tch = result_tch.get('mean_strain', 0) * 1e6
    print(f"  {'MeanStrain (µε)':<20s}  {ms_pv:>14.1f}  {ms_tch:>14.1f}  {ms_tch-ms_pv:>+14.1f}")

    ss_pv = result_pv.get('std_strain', 0) * 1e6
    ss_tch = result_tch.get('std_strain', 0) * 1e6
    print(f"  {'StdStrain (µε)':<20s}  {ss_pv:>14.1f}  {ss_tch:>14.1f}  {ss_tch-ss_pv:>+14.1f}")

    rt_pv = result_pv.get('runtime_s', 0)
    rt_tch = result_tch.get('runtime_s', 0)
    print(f"  {'Runtime (s)':<20s}  {rt_pv:>14.1f}  {rt_tch:>14.1f}  {rt_tch-rt_pv:>+14.1f}")

    # Per-ring SNR comparison if available
    rs_pv = result_pv.get('ring_stats', [])
    rs_tch = result_tch.get('ring_stats', [])
    if rs_pv and rs_tch:
        n = min(len(rs_pv), len(rs_tch))
        print(f"\n  Per-Ring Comparison (last iteration):")
        print(f"  {'Ring':>5s}  {'pV Mean|ΔR| µε':>16s}  {'TCH Mean|ΔR| µε':>16s}  "
              f"{'pV MeanSNR':>12s}  {'TCH MeanSNR':>12s}")
        print(f"  {'-'*70}")
        for i in range(n):
            rp, rt = rs_pv[i], rs_tch[i]
            print(f"  {rp['ring']:>5d}  {rp['mean_abs_dr']:>16.1f}  {rt['mean_abs_dr']:>16.1f}  "
                  f"{rp['mean_snr']:>12.1f}  {rt['mean_snr']:>12.1f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Compare PeakFitMode 0 (pV) vs 1 (TCH) on calibrant datasets')
    parser.add_argument('--pilatus', action='store_true')
    parser.add_argument('--varex', action='store_true')
    parser.add_argument('--ge5', action='store_true')
    parser.add_argument('--ge5-summed', action='store_true', dest='ge5_summed')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if args.all:
        args.pilatus = True
        args.varex = True
        args.ge5 = True
        args.ge5_summed = True
    if not (args.pilatus or args.varex or args.ge5 or args.ge5_summed):
        args.all = True
        args.pilatus = True
        args.varex = True
        args.ge5 = True
        args.ge5_summed = True

    print("=" * 72)
    print("  PeakFitMode Comparison: pV (mode 0) vs TCH (mode 1)")
    print("=" * 72)

    all_results = {}

    for name, cfg in DATASETS.items():
        if not getattr(args, name, False):
            continue

        data_path = os.path.join(cfg['data_dir'], cfg['data_file'])
        if not os.path.exists(data_path):
            print(f"\n  SKIP: {cfg['label']} — data not found at {data_path}")
            continue

        print(f"\n{'#'*72}")
        print(f"  Dataset: {cfg['label']}")
        print(f"{'#'*72}")

        # Run with mode 0 (pV)
        result_pv = run_acz(cfg, peak_fit_mode=0)

        # Run with mode 1 (TCH)
        result_tch = run_acz(cfg, peak_fit_mode=1)

        compare_two_modes(result_pv, result_tch, cfg['label'])
        all_results[name] = {'pv': result_pv, 'tch': result_tch}

    # Cross-dataset summary
    if len(all_results) > 1:
        print(f"\n\n{'='*72}")
        print("  CROSS-DATASET SUMMARY")
        print(f"{'='*72}")
        print(f"  {'Dataset':<40s}  {'pV µε':>8s}  {'TCH µε':>8s}  {'ΔLsd (µm)':>10s}")
        print(f"  {'-'*72}")
        for name, res in all_results.items():
            cfg = DATASETS[name]
            pv, tch = res['pv'], res['tch']
            if pv and tch:
                pv_ue = pv.get('mean_strain', 0) * 1e6
                tch_ue = tch.get('mean_strain', 0) * 1e6
                dlsd = tch.get('lsd_um', 0) - pv.get('lsd_um', 0)
                print(f"  {cfg['label']:<40s}  {pv_ue:>8.1f}  {tch_ue:>8.1f}  {dlsd:>+10.1f}")


if __name__ == '__main__':
    main()
