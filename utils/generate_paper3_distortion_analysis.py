import os
import sys
import subprocess
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------------- #
# PAPER 3: SYSTEMATIC DISTORTION ISOLATION WORKFLOW                                  #
# This script programmatically drives AutoCalibrateZarr to sequentially unlock       #
# geometric distortion terms, capturing the diminishing structural strain artifacts. #
# ---------------------------------------------------------------------------------- #

# The 6 distortion model levels (cumulative)
DISTORTION_LEVELS = [
    {"name": "1_Baseline",   "models": "none"},
    {"name": "2_Tilt",       "models": "tilt"},
    {"name": "3_Spherical",  "models": "tilt,spherical"},
    {"name": "4_Dipole",     "models": "tilt,spherical,dipole"},
    {"name": "5_Trefoil",    "models": "tilt,spherical,dipole,trefoil"},
    {"name": "6_Octupole",   "models": "tilt,spherical,dipole,trefoil,octupole"},
    {"name": "7_Pentafoil",  "models": "tilt,spherical,dipole,trefoil,octupole,pentafoil5"},
    {"name": "8_Hexafoil",   "models": "all"},
]

# Available datasets — CLI args match runAllCalibrations.sh
DATASETS = {
    "ceria_63keV": {
        "data":   "Ceria_63keV_900mm_100x100_0p5s_aero_0_001137.tif",
        "cli_args": ["--px", "150.0", "--wavelength", "0.196793",
                      "--material", "ceo2", "--im-trans", "2", "--cpus", "8"],
        "corr":   "Ceria_63keV_900mm_100x100_0p5s_aero_0_001137..tif.corr.csv",
        "label":  "Varex 4343CT (63 keV, 900 mm)",
    },
    "ceo2_pil": {
        "data":   "CeO2_Pil_100x100_att000_650mm_71p676keV_001956.tif",
        "dark":   "dark_CeO2_Pil_100x100_att000_650mm_71p676keV_001975.tif",
        "cli_args": ["--px", "172.0", "--wavelength", "0.172979",
                      "--material", "ceo2", "--im-trans", "2", "--cpus", "8",
                      "--mult-factor", "2"],
        "corr":   "CeO2_Pil_100x100_att000_650mm_71p676keV_001956..tif.corr.csv",
        "label":  "Pilatus 6M (72 keV, 650 mm)",
    },
    "ge_offset": {
        "data":   "CeO2_1s_65pt351keV_1860mm_000007.edf.ge1",
        "dark":   "dark_6s_000010.ge1",
        "cli_args": ["--px", "200.0", "--wavelength", "0.189714",
                      "--material", "ceo2", "--cpus", "8"],
        "corr":   "CeO2_1s_65pt351keV_1860mm_000007..edf.ge1.corr.csv",
        "label":  "GE Offset (65 keV, 1860 mm)",
    },
}


def build_phases(panel_mode):
    """
    Build the phase list based on the panel mode.

    Parameters
    ----------
    panel_mode : str
        'off'   — 6 phases, all without panel fitting
        'extra' — 7 phases (6 without + 1 all-with-panels at the end)
        'dual'  — 12 phases (6 without + 6 with panels, paired)
    """
    phases = []
    for level in DISTORTION_LEVELS:
        phases.append({
            "name": level["name"],
            "models": level["models"],
            "skip_panels": True,
        })
    if panel_mode == 'extra':
        phases.append({
            "name": "9_PanelFit",
            "models": "all",
            "skip_panels": False,
        })
    elif panel_mode == 'dual':
        for level in DISTORTION_LEVELS:
            phases.append({
                "name": f"{level['name']}_Panels",
                "models": level["models"],
                "skip_panels": False,
            })
    return phases


def run_phases(dataset_key, panel_mode='off'):
    """
    Executes the calibration phases of AutoCalibrateZarr.py sequentially.
    Assumes standard launch from the `Calibration` working directory.

    Parameters
    ----------
    dataset_key : str
        Key into the DATASETS dict selecting which data/params/corr to use.
    panel_mode : str
        'off', 'extra', or 'dual' — see build_phases().
    """
    ds = DATASETS[dataset_key]
    data_tif   = ds["data"]
    corr_csv   = ds["corr"]

    phases = build_phases(panel_mode)

    # Use robust pathing relative to this script precisely to find ACZ
    acz_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AutoCalibrateZarr.py")

    n_phases = len(phases)
    for idx, phase in enumerate(phases):
        print(f"\n========================================================")
        print(f"[{idx+1}/{n_phases}] Phase {phase['name']}: "
              f"Enabling models => {phase['models']}, "
              f"skip_panels={phase['skip_panels']}")
        print(f"========================================================\n")

        output_params = f"refined_params_phase_{phase['name']}_{dataset_key}.txt"

        cmd = [
            sys.executable, acz_script,
            "--data", data_tif,
            "--trimmed-mean-fraction", "0.95",
            "--n-iterations", "10",
            "--fit-p-models", phase['models'],
            "--output", output_params,
            "--no-validate",
            "--fit-residual-map", "0",
            "--plots", "0",
        ] + ds.get("cli_args", [])

        if "dark" in ds:
            cmd.extend(["--dark", ds["dark"]])

        if phase["skip_panels"]:
            cmd.append("--skip-panels")

        print(f"Executing: {' '.join(cmd)}")
        subprocess.check_call(cmd)

        # Safely archive the residual file for later plotting
        dest_csv = f"residuals_phase_{phase['name']}_{dataset_key}.csv"
        if os.path.exists(corr_csv):
            shutil.copy(corr_csv, dest_csv)
            print(f"✅ Extracted residuals to {dest_csv}")
        else:
            print(f"❌ Error: Expected output {corr_csv} not found!")


def render_publication_plot(dataset_key, panel_mode='off'):
    """
    Loads the residual files, excludes the upper 5% statistical strain outliers,
    and generates a master grid visualizing structural decay.

    For 'dual' mode, creates a 6×4 grid: left pair = no panels, right pair = with panels.
    For 'off'/'extra' mode, creates an Nx2 grid as before.
    """
    ds = DATASETS[dataset_key]
    phases = build_phases(panel_mode)
    n_phases = len(phases)

    print(f"\nRendering Paper 3 Evolution Figure for '{ds['label']}' ({n_phases} phases)...")

    if panel_mode == 'dual':
        # Two separate 6×2 figures: one without panels, one with panels
        n_levels = len(DISTORTION_LEVELS)
        for suffix, skip, tag, fname_tag in [
            ("",        True,  "No Panel Fitting", "nopanels"),
            ("_Panels", False, "+Panel Fitting",   "panels"),
        ]:
            fig, axes = plt.subplots(n_levels, 2, figsize=(14, 4.2 * n_levels))
            plt.subplots_adjust(hspace=0.4, wspace=0.2)

            for i, level in enumerate(DISTORTION_LEVELS):
                phase_name = f"{level['name']}{suffix}"
                dest_csv = f"residuals_phase_{phase_name}_{dataset_key}.csv"

                if not os.path.exists(dest_csv):
                    print(f"⚠️ Warning: {dest_csv} not found. Skipping.")
                    continue

                print(f"Plotting {dest_csv}...")
                df = pd.read_csv(dest_csv, sep=r'\s+', skiprows=3)
                df['Strain_uE'] = df['Strain'] * 1e6
                thresh = df['Strain_uE'].abs().quantile(0.95)
                inliers = df[df['Strain_uE'].abs() <= thresh]
                eta_col = '%Eta' if '%Eta' in df.columns else 'Eta'

                # Strain vs Eta
                ax_eta = axes[i, 0]
                # Draw raw points as faint background
                ax_eta.scatter(inliers[eta_col], inliers['Strain_uE'],
                               c='gray', alpha=0.1, s=1, edgecolors='none', zorder=1)
                rings = sorted(inliers['RingNr'].unique())
                colors = plt.get_cmap('turbo')(np.linspace(0, 1, max(1, len(rings))))
                for r_idx, r in enumerate(rings):
                    r_data = inliers[inliers['RingNr'] == r].sort_values(eta_col)
                    if len(r_data) >= 3:
                        # 11-point rolling median
                        r_smooth = r_data['Strain_uE'].rolling(window=11, center=True, min_periods=1).median()
                        ax_eta.scatter(r_data[eta_col], r_smooth, color=colors[r_idx], s=3, zorder=2, edgecolors='none')
                ax_eta.set_title(f"Phase {i+1}: {level['name']} | Strain vs Azimuth",
                                 fontsize=12, fontweight='bold')
                ax_eta.set_ylabel(r"Strain ($\mu\epsilon$)", fontsize=10)
                ax_eta.set_xlabel(r"Azimuth $\eta$ (deg)", fontsize=10)
                if dataset_key == 'ge_offset':
                    ax_eta.set_xlim(-180, -90)
                else:
                    ax_eta.set_xlim(-180, 180)
                ax_eta.grid(True, alpha=0.4)

                # Strain vs RingNr
                ax_rad = axes[i, 1]
                rings = sorted(inliers['RingNr'].unique())
                data = [inliers[inliers['RingNr'] == r]['Strain_uE'].values for r in rings]
                flierprops = dict(marker='.', markerfacecolor='r', markeredgecolor='none', markersize=2, alpha=0.3)
                medianprops = dict(color='black', linewidth=1.5)
                boxprops = dict(color='blue', alpha=0.7)
                whiskerprops = dict(color='black', alpha=0.5)
                ax_rad.boxplot(data, positions=rings, widths=0.5,
                               flierprops=flierprops, medianprops=medianprops,
                               boxprops=boxprops, whiskerprops=whiskerprops)
                if len(rings) > 0:
                    ax_rad.set_xlim(min(rings) - 1, max(rings) + 1)
                ax_rad.set_title(f"Phase {i+1}: {level['name']} | Strain vs Radius",
                                 fontsize=12, fontweight='bold')
                ax_rad.set_ylabel(r"Strain ($\mu\epsilon$)", fontsize=10)
                ax_rad.set_xlabel("Scattering Ring Number", fontsize=10)
                ax_rad.grid(True, alpha=0.4)

            fig.suptitle(f"Distortion Evolution: {ds['label']} ({tag})",
                         fontsize=14, fontweight='bold', y=0.995)
            plt.tight_layout(rect=[0, 0, 1, 0.99])
            plot_name = f"paper3_distortion_evolution_grid_{dataset_key}_{fname_tag}.png"
            plt.savefig(plot_name, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"🎉 Saved {plot_name}")
    else:
        # Standard Nx2 grid
        fig, axes = plt.subplots(n_phases, 2, figsize=(14, 4.2 * n_phases))
        plt.subplots_adjust(hspace=0.4, wspace=0.2)

        if n_phases == 1:
            axes = axes[np.newaxis, :]

        for i, phase in enumerate(phases):
            dest_csv = f"residuals_phase_{phase['name']}_{dataset_key}.csv"

            if not os.path.exists(dest_csv):
                print(f"⚠️ Warning: {dest_csv} not found. Skipping plot row {i+1}.")
                continue

            print(f"Plotting {dest_csv}...")
            df = pd.read_csv(dest_csv, sep=r'\s+', skiprows=3)
            df['Strain_uE'] = df['Strain'] * 1e6
            thresh = df['Strain_uE'].abs().quantile(0.95)
            inliers = df[df['Strain_uE'].abs() <= thresh]
            eta_col = '%Eta' if '%Eta' in df.columns else 'Eta'
            panel_tag = " [+Panels]" if not phase["skip_panels"] else ""

            ax_eta = axes[i, 0]
            # Draw raw points as faint background
            ax_eta.scatter(inliers[eta_col], inliers['Strain_uE'],
                           c='gray', alpha=0.1, s=1, edgecolors='none', zorder=1)
            rings = sorted(inliers['RingNr'].unique())
            colors = plt.get_cmap('turbo')(np.linspace(0, 1, max(1, len(rings))))
            for r_idx, r in enumerate(rings):
                r_data = inliers[inliers['RingNr'] == r].sort_values(eta_col)
                if len(r_data) >= 3:
                    # 11-point rolling median
                    r_smooth = r_data['Strain_uE'].rolling(window=11, center=True, min_periods=1).median()
                    ax_eta.scatter(r_data[eta_col], r_smooth, color=colors[r_idx], s=3, zorder=2, edgecolors='none')
            ax_eta.set_title(f"Phase {i+1}: {phase['name']}{panel_tag} | Strain vs Azimuth",
                             fontsize=12, fontweight='bold')
            ax_eta.set_ylabel(r"Strain ($\mu\epsilon$)", fontsize=10)
            ax_eta.set_xlabel(r"Azimuth $\eta$ (deg)", fontsize=10)
            if dataset_key == 'ge_offset':
                ax_eta.set_xlim(-180, -90)
            else:
                ax_eta.set_xlim(-180, 180)
            ax_eta.grid(True, alpha=0.4)

            ax_rad = axes[i, 1]
            rings = sorted(inliers['RingNr'].unique())
            data = [inliers[inliers['RingNr'] == r]['Strain_uE'].values for r in rings]
            flierprops = dict(marker='.', markerfacecolor='r', markeredgecolor='none', markersize=2, alpha=0.3)
            medianprops = dict(color='black', linewidth=1.5)
            boxprops = dict(color='blue', alpha=0.7)
            whiskerprops = dict(color='black', alpha=0.5)
            ax_rad.boxplot(data, positions=rings, widths=0.5,
                           flierprops=flierprops, medianprops=medianprops,
                           boxprops=boxprops, whiskerprops=whiskerprops)
            if len(rings) > 0:
                ax_rad.set_xlim(min(rings) - 1, max(rings) + 1)
            ax_rad.set_title(f"Phase {i+1}: {phase['name']}{panel_tag} | Strain vs Radius",
                             fontsize=12, fontweight='bold')
            ax_rad.set_ylabel(r"Strain ($\mu\epsilon$)", fontsize=10)
            ax_rad.set_xlabel("Scattering Ring Number", fontsize=10)
            ax_rad.grid(True, alpha=0.4)

        fig.suptitle(f"Distortion Evolution: {ds['label']}",
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plot_name = f"paper3_distortion_evolution_grid_{dataset_key}.png"
        plt.savefig(plot_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {plot_name}")
        # Copy to paper directory
        paper_dir = '/Users/hsharma/Documents/3Papers/paper3_calibration'
        if os.path.isdir(paper_dir):
            shutil.copy(plot_name, os.path.join(paper_dir, plot_name))
            print(f"Copied to {paper_dir}/{plot_name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Paper 3 Distortion Orchestrator")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip calibration and just regenerate plots from existing CSVs")
    parser.add_argument("--dataset", type=str, default="ceria_63keV",
                        choices=list(DATASETS.keys()),
                        help=f"Dataset to run: {', '.join(DATASETS.keys())} (default: ceria_63keV)")
    parser.add_argument("--panel-mode", type=str, default="off",
                        choices=["off", "extra", "dual"],
                        help="Panel fitting mode: "
                             "'off' = 6 phases no panels (default), "
                             "'extra' = 7 phases (6 + 1 all-with-panels), "
                             "'dual' = 12 phases (6 without + 6 with panels)")
    # Backward compat
    parser.add_argument("--no-panel-fit", action="store_true",
                        help="(Deprecated) Same as --panel-mode off")
    args = parser.parse_args()

    # Resolve panel mode
    panel_mode = args.panel_mode
    if args.no_panel_fit:
        panel_mode = 'off'

    if args.dataset not in DATASETS:
        print(f"❌ Unknown dataset '{args.dataset}'. Available: {list(DATASETS.keys())}")
        sys.exit(1)

    phases = build_phases(panel_mode)
    print(f"Dataset: {DATASETS[args.dataset]['label']}")
    print(f"Panel mode: {panel_mode}")
    print(f"Total phases: {len(phases)}")

    if not args.plot_only:
        run_phases(args.dataset, panel_mode=panel_mode)

    render_publication_plot(args.dataset, panel_mode=panel_mode)

