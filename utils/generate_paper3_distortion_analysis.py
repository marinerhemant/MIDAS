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

# The 6 distinct harmonic fitting phases exactly as described in the implementation plan
PHASES = [
    {"name": "1_Baseline",  "models": "none"},
    {"name": "2_Tilt",      "models": "tilt"},
    {"name": "3_Spherical", "models": "tilt,spherical"},
    {"name": "4_Dipole",    "models": "tilt,spherical,dipole"},
    {"name": "5_Trefoil",   "models": "tilt,spherical,dipole,trefoil"},
    {"name": "6_Octupole",  "models": "all"}
]

DATA_TIF = "Ceria_63keV_900mm_100x100_0p5s_aero_0_001137.tif"
CORR_CSV = "Ceria_63keV_900mm_100x100_0p5s_aero_0_001137..tif.corr.csv"
BASE_PARAMS = "params_starting.txt"

def run_phases():
    """
    Executes the 6 phases of AutoCalibrateZarr.py sequentially.
    Assumes standard launch from the `Calibration` working directory.
    """
    # Use robust pathing relative to this script precisely to find ACZ
    acz_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AutoCalibrateZarr.py")
    
    for idx, phase in enumerate(PHASES):
        print(f"\n========================================================")
        print(f"[{idx+1}/6] Phase {phase['name']}: Enabling models => {phase['models']}")
        print(f"========================================================\n")
        
        output_params = f"refined_params_phase_{phase['name']}.txt"
        
        cmd = [
            sys.executable, acz_script,
            "--data", DATA_TIF,
            "--params", BASE_PARAMS,
            "--trimmed-mean-fraction", "0.95",
            "--no-median", "1",
            "--n-iterations", "10",
            "--fit-p-models", phase['models'],
            "--output", output_params,
            "--no-validate" # we just want the .corr.csv, not the interactive pop-ups
        ]
        
        print(f"Executing: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        
        # Safely archive the residual file for later plotting
        dest_csv = f"residuals_phase_{phase['name']}.csv"
        if os.path.exists(CORR_CSV):
            shutil.copy(CORR_CSV, dest_csv)
            print(f"✅ Extracted residuals to {dest_csv}")
        else:
            print(f"❌ Error: Expected output {CORR_CSV} not found!")

def render_publication_plot():
    """
    Loads the 6 residual files, excludes the upper 5% statistical strain outliers, 
    and generates a master 6x2 grid visualizing structural decay.
    """
    print("\nRendering Paper 3 Evolution Figure...")
    
    fig, axes = plt.subplots(6, 2, figsize=(14, 26))
    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    
    for i, phase in enumerate(PHASES):
        dest_csv = f"residuals_phase_{phase['name']}.csv"
        
        if not os.path.exists(dest_csv):
            print(f"⚠️ Warning: {dest_csv} not found. Skipping plot row {i+1}.")
            continue
            
        print(f"Plotting {dest_csv}...")
        # The .corr.csv file has 3 lines of comma-separated metadata, then space-separated tabular data
        df = pd.read_csv(dest_csv, sep=r'\s+', skiprows=3)
        
        # Scale to microstrain
        df['Strain_uE'] = df['Strain'] * 1e6
        
        # Isolate inliers: Reject highest 5% absolute strain
        thresh = df['Strain_uE'].abs().quantile(0.95)
        inliers = df[df['Strain_uE'].abs() <= thresh]
        
        eta_col = '%Eta' if '%Eta' in df.columns else 'Eta'
        
        # Left Column: Strain vs Eta (Azimuth)
        ax_eta = axes[i, 0]
        ax_eta.scatter(inliers[eta_col], inliers['Strain_uE'], alpha=0.3, s=3, c='blue', edgecolors='none')
        ax_eta.set_title(f"Phase {i+1}: {phase['name']} | Strain vs Azimuth", fontsize=12, fontweight='bold')
        ax_eta.set_ylabel(r"Strain ($\mu\epsilon$)", fontsize=10)
        ax_eta.set_xlabel(r"Azimuth $\eta$ (deg)", fontsize=10)
        ax_eta.set_xlim(-180, 180)
        
        # Fix uniform y-limits to make the dampening mathematically apparent
        ax_eta.set_ylim(-150, 150)
        ax_eta.grid(True, alpha=0.4)
        
        # Right Column: Strain vs RingNr (Radius)
        ax_rad = axes[i, 1]
        ax_rad.scatter(inliers['RingNr'], inliers['Strain_uE'], alpha=0.3, s=3, c='red', edgecolors='none')
        ax_rad.set_title(f"Phase {i+1}: {phase['name']} | Strain vs Radius", fontsize=12, fontweight='bold')
        ax_rad.set_ylabel(r"Strain ($\mu\epsilon$)", fontsize=10)
        ax_rad.set_xlabel("Scattering Ring Number", fontsize=10)
        
        # Fix uniform y-limits
        ax_rad.set_ylim(-150, 150)
        ax_rad.grid(True, alpha=0.4)
        
    plt.tight_layout()
    plot_name = "paper3_distortion_evolution_grid.png"
    plt.savefig(plot_name, dpi=300, bbox_inches='tight')
    print(f"🎉 Successfully rendered master figure to {plot_name}!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Paper 3 Distortion Orchestrator")
    parser.add_argument("--plot-only", action="store_true", help="Skip calibration and just regenerate plots from existing CSVs")
    args = parser.parse_args()
    
    if not args.plot_only:
        run_phases()
        
    render_publication_plot()
