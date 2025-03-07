"""
Plotting utilities for diffraction analysis.

This module provides helper functions for creating publication-quality plots
of diffraction data and fitted peaks.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Union
from pathlib import Path

from midas_integrator.core import VoigtFitter


def set_publication_style():
    """
    Configure matplotlib for publication-quality plots.
    
    Sets font sizes, line widths, and other properties for high-quality output.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Set font sizes
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    
    # Set line properties
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    
    # Set figure properties
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 300
    
    # Set font family
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Margins
    plt.rcParams['axes.xmargin'] = 0.05
    plt.rcParams['axes.ymargin'] = 0.05


def plot_diffraction_profile(
    x: np.ndarray, 
    y: np.ndarray, 
    title: str = 'Diffraction Profile', 
    xlabel: str = 'Radius (pixels)',
    ylabel: str = 'Intensity (a.u.)',
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    marker: str = 'o',
    markersize: int = 3,
    linewidth: int = 1,
    grid: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Plot diffraction data.
    
    Parameters:
    -----------
    x : np.ndarray
        X values (typically radius)
    y : np.ndarray
        Y values (typically intensity)
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    output_file : Optional[str]
        Path to save the figure
    figsize : Tuple[int, int]
        Figure size in inches
    marker : str
        Marker style
    markersize : int
        Size of markers
    linewidth : int
        Line width
    grid : bool
        Whether to show grid
    show : bool
        Whether to display the plot
        
    Returns:
    --------
    plt.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot data
    ax.plot(x, y, marker=marker, markersize=markersize, linewidth=linewidth)
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Grid
    if grid:
        ax.grid(True, alpha=0.3)
    
    # Save if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show:
        plt.show()
    
    return fig


def plot_log_scale_profile(
    x: np.ndarray, 
    y: np.ndarray, 
    title: str = 'Diffraction Profile (Log Scale)', 
    xlabel: str = 'Radius (pixels)',
    ylabel: str = 'Intensity (log scale)',
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    marker: str = 'o',
    markersize: int = 3,
    linewidth: int = 1,
    grid: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Plot diffraction data with logarithmic intensity scale.
    
    Parameters match plot_diffraction_profile, but with log scale for y-axis.
    
    Returns:
    --------
    plt.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot data with log scale
    ax.semilogy(x, y, marker=marker, markersize=markersize, linewidth=linewidth)
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Grid
    if grid:
        ax.grid(True, alpha=0.3)
    
    # Save if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show:
        plt.show()
    
    return fig


def plot_peaks(
    x: np.ndarray, 
    y: np.ndarray, 
    params: np.ndarray,
    num_peaks: int = 1,
    title: str = 'Diffraction Profile with Fitted Peaks',
    xlabel: str = 'Radius (pixels)',
    ylabel: str = 'Intensity (a.u.)',
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    marker: str = 'o',
    markersize: int = 3,
    linewidth: int = 2,
    grid: bool = True,
    log_scale: bool = False,
    show_parameters: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Plot diffraction data with fitted Voigt peaks.
    
    Parameters:
    -----------
    x : np.ndarray
        X values (typically radius)
    y : np.ndarray
        Y values (typically intensity)
    params : np.ndarray
        Fitted peak parameters (5 per peak: amp, bg, mix, cen, width)
    num_peaks : int
        Number of peaks
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    output_file : Optional[str]
        Path to save the figure
    figsize : Tuple[int, int]
        Figure size in inches
    marker : str
        Marker style
    markersize : int
        Size of markers
    linewidth : int
        Line width
    grid : bool
        Whether to show grid
    log_scale : bool
        Whether to use logarithmic y-scale
    show_parameters : bool
        Whether to annotate peak parameters
    show : bool
        Whether to display the plot
        
    Returns:
    --------
    plt.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set log scale if requested
    if log_scale:
        ax.set_yscale('log')
    
    # Plot data points
    ax.plot(x, y, marker=marker, markersize=markersize, linewidth=0, label='Data', alpha=0.7)
    
    # Plot fitted peaks
    colors = plt.cm.tab10.colors
    
    if num_peaks == 1:
        # Single peak
        y_fit = VoigtFitter.func_voigt(x, params[0], params[1], params[2], params[3], params[4])
        ax.plot(x, y_fit, 'r-', linewidth=linewidth, label='Voigt Fit')
        
        # Add peak parameters
        if show_parameters:
            amp, bg, mix, cen, width = params
            ax.annotate(f'Peak: A={amp:.1f}, BG={bg:.1f}, M={mix:.2f}, C={cen:.1f}, W={width:.1f}',
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    else:
        # Multiple peaks
        y_fit_total = np.zeros_like(x, dtype=float)
        
        for i in range(num_peaks):
            peak_params = params[i*5:(i+1)*5]
            y_fit_i = VoigtFitter.func_voigt(x, *peak_params)
            y_fit_total += y_fit_i
            
            # Plot individual peak
            ax.plot(x, y_fit_i, '--', color=colors[i % len(colors)], linewidth=1, 
                     label=f'Peak {i+1}')
            
            # Add peak parameters
            if show_parameters:
                amp, bg, mix, cen, width = peak_params
                ax.annotate(f'Peak {i+1}: A={amp:.1f}, C={cen:.1f}, W={width:.1f}',
                            xy=(0.05, 0.95 - 0.05*i), xycoords='axes fraction',
                            color=colors[i % len(colors)],
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Plot total fit
        ax.plot(x, y_fit_total, 'r-', linewidth=linewidth, label='Total Fit')
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Grid
    if grid:
        ax.grid(True, alpha=0.3)
    
    # Legend
    ax.legend(loc='best')
    
    # Save if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show:
        plt.show()
    
    return fig
