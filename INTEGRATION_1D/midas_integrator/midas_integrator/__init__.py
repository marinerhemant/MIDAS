"""
Diffraction Image Processing and Analysis Package
------------------------------------------------

A package for processing diffraction images from 2D detectors, 
performing data integration, and fitting Voigt profiles to the results.

The package provides both programmatic and command-line interfaces.

Author: Hemant Sharma
Date: 2025/03/06
"""

__version__ = '1.0.0'

# Import main classes for easy access
from midas_integrator.core import (
    DiffractionConfig, 
    DiffractionProcessor, 
    VoigtFitter,
    GPUUtils,
    BinaryUtils,
    ImageUtils,
    ImageIntegrator
)

# Define what's available when importing with '*'
__all__ = [
    'DiffractionConfig',
    'DiffractionProcessor',
    'VoigtFitter',
    'GPUUtils', 
    'BinaryUtils',
    'ImageUtils',
    'ImageIntegrator'
]
