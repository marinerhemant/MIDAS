#!/usr/bin/env python3
"""
Diffraction Image Processing and Analysis - Initial Code
------------------------------------------------------
Initial code for processing diffraction images, applying data integration,
and fitting Voigt profiles to the results.

This script provides a simple example of how to integrate image data and fit a Voigt profile
to the results using the NumPy, Numba, and SciPy libraries.

Author: Hemant Sharma
Date: 2025/03/06

For copyright information, see the LICENSE file.
"""
import numpy as np
from numba import njit
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from PIL import Image
import struct
from math import ceil
import os

# Voigt profile function
# This function defines a Voigt profile, which is a combination of a Gaussian and a Lorentzian profile.
def func_voigt(x, amp, bg, mix, cen, width):
    dx = x - cen
    g = np.exp(-0.5 * (dx * dx / (width * width)) / (width * np.sqrt(2 * np.pi)))  # Gaussian component
    l = 1 / (np.pi * width * (1 + dx * dx / (width * width)))  # Lorentzian component
    return bg + amp * (mix * l + (1 - mix) * g)  # Combined Voigt profile

# Function to fit the Voigt profile to data
# This function fits the Voigt profile to the given data using non-linear least squares optimization.
def fitFunc(x, y):
    bounds = ([0, 0, 0, 0, 0], [np.max(y) * 100, np.max(y) * 100, 1, len(y), len(y) / 2])
    p0 = [np.max(y), np.median(y), 0.5, np.argmax(y), len(y) / 4]
    params, params_cov = curve_fit(func_voigt, x, y, p0=p0, bounds=bounds)
    return params, params_cov

# Function to merge two int32 values into a double
# This function combines two 32-bit integers into a single 64-bit double precision floating point number.
def merge_int32_to_double(int1, int2):
    combined_bits = (int2 << 32) | (int1 & 0xFFFFFFFF)
    return struct.unpack('d', struct.pack('Q', combined_bits))[0]

# Function to integrate image data
# This function integrates the image data over specified radial and angular bins.
@njit
def integrateImage(image, pxList, nPxList, fracValues, nRBins, nEtaBins, RMin, RBinSize, badPxIntensity, gapIntensity, nrPixelsY):
    result = np.zeros((nRBins, 2), dtype=np.float32)
    for i in range(nRBins):
        Int1D = 0
        n1ds = 0
        RMean = (RMin + (i + 0.5) * RBinSize)
        for j in range(nEtaBins):
            Pos = i * nEtaBins + j
            nPixels = nPxList[2 * Pos + 0]
            dataPos = nPxList[2 * Pos + 1]
            Intensity = 0
            totArea = 0
            if nPixels == 0:
                continue
            for k in range(nPixels):
                ThisVal = pxList[dataPos + k]
                testPos = ThisVal[1], ThisVal[0]
                if image[testPos] == badPxIntensity or image[testPos] == gapIntensity:
                    continue
                ThisInt = image[testPos]
                Intensity += ThisInt * fracValues[dataPos + k]
                totArea += fracValues[dataPos + k]
            if totArea == 0:
                continue
            Intensity /= totArea
            Int1D += Intensity
            n1ds += 1
        if n1ds == 0:
            continue
        Int1D /= n1ds
        result[i, 0] = RMean
        result[i, 1] = Int1D
    return result

# File paths for image and dark image
imageFN = 'test.tif'
darkFN = 'dark.png'
mapFN = 'Map.bin'
nMapFN = 'nMap.bin'

# Constants for bad and gap pixel intensities
badPxIntensity = -1
gapIntensity = -2

# Radial and angular binning parameters
RMax = 100
RMin = 10
RBinSize = 0.25
EtaMax = 180
EtaMin = -180
EtaBinSize = 1

# Load and preprocess the image
with Image.open(imageFN) as img:
    image = np.array(img)
if (os.path.exists(darkFN) == False):
    dark = np.zeros(image.shape)
else:
    with Image.open(darkFN) as img:
        dark = np.array(img)
nrPixelsY = image.shape[1]
image = np.clip(image - dark, 0, None)
image = image.astype(np.float32)

# Calculate the number of bins
nRBins = int(ceil((RMax - RMin) / RBinSize))
nEtaBins = int(ceil((EtaMax - EtaMin) / EtaBinSize))

# Load pixel list and nPixel list from binary files
pxList = np.fromfile(mapFN, dtype=np.int32)
if pxList.size % 4 != 0:
    raise ValueError("The total number of elements in pxList is not divisible by 4.")
pxList = pxList.reshape(-1, 4)
fracValues = np.zeros((pxList.shape[0]), dtype=np.float64)
for i in range(pxList.shape[0]):
    fracValues[i] = merge_int32_to_double(pxList[i, 2] , pxList[i, 3])
nPxList = np.fromfile(nMapFN, dtype=np.int32)

# Integrate the image and plot the results
result = integrateImage(image, pxList, nPxList, fracValues, nRBins, nEtaBins, RMin, RBinSize, badPxIntensity, gapIntensity, nrPixelsY)
plt.plot(result[:, 0], result[:, 1])
params, params_cov = fitFunc(result[:, 0], result[:, 1])
plt.plot(result[:, 0], func_voigt(result[:, 0], params[0], params[1], params[2], params[3], params[4]))
plt.show()