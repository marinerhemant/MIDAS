#!/usr/bin/env python

##### Save all the data arrays to an hdf5
##### python AutoCalibrateZarr.py -dataFN CeO2_30keV_210mm_20sec_000001.tif -ConvertFile 3 -paramFN ps_orig.txt -BadPxIntensity -2 -GapIntensity -1 -MakePlots 1 -StoppingStrain 0.003 -SavePlotsHDF plots.h5

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import os
import zarr
import subprocess
from skimage import measure
import matplotlib.patches as mpatches
plt.rcParams['figure.figsize'] = [10, 10]
import argparse
import sys
import plotly.graph_objects as go
import pandas as pd
import diplib as dip
from plotly.subplots import make_subplots
from PIL import Image
import math
import logging
from pathlib import Path
import traceback
import h5py
import io
import numba
from numba import jit
from scipy import ndimage
import multiprocessing as mp
from functools import partial

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("autocal.log")
    ]
)
logger = logging.getLogger(__name__)

# Get Python executable path
pytpath = sys.executable

# Global variables
h5_file = None
badGapArr = []

# Determine the installation path from the script's location
def get_install_path():
    """Determines the MIDAS installation path based on script location"""
    script_dir = Path(__file__).resolve().parent
    # Go one directory up from the script directory (utils)
    install_dir = script_dir.parent
    return install_dir

INSTALL_PATH = get_install_path()
env = dict(os.environ)  # No need to modify environment with cmake setup

class MyParser(argparse.ArgumentParser):
    """Custom argument parser with improved error handling"""
    def error(self, message):
        sys.stderr.write(f'Error: {message}\n')
        self.print_help()
        sys.exit(2)

def save_data_to_hdf5(data, h5file, dataset_name, metadata=None):
    """
    Save data arrays to an HDF5 file
    
    Parameters:
    -----------
    data : dict
        Dictionary containing the data arrays to save
    h5file : h5py.File
        Open HDF5 file object
    dataset_name : str
        Name for the dataset group in the HDF5 file
    metadata : dict, optional
        Dictionary containing metadata to save as attributes
    """
    # Create a group for this dataset
    group = h5file.create_group(dataset_name)
    
    # Save each array in the dictionary
    for key, value in data.items():
        group.create_dataset(key, data=value)
    
    # Save metadata as attributes if provided
    if metadata:
        for key, value in metadata.items():
            group.attrs[key] = value

# Functions for saving specific types of data
def save_raw_image_data(raw, h5file):
    """Save raw image data to HDF5"""
    data = {'raw_image': raw}
    metadata = {'description': 'Raw detector image'}
    save_data_to_hdf5(data, h5file, 'raw_image_data', metadata)

def save_background_data(background, h5file):
    """Save background image data to HDF5"""
    data = {'background': background}
    metadata = {'description': 'Computed background using median filter'}
    save_data_to_hdf5(data, h5file, 'background_data', metadata)

def save_threshold_data(thresh, h5file):
    """Save thresholded image data to HDF5"""
    data = {'threshold': thresh}
    metadata = {'description': 'Thresholded image after background subtraction'}
    save_data_to_hdf5(data, h5file, 'threshold_data', metadata)

def save_ring_data(thresh, bc_new, sim_rads, h5file):
    """Save data for ring overlay visualization"""
    data = {
        'threshold': thresh,
        'beam_center': np.array(bc_new),
        'ring_radii': np.array(sim_rads)
    }
    metadata = {
        'description': 'Data for ring overlay visualization',
        'beam_center_x': bc_new[1],
        'beam_center_y': bc_new[0]
    }
    save_data_to_hdf5(data, h5file, 'ring_overlay_data', metadata)

def save_strain_tth_data(unique_tth, mean_strains_per_ring, threshold, h5file, iter_nr):
    """Save strain vs 2-theta data for a specific iteration"""
    data = {
        'two_theta': unique_tth,
        'mean_strain': mean_strains_per_ring,
        'threshold': np.array([threshold])
    }
    metadata = {'iteration': iter_nr}
    save_data_to_hdf5(data, h5file, f'strain_tth_data_iter{iter_nr}', metadata)

def save_final_strain_data(results, unique_tth, mean_strains_per_ring, threshold, h5file):
    """Save final strain results data"""
    data = {
        'results_tth': results[:, -1],
        'results_strain': results[:, 1],
        'unique_tth': unique_tth,
        'mean_strain_per_ring': mean_strains_per_ring,
        'threshold': np.array([threshold])
    }
    metadata = {'description': 'Final strain results after convergence'}
    save_data_to_hdf5(data, h5file, 'final_strain_data', metadata)

def save_results_dataframe(df, h5file):
    """Save the results dataframe"""
    data_group = h5file.create_group('results_data')
    for column in df.columns:
        data_group.create_dataset(column, data=df[column].values)

def fileReader(f, dset):
    """Read data from Zarr file with handling for skipFrames"""
    global NrPixelsY, NrPixelsZ
    try:
        data = f[dset][:]
        data = data[skipFrame:,:,:]
        _, NrPixelsZ, NrPixelsY = data.shape
        data[data < 1] = 1
        return np.mean(data, axis=0).astype(np.uint16)
    except Exception as e:
        logger.error(f"Error reading dataset {dset}: {e}")
        raise

def generateZip(resFol, pfn, dfn='', darkfn='', dloc='', nchunks=-1, preproc=-1, outf='ZipOut.txt', errf='ZipErr.txt',NrPixelsY=0, NrPixelsZ=0):
    """Generate a Zarr zip file from other file formats"""
    cmd = [
        pytpath,
        os.path.join(INSTALL_PATH, 'utils/ffGenerateZipRefactor.py'),
        '-resultFolder', resFol,
        '-paramFN', pfn
    ]
    
    if dfn:
        cmd.extend(['-dataFN', dfn])
    if darkfn:
        cmd.extend(['-darkFN', darkfn])
    if dloc:
        cmd.extend(['-dataLoc', dloc])
    if nchunks != -1:
        cmd.extend(['-numFrameChunks', str(nchunks)])
    if preproc != -1:
        cmd.extend(['-preProcThresh', str(preproc)])
    if NrPixelsY != 0:
        cmd.extend(['-numPxY', str(NrPixelsY)])
    if NrPixelsZ != 0:
        cmd.extend(['-numPxZ', str(NrPixelsZ)])
    
    # Convert list to command string
    cmd_str = ' '.join(cmd)
    
    outfile_path = os.path.join(resFol, outf)
    errfile_path = os.path.join(resFol, errf)
    
    try:
        with open(outfile_path, 'w') as outfile, open(errfile_path, 'w') as errfile:
            return_code = subprocess.call(cmd_str, shell=True, stdout=outfile, stderr=errfile)
        
        # Check return code for non-zero value indicating error
        if return_code != 0:
            with open(errfile_path, 'r') as errfile:
                error_content = errfile.read()
            logger.error(f"Error executing generateZip - Return code: {return_code}")
            logger.error(f"Error content: {error_content}")
            print(f"Error executing generateZip - Return code: {return_code}")
            print(f"Error content: {error_content}")
            sys.exit(1)
        
        # Check output file
        with open(outfile_path, 'r') as outfile:
            lines = outfile.readlines()
            
            # Check for error messages in output
            for line in lines:
                if "Error" in line or "error" in line or "ERROR" in line:
                    logger.error(f"Error detected in generateZip output: {line.strip()}")
                    print(f"Error detected in generateZip output: {line.strip()}")
                    sys.exit(1)
            
            # Check if zip file was created
            if not lines or not any(line.startswith('OutputZipName') for line in lines):
                logger.error("No output zip file was generated")
                print("No output zip file was generated")
                sys.exit(1)
                
            # Return the zip file name
            for line in lines:
                if line.startswith('OutputZipName'):
                    return line.split()[1]
    
    except Exception as e:
        logger.error(f"Exception in generateZip: {e}")
        print(f"Exception in generateZip: {e}")
        sys.exit(1)
    
    # If we get here without finding the output zip name, that's an error
    logger.error("Could not find output zip name in generateZip output")
    print("Could not find output zip name in generateZip output")
    sys.exit(1)

def process_calibrant_output(output_file):
    """Process the output from CalibrantOMP"""
    global nPlanes, lsd_refined, bc_refined, ty_refined, tz_refined
    global p0_refined, p1_refined, p2_refined, p3_refined, mean_strain, std_strain
    
    try:
        with open(output_file) as f:
            output = f.readlines()
        
        useful = 0
        for line in output:
            if 'Number of planes being considered' in line and nPlanes == 0:
                nPlanes = int(line.rstrip().split()[-1][:-1])
            if useful == 1:
                if 'Copy to par' in line:
                    continue
                if 'Lsd ' in line:
                    lsd_refined = line.split()[1]
                if 'BC ' in line:
                    bc_refined = line.split()[1] + ' ' + line.split()[2]
                if 'ty ' in line:
                    ty_refined = line.split()[1]
                if 'tz ' in line:
                    tz_refined = line.split()[1]
                if 'p0 ' in line:
                    p0_refined = line.split()[1]
                if 'p1 ' in line:
                    p1_refined = line.split()[1]
                if 'p2 ' in line:
                    p2_refined = line.split()[1]
                if 'p3 ' in line:
                    p3_refined = line.split()[1]
                if 'MeanStrain ' in line:
                    mean_strain = line.split()[1]
                if 'StdStrain ' in line:
                    std_strain = line.split()[1]
            if 'Mean Values' in line:
                useful = 1
    except Exception as e:
        logger.error(f"Error processing calibrant output: {e}")

def process_calibrant_results(results_file):
    """Process the results from the calibration to find outlier rings"""
    try:
        global h5_file, iterNr  # Access the global h5_file
        
        results = np.genfromtxt(results_file, skip_header=1)
        unique_tth = np.unique(results[:, -1])
        mean_strains_per_ring = np.zeros(len(unique_tth))
        
        for ringNr in range(len(unique_tth)):
            subarr = results[results[:, -1] == unique_tth[ringNr], :]
            mean_strains_per_ring[ringNr] = np.mean(subarr[:, 1])
        
        threshold = multFactor * np.median(mean_strains_per_ring)
        ringsToExcludenew = np.argwhere(mean_strains_per_ring > threshold) + 1
        
        rNew = [ring[0] for ring in ringsToExcludenew]
        
        # Save data instead of plots
        if h5_file:
            save_strain_tth_data(unique_tth, mean_strains_per_ring, threshold, h5_file, iterNr)
                
        if DrawPlots == 1:
            fig = plt.figure()
            plt.scatter(unique_tth, mean_strains_per_ring)
            plt.axhline(threshold, color='black')
            plt.xlabel('2theta [degrees]')
            plt.ylabel('Average strain')
            plt.show()
        
        # Save final strain data if we've reached the convergence criteria
        if len(rNew) == 0 and float(mean_strain) < needed_strain:
            if h5_file:
                save_final_strain_data(results, unique_tth, mean_strains_per_ring, threshold, h5_file)
                
            if DrawPlots == 1:
                fig = plt.figure()
                plt.scatter(results[:, -1], results[:, 1])
                plt.scatter(unique_tth, mean_strains_per_ring, c='red')
                plt.axhline(threshold, color='black')
                plt.xlabel('2theta [degrees]')
                plt.ylabel('Computed strain')
                plt.title(f'Best fit results for {results_file.split(".corr")[0]}')
                plt.show()
        
        return rNew
    except Exception as e:
        logger.error(f"Error processing calibrant results: {e}")
        return []

def runMIDAS(fn):
    """Run MIDAS calibration and process results"""
    global ringsToExclude, folder, fstem, ext, ty_refined, tz_refined, p0_refined, p1_refined
    global p2_refined, p3_refined, darkName, fnumber, pad, lsd_refined, bc_refined, latc
    global nPlanes, mean_strain, std_strain, iterNr, h5_file
    
    ps_file = f"{fn}ps.txt"
    
    try:
        with open(ps_file, 'w') as pf:
            for ringNr in ringsToExclude:
                pf.write(f'RingsToExclude {ringNr}\n')
            pf.write(f'Folder {folder}\n')
            pf.write(f'FileStem {fstem}\n')
            pf.write(f'Ext {ext}\n')
            for transOpt in imTransOpt:
                pf.write(f'ImTransOpt {transOpt}\n')
            pf.write(f'Width {maxW}\n')
            pf.write('tolTilts 3\n')
            pf.write('tolBC 20\n')
            pf.write('tolLsd 15000\n')
            pf.write('tolP 2E-3\n')
            pf.write(f'tx {tx_reference}\n')
            pf.write(f'ty {ty_refined}\n')
            pf.write(f'tz {tz_refined}\n')
            pf.write('Wedge 0\n')
            pf.write(f'p0 {p0_refined}\n')
            pf.write(f'p1 {p1_refined}\n')
            pf.write(f'p2 {p2_refined}\n')
            pf.write(f'p3 {p3_refined}\n')
            pf.write(f'EtaBinSize {etaBinSize}\n')
            pf.write('HeadSize 0\n')
            if not math.isnan(badPxIntensity):
                pf.write(f'BadPxIntensity {badPxIntensity}\n')
            if not math.isnan(gapIntensity):
                pf.write(f'GapIntensity {gapIntensity}\n')
            pf.write(f'Dark {darkName}\n')
            pf.write(f'StartNr {fnumber}\n')
            pf.write(f'EndNr {fnumber}\n')
            pf.write(f'Padding {pad}\n')
            pf.write(f'NrPixelsY {NrPixelsY}\n')
            pf.write(f'NrPixelsZ {NrPixelsZ}\n')
            pf.write(f'px {px}\n')
            pf.write(f'Wavelength {Wavelength}\n')
            pf.write(f'SpaceGroup {space_group}\n')
            pf.write(f'Lsd {lsd_refined}\n')
            pf.write(f'RhoD {RhoDThis}\n')
            pf.write(f'BC {bc_refined}\n')
            pf.write(f'LatticeConstant {" ".join(map(str, latc))}\n')
        
        # Use the INSTALL_PATH to locate the CalibrantOMP executable
        calibrant_cmd = f"{os.path.join(INSTALL_PATH, 'FF_HEDM/bin/CalibrantOMP')} {ps_file} 10"
        with open('calibrant_screen_out.csv', 'w') as f:
            subprocess.call(calibrant_cmd, shell=True, env=env, stdout=f)
        
        # Process output
        process_calibrant_output('calibrant_screen_out.csv')
        
        # Process results file
        return process_calibrant_results(f"{fn}.corr.csv")
        
    except Exception as e:
        logger.error(f"Error running MIDAS: {traceback.format_exc()}")
        return []

def run_get_hkl_list(param_file):
    """Run GetHKLList with proper error handling"""
    try:
        cmd = f"{os.path.join(INSTALL_PATH, 'FF_HEDM/bin/GetHKLList')} {param_file}"
        with open('hkls_screen_out.csv', 'w') as f:
            subprocess.call(cmd, shell=True, env=env, stdout=f)
            
        return np.genfromtxt('hkls.csv', skip_header=1)
    except Exception as e:
        logger.error(f"Error running GetHKLList: {e}")
        raise

@jit(nopython=True)
def _compute_distances(coords, point):
    """JIT-compiled function to compute distances between points"""
    distances = np.zeros(len(coords))
    for i in range(len(coords)):
        diff = coords[i] - point
        distances[i] = np.sqrt(diff[0]**2 + diff[1]**2)
    return distances

@jit(nopython=True)
def _find_center_point(coords, bbox):
    """JIT-compiled function to find a center point"""
    # Find edge coordinates
    edge_indices = np.where(coords[:, 0] == bbox[0])[0]
    
    if len(edge_indices) == 0:
        return np.array([-1.0, -1.0])  # Invalid center
    
    # Select middle edge coordinate
    edgecoorda = coords[edge_indices[len(edge_indices)//2]]
    
    # Calculate distances
    distances = _compute_distances(coords, edgecoorda)
    
    # Find furthest point
    furthest_idx = np.argmax(distances)
    edgecoordb = coords[furthest_idx]
    max_distance = distances[furthest_idx]
    
    # Find points at approximately half the max distance
    arcLen = max_distance / 2
    candidate_indices = np.where(np.abs(distances - arcLen) < 2)[0]
    
    if len(candidate_indices) == 0:
        return np.array([-1.0, -1.0])  # Invalid center
    
    # Use the middle candidate
    candidatea = coords[candidate_indices[len(candidate_indices)//2]]
    candidateb = candidatea
    
    # Calculate midpoints
    midpointa = (edgecoorda + candidatea) / 2
    midpointb = (edgecoordb + candidateb) / 2
    
    # Extract coordinates
    x1, y1 = edgecoorda
    x2, y2 = candidatea
    x3, y3 = candidateb
    x4, y4 = edgecoordb
    x5, y5 = midpointa
    x6, y6 = midpointb
    
    # Check for division by zero
    if (y4 == y3 or y2 == y1):
        return np.array([-1.0, -1.0])  # Invalid center
    
    # Calculate slopes
    m1 = (x1 - x2) / (y2 - y1)
    m2 = (x3 - x4) / (y4 - y3)
    
    if abs(m1 - m2) < 1e-10:  # Avoid floating point issues
        return np.array([-1.0, -1.0])  # Invalid center
    
    # Calculate intersection
    x = (y6 - y5 + m1 * x5 - m2 * x6) / (m1 - m2)
    y = m1 * (x - x5) + y5
    
    return np.array([x, y])

def _process_single_label(label_info):
    """Process a single labeled region using JIT compilation"""
    label, mask, minArea = label_info
    
    # Check size
    if np.sum(mask) < minArea:
        return None
    
    # Get coordinates
    coords = np.array(np.where(mask)).T
    
    # Get bounding box
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
        
    bbox = (rows.min(), cols.min(), rows.max(), cols.max())
    
    # Find center using the JIT-compiled function
    center = _find_center_point(coords, bbox)
    
    # Check if center is valid
    if center[0] < 0 or center[1] < 0:
        return None
        
    logger.info(f"Detected center point: {center[1]}, {center[0]}")
    return center

def detect_beam_center_optimized(thresh, minArea, num_processes=None):
    """
    Detect beam center from thresholded image using both JIT and multiprocessing
    
    Parameters:
    -----------
    thresh : 2D numpy array
        Thresholded binary image
    minArea : int
        Minimum area of regions to consider
    num_processes : int, optional
        Number of processes to use. If None, uses all available cores.
    
    Returns:
    --------
    np.array
        [x, y] coordinates of the beam center, or [0, 0] if none found
    """
    try:
        # First compilation run to avoid startup delay during actual processing
        if not hasattr(detect_beam_center_optimized, "_initialized"):
            # Warm up the JIT compiler with a small dummy array
            dummy_coords = np.array([[0, 0], [1, 1], [2, 2]])
            dummy_bbox = (0, 0, 2, 2)
            _ = _find_center_point(dummy_coords, dummy_bbox)
            detect_beam_center_optimized._initialized = True
        
        # Use ndimage for labeling
        labels, nlabels = ndimage.label(thresh)
        
        if nlabels == 0:
            logger.warning("No beam centers detected!")
            return np.array([0.0, 0.0])
        
        # Prepare data for parallel processing
        label_data = []
        
        for label in range(1, nlabels + 1):
            # Create a mask for this label
            mask = (labels == label)
            
            # Add to processing list
            label_data.append((label, mask, minArea))
        
        # Determine number of processes
        if num_processes is None:
            num_processes = min(mp.cpu_count(), len(label_data))
            
        # Only use multiprocessing if we have enough work
        if len(label_data) > 1 and num_processes > 1:
            # Use multiprocessing to process all labels in parallel
            with mp.Pool(processes=num_processes) as pool:
                results = pool.map(_process_single_label, label_data)
            
            # Filter out None values
            all_centers = [r for r in results if r is not None]
        else:
            # Process sequentially for small workloads
            all_centers = []
            for data in label_data:
                result = _process_single_label(data)
                if result is not None:
                    all_centers.append(result)
        
        if not all_centers:
            logger.warning("No valid beam centers calculated!")
            return np.array([0.0, 0.0])
        
        # Calculate the median center
        centers_array = np.array(all_centers)
        bc_computed = np.array([np.median(centers_array[:, 0]), np.median(centers_array[:, 1])])
        
        return bc_computed
    except Exception as e:
        logger.error(f"Error detecting beam center: {e}")
        return np.array([0.0, 0.0])

def detect_ring_radii(labels, props, bc_computed, minArea):
    """Detect ring radii from labeled image and beam center"""
    try:
        rads = []
        nrads = 0
        nlabels = len(props) + 1
        
        for label in range(1, nlabels):
            if np.sum(labels == label) > minArea:
                coords = props[label-1].coords
                rad = np.mean(np.linalg.norm(np.transpose(coords) - bc_computed[:, None], axis=0))
                
                # Check if this radius is already in our list (within tolerance)
                toAdd = 1
                for radNr in range(nrads):
                    if np.abs(rads[radNr] - rad) < 20:
                        toAdd = 0
                        break
                        
                if toAdd == 1:
                    rads.append(rad)
                    nrads += 1
        
        if not rads:
            logger.warning("No rings detected!")
            return np.array([])
            
        return np.sort(np.array(rads))
    except Exception as e:
        logger.error(f"Error detecting ring radii: {e}")
        return np.array([])

def estimate_lsd(rads, sim_rads, sim_rad_ratios, firstRing, initialLsd):
    """Estimate sample-to-detector distance from ring radii"""
    if len(rads) == 0:
        return initialLsd
        
    try:
        radRatios = rads / rads[0]
        lsds = []
        
        for i in range(len(rads)):
            bestMatch = 10000
            bestRowNr = -1
            
            for j in range(firstRing - 1, len(sim_rads)):
                if np.abs(1 - (radRatios[i] / sim_rad_ratios[j])) < 0.02:
                    match_quality = np.abs(1 - (radRatios[i] / sim_rad_ratios[j]))
                    if match_quality < bestMatch:
                        bestMatch = match_quality
                        bestRowNr = j
            
            if bestRowNr != -1:
                lsds.append(initialLsd * rads[i] / sim_rads[bestRowNr])
        
        if not lsds:
            logger.warning("Could not estimate Lsd from rings - using initial guess")
            return initialLsd
            
        return np.median(lsds)
    except Exception as e:
        logger.error(f"Error estimating Lsd: {e}")
        return initialLsd

def create_param_file(output_file, params):
    """Create a parameter file with the given parameters"""
    try:
        with open(output_file, 'w') as pf:
            for key, value in params.items():
                if isinstance(value, list) or isinstance(value, np.ndarray):
                    pf.write(f"{key} {' '.join(map(str, value))}\n")
                else:
                    pf.write(f"{key} {value}\n")
    except Exception as e:
        logger.error(f"Error creating parameter file {output_file}: {e}")

def process_tiff_input(dataFN, badPxIntensity, gapIntensity):
    """Process TIFF input file and convert to GE format"""
    global NrPixelsY, NrPixelsZ, badGapArr
    
    try:
        img = Image.open(dataFN)
        logger.info("Data was a tiff image. Will convert to a ge file with 16 bit unsigned.")
        img = np.array(img)
        
        if img.dtype == np.int32:
            # Find bad or gap pixels
            if not np.isnan(badPxIntensity) and not np.isnan(gapIntensity):
                badGapArr = img == badPxIntensity
                badGapArr = np.logical_or(badGapArr, img == gapIntensity)
                
            # Scale data for 16-bit
            img = img.astype(np.double)
            img /= 2
            img = img.astype(np.uint16)
            
            # Set dimensions if not 2048x2048
            if img.shape[1] != 2048:
                NrPixelsY = img.shape[1]
            if img.shape[0] != 2048:
                NrPixelsZ = img.shape[0]
        
        # Save as GE file
        ge_file = f"{dataFN}.ge"
        with open(ge_file, 'wb') as f:
            f.write(b'\x00' * 8192)  # Header
            img.tofile(f)
            
        return ge_file
    except Exception as e:
        logger.error(f"Error processing TIFF input: {e}")
        sys.exit(1)

def main():
    """Main function to run the automated calibration"""
    global NrPixelsY, NrPixelsZ, skipFrame, space_group, px, latc, Wavelength
    global badPxIntensity, gapIntensity, DrawPlots, firstRing, multFactor, needed_strain
    global imTransOpt, etaBinSize, threshold, mrr, initialLsd, minArea, maxW
    global folder, fstem, ext, ty_refined, tz_refined, p0_refined, p1_refined
    global p2_refined, p3_refined, darkName, fnumber, pad, lsd_refined, bc_refined
    global ringsToExclude, nPlanes, mean_strain, std_strain, RhoDThis, h5_file, iterNr
    global badGapArr, tx_reference
    
    try:
        parser = MyParser(
            description='Automated Calibration for WAXS using continuous rings-like signal. This code takes either Zarr.Zip files or HDF5 files or GE binary files.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Add arguments
        parser.add_argument('-dataFN', type=str, required=True, help='DataFileName.zip, DataFileName.h5 or DataFileName.geX')
        parser.add_argument('-darkFN', type=str, required=False, default='', help='If separate file consists dark signal, provide this parameter.')
        parser.add_argument('-dataLoc', type=str, required=False, default='', help='If data is located in any location except /exchange/data in the hdf5 files, provide this.')
        parser.add_argument('-MakePlots', type=int, required=False, default=0, help='MakePlots: to draw, use 1.')
        parser.add_argument('-FirstRingNr', type=int, required=False, default=1, help='FirstRingNumber on data.')
        parser.add_argument('-EtaBinSize', type=float, required=False, default=5, help='EtaBinSize in degrees.')
        parser.add_argument('-MultFactor', type=float, required=False, default=2.5, help='If set, any ring MultFactor times average would be thrown away.')
        parser.add_argument('-Threshold', type=float, required=False, default=0, help='If you want to give a manual threshold, typically 500, otherwise, it will calculate automatically.')
        parser.add_argument('-StoppingStrain', type=float, required=False, default=0.00004, help='If refined pseudo-strain is below this value and all rings are "good", we would have converged.')
        parser.add_argument('-ImTransOpt', type=int, required=False, default=[0], nargs='*', help="If you want to do any transformations to the data: \n0: nothing, 1: flip LR, 2: flip UD, 3: transpose. Give as many as needed in the right order.")
        parser.add_argument('-ConvertFile', type=int, required=False, default=0, help="If you want to generate the zarr zip file from a different format: \n0: input is zarr zip file, 1: HDF5 input will be used to generarte a zarr zip file, 2: Binary GE-type file, 3: TIFF input.")
        parser.add_argument('-paramFN', type=str, required=False, default='', help="If you use convertFile != 0, you need to provide the parameter file consisting of all settings: SpaceGroup, SkipFrame, px, LatticeParameter, Wavelength and (optionally) tx in case tx is not 0.")
        parser.add_argument('-LsdGuess', type=float, required=False, default=1000000, help="If you know a guess for the Lsd, it might be good to kickstart things.")
        parser.add_argument('-BCGuess', type=float, required=False, default=[0.0, 0.0], nargs=2, help="If you know a guess for the BC, it might be good to kickstart things.")
        parser.add_argument('-BadPxIntensity', type=float, required=False, default=np.nan, help="If you know the bad pixel intensity, provide the value.")
        parser.add_argument('-GapIntensity', type=float, required=False, default=np.nan, help="If you know the gap intensity, provide the value. If you provide bad or gap, provide both!!!!")
        parser.add_argument('-SavePlotsHDF', type=str, required=False, default='', help="If provided, save all data arrays to this HDF5 file.")
        
        args, unparsed = parser.parse_known_args()
        
        # Initialize globals from arguments
        dataFN = args.dataFN
        darkFN = args.darkFN
        dataLoc = args.dataLoc
        LsdGuess = args.LsdGuess
        convertFile = args.ConvertFile
        badPxIntensity = args.BadPxIntensity
        gapIntensity = args.GapIntensity
        bcg = args.BCGuess
        DrawPlots = int(args.MakePlots)
        firstRing = int(args.FirstRingNr)
        multFactor = float(args.MultFactor)
        needed_strain = float(args.StoppingStrain)
        imTransOpt = args.ImTransOpt
        etaBinSize = args.EtaBinSize
        threshold = args.Threshold
        savePlotsHDF = args.SavePlotsHDF
        
        # Create HDF5 file for data arrays if requested
        if savePlotsHDF:
            logger.info(f"Will save data arrays to HDF5 file: {savePlotsHDF}")
            h5_file = h5py.File(savePlotsHDF, 'w')
            # Create a group for metadata
            meta_group = h5_file.create_group('metadata')
            meta_group.attrs['file_name'] = os.path.basename(dataFN)
            meta_group.attrs['date_created'] = pd.Timestamp.now().isoformat()
        
        # Set other constants
        NrPixelsY = 0
        NrPixelsZ = 0
        mrr = 2000000  # maximum radius to simulate rings
        initialLsd = LsdGuess  # Only used for simulation, real Lsd can be anything
        minArea = 300  # Minimum number of pixels that constitutes signal
        maxW = 1000  # Maximum width around the ideal ring
        
        logger.info(f"Starting automated calibration for: {dataFN}")
        
        # Process TIFF input if needed
        badGapArr = []
        if convertFile == 3:
            logger.info("Processing TIFF input")
            dataFN = process_tiff_input(dataFN, badPxIntensity, gapIntensity)
            logger.info(f"Converted TIFF to GE format: {dataFN} and NrPixelsY={NrPixelsY}, NrPixelsZ={NrPixelsZ}. Now converting to Zarr zip file.")
            psFN = args.paramFN
            if not psFN:
                logger.error("Parameter file is required for conversion")
                print("ERROR: Parameter file is required for conversion")
                sys.exit(1)
            dataFN = generateZip('.', psFN, dfn=dataFN, nchunks=100, preproc=0, NrPixelsY=NrPixelsY, NrPixelsZ=NrPixelsZ)
        
        # Generate Zarr zip file if needed
        if convertFile == 1 or convertFile == 2:
            psFN = args.paramFN
            if not psFN:
                logger.error("Parameter file is required for conversion")
                print("ERROR: Parameter file is required for conversion")
                sys.exit(1)
                
            logger.info("Generating zip file")
            dataFN = generateZip('.', psFN, dfn=dataFN, nchunks=100, preproc=0, darkfn=darkFN, dloc=dataLoc)
            
        # Read Zarr file
        logger.info(f"Reading Zarr file: {dataFN}")
        dataF = zarr.open(dataFN, mode='r')
        dataFN = os.path.basename(dataFN)
        
        # Extract parameters from Zarr
        skipFrame = 0
        space_group = dataF['/analysis/process/analysis_parameters/SpaceGroup'][0].item()
        if '/analysis/process/analysis_parameters/SkipFrame' in dataF:
            skipFrame = dataF['/analysis/process/analysis_parameters/SkipFrame'][0].item()
        px = dataF['/analysis/process/analysis_parameters/PixelSize'][0].item()
        latc = dataF['/analysis/process/analysis_parameters/LatticeParameter'][:]
        Wavelength = dataF['/analysis/process/analysis_parameters/Wavelength'][:].item()
        if '/analysis/process/analysis_parameters/tx' in dataF:
            tx_reference = dataF['/analysis/process/analysis_parameters/tx'][:].item()
        else:
            tx_reference = 0.0

        # Read data and dark
        raw = fileReader(dataF, '/exchange/data')
        dark = fileReader(dataF, '/exchange/dark')
        
        # Save as GE files
        rawFN = dataFN.split('.zip')[0] + '.ge5'
        darkFN = 'dark_' + rawFN
        raw.tofile(rawFN)
        if len(badGapArr) != 0:
            raw = np.ma.masked_array(raw, mask=badGapArr)
        dark.tofile(darkFN)
        darkName = darkFN
        
        # Apply image transformations
        for transOpt in imTransOpt:
            if transOpt == 1:
                raw = np.fliplr(raw)
                dark = np.fliplr(dark)
            elif transOpt == 2:
                raw = np.flipud(raw)
                dark = np.flipud(dark)
            elif transOpt == 3:
                raw = np.transpose(raw)
                dark = np.transpose(dark)
        
        # Save raw image data if needed
        if h5_file:
            save_raw_image_data(raw, h5_file)
            
        # Display raw image if needed
        if DrawPlots == 1:
            fig = plt.figure()
            plt.imshow(np.log(raw), clim=[np.median(np.log(raw)), np.median(np.log(raw)) + np.std(np.log(raw))], 
                      origin='lower')  # Set origin to lower left
            plt.colorbar()
            plt.title('Raw image')
            plt.show()
        
        # Create simulation parameter file
        logger.info("Running initial ring simulation")
        sim_params = {
            'Wavelength': Wavelength,
            'SpaceGroup': space_group,
            'Lsd': initialLsd,
            'MaxRingRad': mrr,
            'LatticeConstant': latc
        }
        create_param_file('ps_init_sim.txt', sim_params)
        
        # Run GetHKLList
        hkls = run_get_hkl_list('ps_init_sim.txt')
        sim_rads = np.unique(hkls[:, -1]) / px
        sim_rad_ratios = sim_rads / sim_rads[0]
        
        # Process image for calibration
        data = raw.astype(np.uint16)
        
        # Apply median filter for background estimation
        logger.info("Applying median filter for background estimation")
        data2 = dip.MedianFilter(data, 101)
        for _ in range(4):  # Apply multiple times for better background
            data2 = dip.MedianFilter(data2, 101)
            
        logger.info('Finished with median, now processing data.')
        data = data.astype(float)
        
        # Save background data if needed
        if h5_file:
            save_background_data(data2, h5_file)
            
        # Display background if needed
        if DrawPlots == 1:
            fig = plt.figure()
            plt.imshow(np.log(data2), origin='lower')  # Set origin to lower left
            plt.colorbar()
            plt.title('Computed background')
            plt.show()
        
        # Background subtraction and thresholding
        data_corr = data - data2
        if threshold == 0:
            threshold = 100 * (1 + np.std(data_corr) // 100)
        data_corr[data_corr < threshold] = 0
        thresh = data_corr.copy()
        thresh[thresh > 0] = 255
        
        # Save threshold data if needed
        if h5_file:
            save_threshold_data(thresh, h5_file)
            
        # Display threshold if needed
        if DrawPlots == 1:
            fig = plt.figure()
            plt.imshow(thresh, origin='lower')  # Set origin to lower left
            plt.colorbar()
            plt.title('Cleaned image')
            plt.show()
        
        # Detect beam center and ring radii
        if bcg[0] == 0:
            logger.info("Auto-detecting beam center")
            labels, nlabels = measure.label(thresh, return_num=True)
            props = measure.regionprops(labels)
            
            # Remove small regions
            for label in range(1, nlabels):
                if np.sum(labels == label) < minArea:
                    thresh[labels == label] = 0
            
            # Detect beam center
            bc_computed = detect_beam_center_optimized(thresh, minArea,6)
            
            # Detect ring radii
            rads = detect_ring_radii(labels, props, bc_computed, minArea)
            
            # Estimate Lsd
            if LsdGuess == 1000000:
                initialLsd = estimate_lsd(rads, sim_rads, sim_rad_ratios, firstRing, initialLsd)
            
        else:
            bc_computed = np.flip(np.array(bcg))
        
        bc_new = bc_computed
        logger.info(f"FN: {rawFN}, Beam Center guess: {np.flip(bc_new)}, Lsd guess: {initialLsd}")
        
        # Create parameter file for ring simulation
        sim_params = {
            'Wavelength': Wavelength,
            'SpaceGroup': space_group,
            'Lsd': initialLsd,
            'MaxRingRad': mrr,
            'LatticeConstant': latc
        }
        create_param_file('ps.txt', sim_params)
        
        # Run GetHKLList again with updated parameters
        hkls = run_get_hkl_list('ps.txt')
        sim_rads = np.unique(hkls[:, -1]) / px
        sim_rad_ratios = sim_rads / sim_rads[0]
        
        # Save ring overlay data if needed
        if h5_file:
            save_ring_data(thresh, bc_new, sim_rads, h5_file)
            
        # Display rings on image if needed
        if DrawPlots == 1:
            fig, ax = plt.subplots()
            plt.imshow(thresh, origin='lower')  # Set origin to lower left
            for rad in sim_rads:
                e1 = mpatches.Arc((bc_new[1], bc_new[0]), 2*rad, 2*rad, 
                                 angle=0, theta1=-180, theta2=180, color='blue')
                ax.add_patch(e1)
            ax.axis([0, NrPixelsY, 0, NrPixelsZ])
            ax.set_aspect('equal')
            plt.title('Overlaid rings')
            plt.show()
        
        # Prepare for MIDAS calibration
        fnumber = int(rawFN.split('_')[-1].split('.')[0])
        pad = len(rawFN.split('_')[-1].split('.')[0])
        fstem = os.path.basename('_'.join(rawFN.split('_')[:-1]))
        ext = '.' + '.'.join(rawFN.split('_')[-1].split('.')[1:])
        folder = os.path.dirname(rawFN)
        if not folder:
            folder = os.getcwd()
            
        lsd_refined = str(initialLsd)
        bc_refined = f"{bc_new[1]} {bc_new[0]}"
        ty_refined = '0'
        tz_refined = '0'
        p0_refined = '0'
        p1_refined = '0'
        p2_refined = '0'
        p3_refined = '0'
        ringsToExclude = []
        nPlanes = 0
        
        # Calculate RhoD - maximum radius to edge from beam center
        edges = np.array([[0, 0], [NrPixelsY, 0], [NrPixelsY, NrPixelsZ], [0, NrPixelsZ]])
        RhoDThis = np.max(np.linalg.norm(np.transpose(edges) - bc_new[:, None], axis=0)) * px
        
        # Run MIDAS calibration iterations
        iterNr = 1
        logger.info(f'Running MIDAS calibration, might take a few minutes. Trial Nr: {iterNr}')
        
        rOrig = runMIDAS(rawFN)
        iterNr += 1
        ringsToExclude = rOrig
        rNew = rOrig
        
        # Initialize ring exclusion list
        ringListExcluded = np.zeros(nPlanes + 1)
        for i in ringsToExclude:
            ringListExcluded[i] = 1
        
        # Iterative refinement
        while (len(rNew) > 0 or float(mean_strain) > needed_strain):
            logger.info(
                f'Running MIDAS calibration again with updated parameters. Trial Nr: {iterNr}\n'
                f'\tPrevious strain: {mean_strain}. Number of new rings to exclude: {len(rNew)}.'
                f' Number of rings to use: {int(nPlanes-1-sum(ringListExcluded))}'
            )
            
            rNew = runMIDAS(rawFN)
            iterNr += 1
            
            # Update ring exclusion list
            currentRingNr = 0
            for i in range(1, nPlanes + 1):
                if ringListExcluded[i] == 1:
                    continue
                else:
                    currentRingNr += 1
                if currentRingNr in rNew:
                    ringListExcluded[i] = 1
                    ringsToExclude.append(i)
        
        # Generate final results
        logger.info("Generating final results data")
        df = pd.read_csv(f"{rawFN}.corr.csv", delimiter=' ')
        
        # Generate interactive plot for HTML export
        if DrawPlots == 1:
            fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scatter"}, {"type": "scatterpolar"}]])
            
            fig.add_trace(
                go.Scatter(
                    mode='markers',
                    x=df['RadFit'],
                    y=df['Strain'],
                    marker=dict(color=df['Ideal2Theta']),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatterpolar(
                    r=df['Strain'],
                    theta=df['EtaCalc'],
                    mode='markers',
                    marker=dict(color=df['Ideal2Theta']),
                    showlegend=True
                ),
                row=1, col=2
            )
            
            # Save the interactive plot to HTML
            html_file = f"{rawFN}.html"
            fig.write_html(html_file)
            logger.info(f"Interactive plots written to: {html_file}")
        
        # Save results data to HDF5 if requested
        if h5_file:
            # Save the dataframe with all results data
            save_results_dataframe(df, h5_file)
                
            # Add parameters used for the run
            params_group = h5_file.create_group('parameters')
            for key, value in {
                'lsd': lsd_refined,
                'beam_center': bc_refined,
                'ty': ty_refined,
                'tz': tz_refined,
                'p0': p0_refined,
                'p1': p1_refined,
                'p2': p2_refined,
                'p3': p3_refined,
                'mean_strain': mean_strain,
                'std_strain': std_strain,
                'wavelength': Wavelength,
                'pixel_size': px,
                'space_group': space_group,
                'rings_excluded': ','.join(map(str, ringsToExclude))
            }.items():
                params_group.attrs[key] = value
                
            # Close the HDF5 file
            h5_file.close()
            logger.info(f"All data arrays saved to HDF5 file: {savePlotsHDF}")
        
        # Print final results
        logger.info("Converged to a good set of parameters.\nBest values:")
        logger.info(f'Lsd {lsd_refined}')
        logger.info(f'BC {bc_refined}')
        logger.info(f'tx {tx_reference}')
        logger.info(f'ty {ty_refined}')
        logger.info(f'tz {tz_refined}')
        logger.info(f'p0 {p0_refined}')
        logger.info(f'p1 {p1_refined}')
        logger.info(f'p2 {p2_refined}')
        logger.info(f'p3 {p3_refined}')
        logger.info(f'Mean Strain: {mean_strain}')
        logger.info(f'Std Strain: {std_strain}')
        
        # Write final parameters
        psName = 'refined_MIDAS_params.txt'
        logger.info(f"Writing final parameters to {psName}")
        
        final_params = {
            'Lsd': lsd_refined,
            'BC': bc_refined,
            'tx': tx_reference,
            'ty': ty_refined,
            'tz': tz_refined,
            'p0': p0_refined,
            'p1': p1_refined,
            'p2': p2_refined,
            'p3': p3_refined,
            'RhoD': RhoDThis,
            'Wavelength': Wavelength,
            'px': px,
            'RMin': 10,
            'RMax': 1000,
            'RBinSize': 1,
            'EtaMin': -180,
            'EtaMax': 180,
            'NrPixelsY': NrPixelsY,
            'NrPixelsZ': NrPixelsZ,
            'EtaBinSize': 5,
            'DoSmoothing': 1,
            'DoPeakFit': 1,
            'skipFrame': skipFrame,
            'MultiplePeaks': 1
        }
        
        with open(psName, 'w') as pf:
            for key, value in final_params.items():
                pf.write(f"{key} {value}\n")
            
            # Add transformation options
            for transOpt in imTransOpt:
                pf.write(f"ImTransOpt {transOpt}\n")
                
            # Add bad pixel handling if available
            if not math.isnan(badPxIntensity):
                pf.write(f"BadPxIntensity {badPxIntensity}\n")
            if not math.isnan(gapIntensity):
                pf.write(f"GapIntensity {gapIntensity}\n")
                
        logger.info("Calibration completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main function: {traceback.format_exc()}")
        if h5_file:
            h5_file.close()
        sys.exit(1)

if __name__ == "__main__":
    main()
