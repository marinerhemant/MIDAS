#!/usr/bin/env python

import subprocess
import numpy as np
import argparse
import warnings
import time
import os, sys, glob
from pathlib import Path
import shutil
import logging
from math import floor, isnan, fabs
import pandas as pd
from parsl.app.app import python_app
import parsl
# Add TOMO directory for midas_tomo_python import
_tomo_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'TOMO')
if _tomo_dir not in sys.path:
    sys.path.insert(0, _tomo_dir)
from midas_tomo_python import run_tomo_from_sinos
from PIL import Image
import h5py
import zarr
from numba import jit
import traceback
import re


def save_sinogram_variants(topdir, nGrs, maxNHKLs, nScans, grainSpots, omegas):
    """Save 4 sinogram TIF files per grain for each processing combination.

    Reads the 4 sinogram combinations saved by findSingleSolutionPFRefactored
    (raw, norm, abs, normabs) and saves each grain's sinogram as a
    separate TIF, preserving the raw double-precision intensity values.

    Output files in Sinos/:
        sino_raw_grNr_NNNN.tif       — raw intensity
        sino_norm_grNr_NNNN.tif      — normalized (I/Imax)
        sino_abs_grNr_NNNN.tif       — exp(-I)
        sino_normabs_grNr_NNNN.tif   — exp(-I/Imax)
    """
    combo_labels = ['raw', 'norm', 'abs', 'normabs']

    # Load all 4 combo arrays
    combo_data = {}
    for label in combo_labels:
        fns = glob.glob(os.path.join(topdir, f'sinos_{label}_*.bin'))
        if not fns:
            logging.getLogger('pf_midas').warning(
                f"Sinogram combo file sinos_{label}_*.bin not found, skipping variant saves.")
            return
        fn = fns[0]
        combo_data[label] = np.fromfile(fn, dtype=np.double,
                                        count=nGrs * maxNHKLs * nScans
                                        ).reshape((nGrs, maxNHKLs, nScans))

    os.makedirs(os.path.join(topdir, 'Sinos'), exist_ok=True)

    for grNr in range(nGrs):
        nSp = grainSpots[grNr]
        if nSp <= 0:
            continue
        grStr = str(grNr).zfill(4)
        for label in combo_labels:
            sino = np.transpose(combo_data[label][grNr, :nSp, :])  # (nScans, nSp)
            Image.fromarray(sino).save(
                os.path.join(topdir, f'Sinos/sino_{label}_grNr_{grStr}.tif'))

    logging.getLogger('pf_midas').info(
        f"Saved 4 sinogram variants for {nGrs} grains to Sinos/")

# Set paths dynamically using script location
def get_installation_dir():
    """Get the installation directory from the script's location."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to get to the installation directory
    install_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
    return install_dir

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MIDAS")
# Silence all Parsl loggers completely
logging.getLogger("parsl").setLevel(logging.CRITICAL)  # Only show critical errors
# Also silence these specific Parsl sub-loggers
for logger_name in ["parsl.dataflow.dflow", "parsl.dataflow.memoization", 
                    "parsl.process_loggers", "parsl.jobs.strategy",
                    "parsl.executors.threads"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

def check_and_exit_on_errors(error_files):
    """
    Check multiple error files and exit if any have content.
    
    Args:
        error_files: List of error file paths to check
    """
    def check_error_file(filename):
        """
        Check if an error file exists and has content.
        
        Args:
            filename: Path to the error file
            
        Returns:
            True if file exists and has content, False otherwise
        """
        if not os.path.exists(filename):
            return False
        
        with open(filename, 'r') as f:
            content = f.read().strip()
            
        if not content:
            return False
            
        # Check if content is just initialization or contains actual error
        if "I was able to do something" in content:
            return False
            
        return True
        
    for err_file in error_files:
        if check_error_file(err_file):
            logger.error(f"Error detected in {err_file}:")
            with open(err_file, 'r') as f:
                logger.error(f.read())
            sys.exit(1)
    logger.info("No errors detected in error files.")

@python_app
def parallel_peaks(layerNr, positions, startNrFirstLayer, nrFilesPerSweep, topdir,
                  paramContents, baseNameParamFN, ConvertFiles, nchunks, preproc,
                  midas_path, doPeakSearch, numProcs, startNr, endNr, Lsd, NormalizeIntensities,
                  omegaValues, minThresh, fStem, omegaFF, Ext, padding=6):
    """
    Run peak search in parallel for a specific layer.
    
    Args:
        Multiple parameters needed for peak searching
        midas_path: Path to MIDAS installation
        
    Returns:
        Success status message
    """
    import subprocess
    import numpy as np
    import time
    import os, sys
    from pathlib import Path
    import shutil
    from math import fabs
    import pandas as pd
    import zarr
    from numba import jit
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(f"MIDAS-Layer{layerNr}")
    
    # Import required modules using midas_path
    utils_dir = os.path.join(midas_path, 'utils')
    v7_dir = os.path.join(midas_path, 'FF_HEDM/v7')
    sys.path.insert(0, utils_dir)
    sys.path.insert(0, v7_dir)
    
    # Get Python executable
    pytpath = sys.executable
    
    def check_error_file(filename):
        """
        Check if an error file exists and has content.
        
        Args:
            filename: Path to the error file
            
        Returns:
            True if file exists and has content, False otherwise
        """
        if not os.path.exists(filename):
            return False
        
        with open(filename, 'r') as f:
            content = f.read().strip()
            
        if not content:
            return False
            
        # Check if content is just initialization or contains actual error
        if "I was able to do something" in content:
            return False
            
        return True
    
    def generate_zip(resFol, pfn, layerNr, midas_path, dfn='', dloc='', nchunks=-1, preproc=-1, outf='ZipOut.txt', errf='ZipErr.txt',numFilesPerScan=1):
        """
        Generate a zip file for the given parameters.
        
        Args:
            resFol: Result folder
            pfn: Parameter file name
            layerNr: Layer number
            midas_path: Path to MIDAS installation
            dfn: Data file name (optional)
            dloc: Data location (optional)
            nchunks: Number of frame chunks (optional)
            preproc: Pre-processing threshold (optional)
            outf: Output file name (optional)
            errf: Error file name (optional)
            numFilesPerScan: Number of files per scan (optional)
            
        Returns:
            The name of the generated zip file if successful
        """
        cmd = f"{pytpath} {os.path.join(midas_path, 'utils/ffGenerateZipRefactor.py')} -resultFolder {resFol} -paramFN {pfn} -LayerNr {str(layerNr)}"
        
        if dfn:
            cmd += f" -dataFN {dfn}"
        if dloc:
            cmd += f" -dataLoc {dloc}"
        if nchunks != -1:
            cmd += f" -numFrameChunks {str(nchunks)}"
        if preproc != -1:
            cmd += f" -preProcThresh {str(preproc)}"
        if numFilesPerScan > 1:
            cmd += f" -numFilesPerScan {str(numFilesPerScan)}"
            
        outf_path = f"{resFol}/midas_log/{outf}"
        errf_path = f"{resFol}/midas_log/{errf}"
        
        logger.info(f"Generating zip for layer {layerNr}: {cmd}")
        
        try:
            subprocess.call(cmd, shell=True, stdout=open(outf_path, 'w'), stderr=open(errf_path, 'w'))
            
            # Check for errors
            if check_error_file(errf_path):
                logger.error(f"Error in generate_zip for layer {layerNr}")
                with open(errf_path, 'r') as f:
                    logger.error(f.read())
                return None
                
            lines = open(outf_path, 'r').readlines()
            if lines and lines[-1].startswith('OutputZipName'):
                return lines[-1].split()[1]
                
            logger.warning(f"No output zip name found for layer {layerNr}")
            return None
        except Exception as e:
            logger.error(f"Exception in generate_zip for layer {layerNr}: {str(e)}")
            return None
    
    def CalcEtaAngleAll(y, z):
        alpha = rad2deg * np.arccos(z / np.linalg.norm(np.array([y, z]), axis=0))
        alpha[y > 0] *= -1
        return alpha
        
    rad2deg = 57.2957795130823
    deg2rad = 0.0174532925199433
    
    # Initialize error tracking
    error_files = []
    
    try:
        # Run peaksearch using nblocks 1 and blocknr 0
        logger.info(f'Processing LayerNr: {layerNr}')
        ypos = float(positions[layerNr - 1])
        thisStartNr = startNrFirstLayer + (layerNr - 1) * nrFilesPerSweep
        folderName = str(thisStartNr)
        thisDir = os.path.join(topdir, folderName)
        Path(thisDir).mkdir(parents=True, exist_ok=True)
        os.chdir(thisDir)
        
        # Create parameter file
        thisParamFN = os.path.join(thisDir, baseNameParamFN)
        with open(thisParamFN, 'w') as thisPF:
            for line in paramContents:
                thisPF.write(line)
                
                # Check for PanelShiftsFile and copy if exists
                if line.strip().startswith('PanelShiftsFile'):
                    parts = line.split()
                    if len(parts) >= 2:
                        psShiftFile = parts[1]
                        if not os.path.isabs(psShiftFile):
                            psShiftFile = f'{topdir}/{psShiftFile}'
                        if os.path.exists(psShiftFile):
                            shutil.copy2(psShiftFile, thisDir)
                            logger.info(f"Copied PanelShiftsFile {psShiftFile} to {thisDir}")
                        else:
                            logger.warning(f"PanelShiftsFile specified {psShiftFile} but does not exist.")
        
        # Create necessary directories
        Path(os.path.join(thisDir, 'Temp')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(thisDir, 'output')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(thisDir, 'midas_log')).mkdir(parents=True, exist_ok=True)
        sub_logDir = os.path.join(thisDir, 'output')
        
        # Generate zip or use existing file
        if ConvertFiles == 1:
            outFStem = generate_zip(thisDir, baseNameParamFN, layerNr, midas_path, nchunks=nchunks, preproc=preproc,numFilesPerScan=nrFilesPerSweep)
            if not outFStem:
                logger.error(f"Failed to generate zip for layer {layerNr}")
                return f"Failed at generateZip for layer {layerNr}"
        else:
            outFStem = f'{thisDir}/{fStem}_{str(thisStartNr).zfill(padding)}.MIDAS.zip'
        
        logger.info(f'Using FileStem: {outFStem}')
        
        # Process output files
        f_out_path = f'{sub_logDir}/processing_out0.csv'
        f_err_path = f'{sub_logDir}/processing_err0.csv'
        error_files.append(f_err_path)
        
        with open(f_out_path, 'w') as f, open(f_err_path, 'w') as f_err:
            # Get HKL list
            cmd = f"{os.path.join(midas_path, 'FF_HEDM/bin/GetHKLListZarr')} {outFStem} {thisDir}"
            logger.info(f"Running GetHKLListZarr: {cmd}")
            subprocess.call(cmd, shell=True, stdout=f, stderr=f_err)
            
            # Check for errors
            if check_error_file(f_err_path):
                with open(f_err_path, 'r') as ef:
                    logger.error(f"Error in GetHKLListZarr: {ef.read()}")
                return f"Failed at GetHKLListZarr for layer {layerNr}"
            
            # Do peak search if required
            if doPeakSearch == 1:
                t_st = time.time()
                logger.info(f'Starting PeakSearch for layer {layerNr}')
                cmd = f"{os.path.join(midas_path, 'FF_HEDM/bin/PeaksFittingOMPZarrRefactor')} {outFStem} 0 1 {numProcs} {thisDir}"
                subprocess.call(cmd, shell=True, stdout=f, stderr=f_err)
                
                if check_error_file(f_err_path):
                    with open(f_err_path, 'r') as ef:
                        logger.error(f"Error in PeaksFittingOMPZarrRefactor: {ef.read()}")
                    return f"Failed at PeakSearch for layer {layerNr}"
                    
                logger.info(f'PeakSearch Done for layer {layerNr}. Time taken: {time.time() - t_st} seconds.')
            
            # Merge overlapping peaks
            cmd = f"{os.path.join(midas_path, 'FF_HEDM/bin/MergeOverlappingPeaksAllZarr')} {outFStem} {thisDir}"
            logger.info(f"Running MergeOverlappingPeaksAllZarr: {cmd}")
            subprocess.call(cmd, shell=True, stdout=f, stderr=f_err)
            
            if check_error_file(f_err_path):
                with open(f_err_path, 'r') as ef:
                    logger.error(f"Error in MergeOverlappingPeaksAllZarr: {ef.read()}")
                return f"Failed at MergeOverlappingPeaksAllZarr for layer {layerNr}"
        
        # Process omega if needed
        store = zarr.storage.ZipStore(outFStem, mode='r')
        zf = zarr.open_group(store, mode='r')
        searchStr = 'measurement/process/scan_parameters/startOmeOverride'
        if searchStr in zf or len(omegaValues) > 0:
            if searchStr in zf:
                thisOmega = zf[searchStr][:][0]
            else:
                thisOmega = float(omegaValues[layerNr - 1])
                
            if thisOmega != 0:
                signTO = thisOmega / fabs(thisOmega)
            else:
                signTO = 1
                
            delOmega = signTO * (fabs(thisOmega) % 360) - omegaFF
            delOmega = delOmega * (fabs(delOmega) % 360) / fabs(delOmega)
            omegaOffsetThis = -delOmega  # Because we subtract this
            
            logger.info(f"Offsetting omega: {omegaOffsetThis}, original value: {thisOmega}.")
            
            tOme = time.time()
            
            result_file = f'Result_StartNr_{startNr}_EndNr_{endNr}.csv'
            if os.path.exists(result_file):
                shutil.copy2(result_file, f"{result_file}.old")
                
                try:
                    Result = np.genfromtxt(result_file, skip_header=1, delimiter=' ')
                    headRes = open(result_file).readline()
                    
                    if len(Result.shape) > 1:
                        Result = Result[Result[:, 5] > minThresh]
                        
                        if len(Result.shape) > 1:
                            Result[:, 2] -= omegaOffsetThis
                            
                            # Adjust values outside range
                            Result[Result[:, 2] < -180, 6] += 360
                            Result[Result[:, 2] < -180, 7] += 360
                            Result[Result[:, 2] < -180, 2] += 360
                            Result[Result[:, 2] > 180, 6] -= 360
                            Result[Result[:, 2] > 180, 7] -= 360
                            Result[Result[:, 2] > 180, 2] -= 360
                            
                            np.savetxt(result_file, Result, fmt="%.6f", delimiter=' ', 
                                       header=headRes.split('\n')[0], comments='')
                            
                    logger.info(f"Omega offset done for layer {layerNr}. Time taken: {time.time() - tOme} seconds. SpotsShape {Result.shape}")
                except Exception as e:
                    logger.error(f"Error processing omega for layer {layerNr}: {str(e)}")
                    return f"Failed at omega processing for layer {layerNr}"
        
        # Calculate radius and fit setup
        with open(f_out_path, 'a') as f, open(f_err_path, 'a') as f_err:
            cmd = f"{os.path.join(midas_path, 'FF_HEDM/bin/CalcRadiusAllZarr')} {outFStem} {thisDir}"
            logger.info(f"Running CalcRadiusAllZarr: {cmd}")
            subprocess.call(cmd, shell=True, stdout=f, stderr=f_err)
            
            if check_error_file(f_err_path):
                with open(f_err_path, 'r') as ef:
                    logger.error(f"Error in CalcRadiusAllZarr: {ef.read()}")
                return f"Failed at CalcRadiusAllZarr for layer {layerNr}"
            
            cmd = f"{os.path.join(midas_path, 'FF_HEDM/bin/FitSetupZarr')} {outFStem} {thisDir}"
            logger.info(f"Running FitSetupZarr: {cmd}")
            subprocess.call(cmd, shell=True, stdout=f, stderr=f_err)
            
            if check_error_file(f_err_path):
                with open(f_err_path, 'r') as ef:
                    logger.error(f"Error in FitSetupZarr: {ef.read()}")
                return f"Failed at FitSetupZarr for layer {layerNr}"
        
        # Process results and normalize intensities
        radius_file = f'Radius_StartNr_{startNr}_EndNr_{endNr}.csv'
        if not os.path.exists(radius_file):
            logger.error(f"Radius file not found for layer {layerNr}: {radius_file}")
            return f"Failed: Radius file not found for layer {layerNr}"
            
        Result = np.genfromtxt(radius_file, skip_header=1, delimiter=' ')
        
        if len(Result.shape) < 2:
            logger.warning(f"Result shape too small for layer {layerNr}, copying and continuing")
            shutil.copy2('InputAllExtraInfoFittingAll.csv', 
                         os.path.join(topdir, f'InputAllExtraInfoFittingAll{layerNr-1}.csv'))
            os.chdir(topdir)
            return f"Completed with small result shape for layer {layerNr}"
        
        try:
            # Read and process fitting data
            dfAllF = pd.read_csv('InputAllExtraInfoFittingAll.csv', delimiter=' ', skipinitialspace=True)
            dfAllF.loc[dfAllF['GrainRadius'] > 0.001, '%YLab'] += ypos
            dfAllF.loc[dfAllF['GrainRadius'] > 0.001, 'YOrig(NoWedgeCorr)'] += ypos
            dfAllF['Eta'] = CalcEtaAngleAll(dfAllF['%YLab'], dfAllF['ZLab'])
            dfAllF['Ttheta'] = rad2deg * np.arctan(np.linalg.norm(np.array([dfAllF['%YLab'], dfAllF['ZLab']]), axis=0) / Lsd)
            
            logger.info(f"Spots shape final for layer {layerNr}: {dfAllF.shape}")
            
            outFN2 = os.path.join(topdir, f'InputAllExtraInfoFittingAll{layerNr-1}.csv')
            t_st = time.time()
            
            # Handle different normalization modes
            if NormalizeIntensities == 0:
                dfAllF.to_csv(outFN2, sep=' ', header=True, float_format='%.6f', index=False)
                
            elif NormalizeIntensities == 1:
                uniqueRings, uniqueIndices = np.unique(Result[:, 13], return_index=True)
                ringPowderIntensity = []
                
                for iter in range(len(uniqueIndices)):
                    ringPowderIntensity.append([uniqueRings[iter], Result[uniqueIndices[iter], 16]])
                    
                ringPowderIntensity = np.array(ringPowderIntensity)
                
                for iter in range(len(ringPowderIntensity)):
                    ringNr = ringPowderIntensity[iter, 0]
                    powInt = ringPowderIntensity[iter, 1]
                    dfAllF.loc[dfAllF['RingNumber'] == ringNr, 'GrainRadius'] *= powInt**(1/3)
                    
                dfAllF.to_csv(outFN2, sep=' ', header=True, float_format='%.6f', index=False)
                
            elif NormalizeIntensities == 2:
                # Safely map the Spot ID to the Integrated Intensity from Radius.csv (Result)
                # Result[:, 0] is the Spot IDs, Result[:, 1] is the IntInt
                intensity_map = dict(zip(Result[:, 0], Result[:, 1]))
                
                # hashArr[:, 1] gets the original Spot IDs corresponding to the ordered dataframe
                hashArr = np.genfromtxt('IDRings.csv', skip_header=1)
                spot_ids = hashArr[:, 1]
                
                # Identify which rows are valid
                valid_mask = dfAllF['GrainRadius'] > 0.001
                
                # Apply the safe mapping list comprehension
                dfAllF.loc[valid_mask, 'GrainRadius'] = [intensity_map.get(sid, 0.0) for sid in spot_ids[valid_mask]]
                
                dfAllF.to_csv(outFN2, sep=' ', header=True, float_format='%.6f', index=False)
            
            elif NormalizeIntensities == 3:
                # Use total raw intensity for the 3D peak
                dfAllF['GrainRadius'] = dfAllF['RawSumIntensity']
                dfAllF.to_csv(outFN2, sep=' ', header=True, float_format='%.6f', index=False)
            
            # Copy necessary files
            shutil.copy2(os.path.join(thisDir, 'paramstest.txt'), os.path.join(topdir, 'paramstest.txt'))
            shutil.copy2(os.path.join(thisDir, 'hkls.csv'), os.path.join(topdir, 'hkls.csv'))
            
            logger.info(f'Normalization and writing done for layer {layerNr}. Time taken: {time.time() - t_st} seconds')
            
            os.chdir(topdir)
            return f"Successfully completed processing for layer {layerNr}"
            
        except Exception as e:
            logger.error(f"Error in final processing for layer {layerNr}: {str(e)}")
            logger.error(traceback.format_exc())
            os.chdir(topdir)
            return f"Failed at final processing for layer {layerNr}: {str(e)}"
            
    except Exception as e:
        logger.error(f"Exception in parallel_peaks for layer {layerNr}: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Failed with exception for layer {layerNr}: {str(e)}"

@python_app
def peaks(resultDir, zipFN, numProcs, midas_path, blockNr=0, numBlocks=1):
    """
    Run peak search on a specific block. Not used now!
    
    Args:
        resultDir: Directory for results
        zipFN: Zip file name
        numProcs: Number of processors to use
        midas_path: Path to MIDAS installation
        blockNr: Block number
        numBlocks: Total number of blocks
        
    Returns:
        Success status
    """
    import subprocess
    import os
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(f"MIDAS-Peaks-{blockNr}")
    
    def check_error_file(filename):
        """
        Check if an error file exists and has content.
        
        Args:
            filename: Path to the error file
            
        Returns:
            True if file exists and has content, False otherwise
        """
        if not os.path.exists(filename):
            return False
        
        with open(filename, 'r') as f:
            content = f.read().strip()
            
        if not content:
            return False
            
        # Check if content is just initialization or contains actual error
        if "I was able to do something" in content:
            return False
            
        return True
    
    # Create output files
    os.makedirs(f'{resultDir}/midas_log', exist_ok=True)
    f_out_path = f'{resultDir}/midas_log/peaksearch_out{blockNr}.csv'
    f_err_path = f'{resultDir}/midas_log/peaksearch_err{blockNr}.csv'
    
    try:
        with open(f_out_path, 'w') as f, open(f_err_path, 'w') as f_err:
            # Log initialization to error file to distinguish from real errors
            f_err.write("I was able to do something.\n")
            
            # Build command
            cmd_this = f"{os.path.join(midas_path, 'FF_HEDM/bin/PeaksFittingOMPZarrRefactor')} {zipFN} {blockNr} {numBlocks} {numProcs} {resultDir}"
            logger.info(f"Running PeaksFittingOMPZarrRefactor: {cmd_this}")
            f_err.write(f"{cmd_this}\n")
            
            # Run command
            subprocess.call(cmd_this, shell=True, stdout=f, stderr=f_err)
            
            # Verify completion
            f_err.write(f"{cmd_this}\n")
        
        # Check for errors
        with open(f_err_path, 'r') as f_err:
            content = f_err.read()
            if "Error" in content or "error" in content:
                logger.error(f"Error in peaks for block {blockNr}: {content}")
                return f"Failed for block {blockNr}"
        
        return f"Successfully completed peaks for block {blockNr}"
        
    except Exception as e:
        logger.error(f"Exception in peaks for block {blockNr}: {str(e)}")
        with open(f_err_path, 'a') as f_err:
            f_err.write(f"Exception: {str(e)}\n")
        return f"Failed with exception for block {blockNr}"

@python_app
def binData(resultDir, num_scans, midas_path):
    """
    Bin data for scanning.
    
    Args:
        resultDir: Directory for results
        num_scans: Number of scans
        midas_path: Path to MIDAS installation
        
    Returns:
        Status message
    """
    import subprocess
    import os
    import socket
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("MIDAS-Binning")
    
    def check_error_file(filename):
        """
        Check if an error file exists and has content.
        
        Args:
            filename: Path to the error file
            
        Returns:
            True if file exists and has content, False otherwise
        """
        if not os.path.exists(filename):
            return False
        
        with open(filename, 'r') as f:
            content = f.read().strip()
            
        if not content:
            return False
            
        # Check if content is just initialization or contains actual error
        if "I was able to do something" in content:
            return False
            
        return True
    
    os.chdir(resultDir)
    
    # Create output files
    os.makedirs(f'{resultDir}/midas_log', exist_ok=True)
    f_out_path = f'{resultDir}/midas_log/mapping_out.csv'
    f_err_path = f'{resultDir}/midas_log/mapping_err.csv'
    
    try:
        with open(f_out_path, 'w') as f, open(f_err_path, 'w') as f_err:
            # Build command
            cmd_this = f"{os.path.join(midas_path, 'FF_HEDM/bin/SaveBinDataScanning')} {num_scans}"
            logger.info(f"Running SaveBinDataScanning: {cmd_this}")
            # Log hostname for debugging
            subprocess.call(cmd_this, shell=True, stdout=f, stderr=f_err)
       
        # Check for errors
        with open(f_err_path, 'r') as f_err:
            content = f_err.read()
            if "Error" in content or "error" in content:
                logger.error(f"Error in binData: {content}")
                return "Failed to bin data"
        
        return "Successfully binned data"
        
    except Exception as e:
        logger.error(f"Exception in binData: {str(e)}")
        with open(f_err_path, 'a') as f_err:
            f_err.write(f"Exception: {str(e)}\n")
        return f"Failed with exception: {str(e)}"

@python_app
def indexscanning(resultDir, numProcs, num_scans, midas_path, blockNr=0, numBlocks=1):
    """
    Run indexing for scanning.
    
    Args:
        resultDir: Directory for results
        numProcs: Number of processors to use
        num_scans: Number of scans
        midas_path: Path to MIDAS installation
        blockNr: Block number
        numBlocks: Total number of blocks
        
    Returns:
        Status message
    """
    import subprocess
    import os
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(f"MIDAS-Indexing-{blockNr}")
    
    def check_error_file(filename):
        """
        Check if an error file exists and has content.
        
        Args:
            filename: Path to the error file
            
        Returns:
            True if file exists and has content, False otherwise
        """
        if not os.path.exists(filename):
            return False
        
        with open(filename, 'r') as f:
            content = f.read().strip()
            
        if not content:
            return False
            
        # Check if content is just initialization or contains actual error
        if "I was able to do something" in content:
            return False
            
        return True
    
    os.chdir(resultDir)
    
    # Create output files
    os.makedirs(f'{resultDir}/midas_log', exist_ok=True)
    f_out_path = f'{resultDir}/midas_log/indexing_out{blockNr}.csv'
    f_err_path = f'{resultDir}/midas_log/indexing_err{blockNr}.csv'
    
    try:
        with open(f_out_path, 'w') as f, open(f_err_path, 'w') as f_err:
            # Build command
            cmd_this = f"{os.path.join(midas_path, 'FF_HEDM/bin/IndexerScanningOMP')} paramstest.txt {blockNr} {numBlocks} {num_scans} {numProcs}"
            logger.info(f"Running IndexerScanningOMP: {cmd_this}")
            
            # Run command
            subprocess.call(cmd_this, shell=True, stdout=f, stderr=f_err)
        
        # Check for errors
        with open(f_err_path, 'r') as f_err:
            content = f_err.read()
            if "Error" in content or "error" in content:
                logger.error(f"Error in indexscanning for block {blockNr}: {content}")
                return f"Failed for block {blockNr}"
        
        return f"Successfully completed indexing for block {blockNr}"
        
    except Exception as e:
        logger.error(f"Exception in indexscanning for block {blockNr}: {str(e)}")
        with open(f_err_path, 'a') as f_err:
            f_err.write(f"Exception: {str(e)}\n")
        return f"Failed with exception for block {blockNr}"

@python_app
def refinescanning(resultDir, numProcs, midas_path, blockNr=0, numBlocks=1):
    """
    Run refinement for scanning.
    
    Args:
        resultDir: Directory for results
        numProcs: Number of processors to use
        midas_path: Path to MIDAS installation
        blockNr: Block number
        numBlocks: Total number of blocks
        
    Returns:
        Status message
    """
    import subprocess
    import os
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(f"MIDAS-Refining-{blockNr}")
    
    def check_error_file(filename):
        """
        Check if an error file exists and has content.
        
        Args:
            filename: Path to the error file
            
        Returns:
            True if file exists and has content, False otherwise
        """
        if not os.path.exists(filename):
            return False
        
        with open(filename, 'r') as f:
            content = f.read().strip()
            
        if not content:
            return False
            
        # Check if content is just initialization or contains actual error
        if "I was able to do something" in content:
            return False
            
        return True
    
    os.chdir(resultDir)
    
    # Create output files
    os.makedirs(f'{resultDir}/midas_log', exist_ok=True)
    f_out_path = f'{resultDir}/midas_log/refining_out{blockNr}.csv'
    f_err_path = f'{resultDir}/midas_log/refining_err{blockNr}.csv'
    
    try:
        # Count spots to index
        with open("SpotsToIndex.csv", "r") as f:
            num_lines = len(f.readlines())
        
        logger.info(f"Number of spots to refine: {num_lines}")
        
        # Build command
        cmd = f"{os.path.join(midas_path, 'FF_HEDM/bin/FitOrStrainsScanningOMP')} paramstest.txt {blockNr} {numBlocks} {num_lines} {numProcs}"
        logger.info(f"Running refining command: {cmd}")
        
        with open(f_out_path, 'w') as f, open(f_err_path, 'w') as f_err:
            # Run command
            subprocess.call(cmd, shell=True, cwd=resultDir, stdout=f, stderr=f_err)
        
        # Check for errors
        with open(f_err_path, 'r') as f_err:
            content = f_err.read()
            if "Error" in content or "error" in content:
                logger.error(f"Error in refinescanning for block {blockNr}: {content}")
                return f"Failed for block {blockNr}"
        
        return f"Successfully completed refinement for block {blockNr}"
        
    except Exception as e:
        logger.error(f"Exception in refinescanning for block {blockNr}: {str(e)}")
        with open(f_err_path, 'a') as f_err:
            f_err.write(f"Exception: {str(e)}\n")
        return f"Failed with exception for block {blockNr}"

class MyParser(argparse.ArgumentParser):
    """Custom argument parser that prints help on error."""
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

def main():
    """Main function to run the MIDAS processing workflow."""
    # Define constants that might be needed
    rad2deg = 57.2957795130823
    deg2rad = 0.0174532925199433
    
    # Define local check_error_file function
    def check_error_file(filename):
        """
        Check if an error file exists and has content.
        
        Args:
            filename: Path to the error file
            
        Returns:
            True if file exists and has content, False otherwise
        """
        if not os.path.exists(filename):
            return False
        
        with open(filename, 'r') as f:
            content = f.read().strip()
            
        if not content:
            return False
            
        # Check if content is just initialization or contains actual error
        if "I was able to do something" in content:
            return False
            
        return True

    startTime = time.time()
    
    # Parse command line arguments
    parser = MyParser(description='''
    PF_MIDAS, contact hsharma@anl.gov 
    Provide positions.csv file (negative positions with respect to actual motor position, 
                    motor position is normally position of the rotation axis, opposite to the voxel position).
    Parameter file and positions.csv file must be in the same folder.
    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-nCPUs', type=int, required=False, default=32, help='Number of CPUs to use')
    parser.add_argument('-nCPUsLocal', type=int, required=False, default=4, help='Local Number of CPUs to use')
    parser.add_argument('-paramFile', type=str, required=True, help='ParameterFileName: Do not use the full path.')
    parser.add_argument('-micFN', type=str, default="", help='Path to mic file')
    parser.add_argument('-grainsFN', type=str, default="", help='Path to grains file')
    parser.add_argument('-nNodes', type=int, required=False, default=1, help='Number of Nodes')
    parser.add_argument('-machineName', type=str, required=False, default='local', help='Machine Name: local,orthrosall,orthrosnew,umich')
    parser.add_argument('-omegaFile', type=str, required=False, default='', help='If you want to override omegas')
    parser.add_argument('-doPeakSearch', type=int, required=False, default=1, help='0 if PeakSearch is already done. InputAllExtra...0..n.csv should exist in the folder. -1 if you want to reprocess the peaksearch output, without doing peaksearch again.')
    parser.add_argument('-oneSolPerVox', type=int, required=False, default=1, help='0 if want to allow multiple solutions per voxel. 1 if want to have only 1 solution per voxel.')
    parser.add_argument('-resultDir', type=str, required=False, default='', help='Directory where you want to save the results. If omitted, the current directory will be used.')
    parser.add_argument('-numFrameChunks', type=int, required=False, default=-1, help='If low on RAM, it can process parts of the dataset at the time. -1 will disable.')
    parser.add_argument('-preProcThresh', type=int, required=False, default=-1, help='If want to save the dark corrected data, then put to whatever threshold wanted above dark. -1 will disable. 0 will just subtract dark. Negative values will be reset to 0.')
    parser.add_argument('-doTomo', type=int, required=False, default=1, help='If want to do tomography, put to 1. Only for OneSolPerVox.')
    parser.add_argument('-normalizeIntensities', type=int, required=False, default=2, help='Normalization mode for intensity in sinograms: 0=equivalent grain size, 1=powder-scaled, 2=integrated intensity (default), 3=raw sum intensity from PeaksFitting.')
    parser.add_argument('-convertFiles', type=int, required=False, default=1, help='If want to convert to zarr, if zarr files exist already, put to 0.')
    parser.add_argument('-runIndexing', type=int, required=False, default=1, help='If want to skip Indexing, put to 0.')
    parser.add_argument('-startScanNr', type=int, required=False, default=1, help='If you want to do partial peaksearch. Default: 1')
    parser.add_argument('-minThresh', type=int, required=False, default=-1, help='If you want to filter out peaks with intensity less than this number. -1 disables this. This is only used for filtering out peaksearch results for small peaks, peaks with maxInt smaller than this will be filtered out.')
    parser.add_argument('-sinoType', type=str, required=False, default='raw', choices=['raw', 'norm', 'abs', 'normabs'], help='Sinogram type to use for reconstruction (raw, norm, abs, normabs). Default: raw')
    # Parse arguments
    args, unparsed = parser.parse_known_args()
    
    # Extract arguments
    baseNameParamFN = args.paramFile
    machineName = args.machineName
    omegaFile = args.omegaFile
    doPeakSearch = args.doPeakSearch
    oneSolPerVox = args.oneSolPerVox
    numProcs = args.nCPUs
    numProcsLocal = args.nCPUsLocal
    nNodes = args.nNodes
    topdir = args.resultDir
    nchunks = args.numFrameChunks
    preproc = args.preProcThresh
    doTomo = args.doTomo
    ConvertFiles = args.convertFiles
    runIndexing = args.runIndexing
    NormalizeIntensities = args.normalizeIntensities
    startScanNr = args.startScanNr
    minThresh = args.minThresh
    micFN = args.micFN
    grainsFN = args.grainsFN
    sinoType = args.sinoType
    # Use current directory if no result directory specified
    if not topdir:
        topdir = os.getcwd()
    
    logger.info(f'Working directory: {topdir}')
    logDir = os.path.join(topdir, 'output')
    
    # Create directories
    os.makedirs(topdir, exist_ok=True)
    os.makedirs(logDir, exist_ok=True)
    
    # Get MIDAS installation directory dynamically
    midas_path = get_installation_dir()
    logger.info(f"Using MIDAS installation directory: {midas_path}")
    
    # Import required modules using midas_path
    utils_dir = os.path.join(midas_path, 'utils')
    v7_dir = os.path.join(midas_path, 'FF_HEDM/v7')
    sys.path.insert(0, utils_dir)
    sys.path.insert(0, v7_dir)
    
    # Import from MIDAS libraries
    try:
        from calcMiso import MakeSymmetries, GetMisOrientationAngleOM, OrientMat2Euler, OrientMat2Quat, BringDownToFundamentalRegionSym
    except ImportError:
        logger.error(f"Failed to import calcMiso. Make sure MIDAS is properly installed at {midas_path}")
        sys.exit(1)
    
    # Load appropriate machine configuration
    from parsl.config import Config
    from parsl.executors import ThreadPoolExecutor
    
    # Default configuration that works everywhere
    default_config = Config(
        executors=[ThreadPoolExecutor(max_threads=numProcsLocal)],
        retries=2
    )
    
    try:
        if machineName == 'local':
            nNodes = 1
            try:
                # Use import module instead of import *
                import localConfig
                parsl.load(config=localConfig.localConfig)
                logger.info("Loaded local configuration")
            except ImportError:
                logger.warning("Could not import localConfig, using default configuration")
                parsl.load(config=default_config)
        elif machineName == 'orthrosnew':
            os.environ['MIDAS_SCRIPT_DIR'] = logDir
            nNodes = 11
            numProcs = 32
            try:
                # Use import module instead of import *
                import orthrosAllConfig
                parsl.load(config=orthrosAllConfig.orthrosNewConfig)
                logger.info("Loaded orthrosnew configuration")
            except ImportError:
                logger.error("Could not import orthrosAllConfig for orthrosnew machine")
                logger.error("Using default configuration instead")
                parsl.load(config=default_config)
        elif machineName == 'orthrosall':
            os.environ['MIDAS_SCRIPT_DIR'] = logDir
            nNodes = 5
            numProcs = 64
            try:
                # Use import module instead of import *
                import orthrosAllConfig
                parsl.load(config=orthrosAllConfig.orthrosAllConfig)
                logger.info("Loaded orthrosall configuration")
            except ImportError:
                logger.error("Could not import orthrosAllConfig for orthrosall machine")
                logger.error("Using default configuration instead")
                parsl.load(config=default_config)
        elif machineName == 'umich':
            os.environ['MIDAS_SCRIPT_DIR'] = logDir
            os.environ['nNodes'] = str(nNodes)
            numProcs = 36
            try:
                # Use import module instead of import *
                import uMichConfig
                parsl.load(config=uMichConfig.uMichConfig)
                logger.info("Loaded umich configuration")
            except ImportError:
                logger.error("Could not import uMichConfig")
                logger.error("Using default configuration instead")
                parsl.load(config=default_config)
        elif machineName == 'marquette':
            os.environ['MIDAS_SCRIPT_DIR'] = logDir
            os.environ['nNodes'] = str(nNodes)
            try:
                # Use import module instead of import *
                import marquetteConfig
                parsl.load(config=marquetteConfig.marquetteConfig)
                logger.info("Loaded marquette configuration")
            except ImportError:
                logger.error("Could not import marquetteConfig")
                logger.error("Using default configuration instead")
                parsl.load(config=default_config)
        else:
            logger.warning(f"Unknown machine name '{machineName}', using default configuration")
            nNodes = 1
            parsl.load(config=default_config)
    except Exception as e:
        logger.error(f"Error loading Parsl configuration: {str(e)}")
        logger.error("Falling back to default configuration")
        nNodes = 1
        parsl.load(config=default_config)
    
    # Process parameter file
    try:
        # Read parameter file
        with open(baseNameParamFN, 'r') as f:
            paramContents = f.readlines()
        
        # Initialize variables
        RingNrs = []
        nMerges = 0
        maxang = 1
        tol_ome = 1
        tol_eta = 1
        omegaFN = ''
        omegaFF = -1
        padding = 6
        
        # Parse parameters
        for line in paramContents:
            if line.startswith('StartFileNrFirstLayer'):
                startNrFirstLayer = int(line.split()[1])
            elif line.startswith('MaxAng'):
                maxang = float(line.split()[1])
            elif line.startswith('TolEta'):
                tol_eta = float(line.split()[1])
            elif line.startswith('TolOme'):
                tol_ome = float(line.split()[1])
            elif line.startswith('NrFilesPerSweep'):
                nrFilesPerSweep = int(line.split()[1])
            elif line.startswith('MicFile'):
                micFN = line.split()[1]
            elif line.startswith('GrainsFile'):
                grainsFN = line.split()[1]
            elif line.startswith('FileStem'):
                fStem = line.split()[1]
            elif line.startswith('Ext'):
                Ext = line.split()[1]
            elif line.startswith('StartNr'):
                startNr = int(line.split()[1])
            elif line.startswith('EndNr'):
                endNr = int(line.split()[1])
            elif line.startswith('SpaceGroup'):
                sgnum = int(line.split()[1])
            elif line.startswith('nStepsToMerge'):
                nMerges = int(line.split()[1])
            elif line.startswith('nScans'):
                nScans = int(line.split()[1])
            elif line.startswith('Lsd'):
                Lsd = float(line.split()[1])
            elif line.startswith('OverAllRingToIndex'):
                RingToIndex = int(line.split()[1])
            elif line.startswith('BeamSize'):
                BeamSize = float(line.split()[1])
            elif line.startswith('OmegaStep'):
                omegaStep = float(line.split()[1])
            elif line.startswith('OmegaFirstFile'):
                omegaFF = float(line.split()[1])
            elif line.startswith('px'):
                px = float(line.split()[1])
            elif line.startswith('RingThresh'):
                RingNrs.append(int(line.split()[1]))
            elif line.startswith('Padding'):
                padding = int(line.split()[1])
        
        # Call GetHKLList to generate hkls.csv
        cmd = f"{os.path.join(midas_path, 'FF_HEDM/bin/GetHKLList')} {baseNameParamFN}"
        logger.info(f"Running GetHKLList: {cmd}")
        subprocess.call(cmd, shell=True)
        
        # Check for hkls.csv
        if not os.path.exists('hkls.csv'):
            logger.error("Failed to generate hkls.csv")
            sys.exit(1)
        
        # Process HKL list
        hkls = np.genfromtxt('hkls.csv', skip_header=1)
        _, idx = np.unique(hkls[:, 4], return_index=True)
        hkls = hkls[idx, :]
        rads = [hkl[-1] for rnr in RingNrs for hkl in hkls if hkl[4] == rnr]
        
        logger.info(f"Ring numbers: {RingNrs}")
        logger.info(f"Ring radii: {rads}")
        
        # Handle merges
        if nMerges != 0:
            os.chdir(topdir)
            if os.path.exists('original_positions.csv'):
                shutil.move('original_positions.csv', 'positions.csv')
        
        # Read positions
        with open(os.path.join(topdir, 'positions.csv'), 'r') as f:
            positions = f.readlines()
        
        # Read omega values if provided
        omegaValues = []
        if omegaFile:
            omegaValues = np.genfromtxt(omegaFile)
        
        # Run peak search if requested
        if doPeakSearch == 1 or doPeakSearch == -1:
            logger.info(f"Starting peak search for {nScans} scans starting from {startScanNr}")
            
            # Use parsl to run in parallel
            res = []
            for layerNr in range(startScanNr, nScans + 1):
                res.append(parallel_peaks(
                    layerNr, positions, startNrFirstLayer, nrFilesPerSweep, topdir,
                    paramContents, baseNameParamFN, ConvertFiles, nchunks, preproc,
                    midas_path, doPeakSearch, numProcs, startNr, endNr, Lsd, NormalizeIntensities,
                    omegaValues, minThresh, fStem, omegaFF, Ext, padding
                ))
            
            # Wait for all tasks to complete
            outputs = [i.result() for i in res]
            
            # Check for errors in outputs
            for i, output in enumerate(outputs):
                if output and "Failed" in output:
                    logger.error(f"Error in peak search for layer {startScanNr + i}: {output}")
                    
                    # Check error files for this layer
                    layerNr = startScanNr + i
                    thisStartNr = startNrFirstLayer + (layerNr - 1) * nrFilesPerSweep
                    folderName = str(thisStartNr)
                    thisDir = os.path.join(topdir, folderName)
                    err_file = os.path.join(thisDir, 'output', 'processing_err0.csv')
                    
                    if os.path.exists(err_file):
                        with open(err_file, 'r') as f:
                            logger.error(f"Error file content: {f.read()}")
                    
                    sys.exit(1)
            
            logger.info(f'Peak search completed on {nNodes} nodes.')
        else:
            if nMerges != 0:
                for layerNr in range(0, nMerges * (nScans // nMerges)):
                    if os.path.exists(f'original_InputAllExtraInfoFittingAll{layerNr}.csv'):
                        shutil.move(
                            f'original_InputAllExtraInfoFittingAll{layerNr}.csv',
                            f'InputAllExtraInfoFittingAll{layerNr}.csv'
                        )
        
        # Handle merges
        if nMerges != 0:
            logger.info(f"Merging {nMerges} scans")
            
            os.chdir(topdir)
            shutil.move('positions.csv', 'original_positions.csv')
            
            for layerNr in range(0, nMerges * (nScans // nMerges)):
                if os.path.exists(f'InputAllExtraInfoFittingAll{layerNr}.csv'):
                    shutil.move(
                        f'InputAllExtraInfoFittingAll{layerNr}.csv',
                        f'original_InputAllExtraInfoFittingAll{layerNr}.csv'
                    )
            
            # Run merge command
            merge_cmd = f"{os.path.join(midas_path, 'FF_HEDM/bin/mergeScansScanning')} {nMerges*(nScans//nMerges)} {nMerges} {2*px} {2*omegaStep} {numProcsLocal}"
            logger.info(f"Running merge: {merge_cmd}")
            subprocess.call(merge_cmd, shell=True)
            
            # Read new positions after merge
            with open(os.path.join(topdir, 'positions.csv'), 'r') as f:
                positions = f.readlines()
                
            nScans = int(floor(nScans / nMerges))
            BeamSize *= nMerges
        
        # Prepare for indexing and refinement
        os.chdir(topdir)
        Path(os.path.join(topdir, 'Output')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(topdir, 'Results')).mkdir(parents=True, exist_ok=True)
        
        # Update parameters file
        with open('paramstest.txt', 'r') as paramsf:
            lines = paramsf.readlines()
            
        with open('paramstest.txt', 'w') as paramsf:
            for line in lines:
                if any(line.startswith(x) for x in ['RingNumbers', 'MarginRadius', 'RingRadii', 'RingToIndex', 'BeamSize', 'px']):
                    continue
                if line.startswith('MicFile') and not micFN:
                    continue
                if line.startswith('GrainsFile') and not grainsFN:
                    continue
                if line.startswith('OutputFolder'):
                    paramsf.write(f'OutputFolder {topdir}/Output\n')
                elif line.startswith('ResultFolder'):
                    paramsf.write(f'ResultFolder {topdir}/Results\n')
                else:
                    paramsf.write(line)
            
            # Add ring numbers and radii
            for idx in range(len(RingNrs)):
                paramsf.write(f'RingNumbers {RingNrs[idx]}\n')
                paramsf.write(f'RingRadii {rads[idx]}\n')
                
            paramsf.write(f'BeamSize {BeamSize}\n')
            paramsf.write('MarginRadius 10000000;\n')
            paramsf.write(f'px {px}\n')
            paramsf.write(f'RingToIndex {RingToIndex}\n')
            
            if args.micFN:
                paramsf.write(f'MicFile {args.micFN}\n')
                micFN = args.micFN
            if args.grainsFN:
                paramsf.write(f'GrainsFile {args.grainsFN}\n')
                print("Added grains file to parameters file: ", args.grainsFN)
                grainsFN = args.grainsFN
        
        # Run indexing if requested
        if runIndexing == 1:
            logger.info("Starting data binning")
            bin_result = binData(topdir, nScans, midas_path).result()
            logger.info(f"Binning result: {bin_result}")
            
            # Check for errors in binning
            bin_err_file = os.path.join(topdir, 'output', 'mapping_err.csv')
            if check_error_file(bin_err_file):
                logger.error("Error in data binning")
                with open(bin_err_file, 'r') as f:
                    logger.error(f.read())
                sys.exit(1)
            
            logger.info("Data binning finished. Running indexing.")
            
            # Run indexing in parallel
            resIndex = []
            for nodeNr in range(nNodes):
                resIndex.append(indexscanning(topdir, numProcs, nScans, midas_path, blockNr=nodeNr, numBlocks=nNodes))
                
            # Wait for all indexing tasks to complete
            outputIndex = [i.result() for i in resIndex]
            
            # Check for errors in indexing
            for i, output in enumerate(outputIndex):
                if "Failed" in output:
                    logger.error(f"Error in indexing for node {i}: {output}")
                    
                    # Check error file
                    err_file = os.path.join(topdir, 'output', f'indexing_err{i}.csv')
                    if os.path.exists(err_file):
                        with open(err_file, 'r') as f:
                            logger.error(f"Error file content: {f.read()}")
                    
                    sys.exit(1)
        
        # Handle single solution per voxel
        if oneSolPerVox == 1:
            # Prepare for tomography if requested
            if doTomo == 1:
                # Remove existing sinos
                for pattern in ["sinos_*.bin", "omegas_*.bin", "nrHKLs_*.bin"]:
                    sinoFNs = glob.glob(pattern)
                    for sinoF in sinoFNs:
                        os.remove(sinoF)
                
                # Remove existing directories
                for dirn in ['Sinos', 'Recons', 'Thetas']:
                    if os.path.isdir(dirn):
                        shutil.rmtree(dirn)
                
                # Move result directories if they exist
                for dirn in ['fullResults', 'fullOutput']:
                    if os.path.isdir(dirn):
                        if os.path.isdir(dirn[4:]):
                            shutil.rmtree(dirn[4:])
                        shutil.move(dirn, dirn[4:])
            
            # Run find single solution
            cmd = f"{os.path.join(midas_path, 'FF_HEDM/bin/findSingleSolutionPFRefactored')} {topdir} {sgnum} {maxang} {nScans} {numProcsLocal} {tol_ome} {tol_eta} {baseNameParamFN} {NormalizeIntensities} 1"
            logger.info(f"Running findSingleSolutionPFRefactored: {cmd}")
            result = subprocess.call(cmd, cwd=topdir, shell=True)
            
            if result != 0:
                logger.error("Error in findSingleSolutionPFRefactored")
                sys.exit(1)
                
            os.makedirs('Recons', exist_ok=True)
            
            # Run tomography if requested
            if doTomo == 1:
                # Find sino file
                sinoFNs = glob.glob("sinos_*.bin")
                if not sinoFNs:
                    logger.error("No sino files found")
                    sys.exit(1)
                    
                sinoFN = sinoFNs[0]
                nGrs = int(sinoFN.split('_')[1])
                maxNHKLs = int(sinoFN.split('_')[2])
                
                # Read sino data
                try:
                    Sinos = np.fromfile(sinoFN, dtype=np.double, count=nGrs*maxNHKLs*nScans).reshape((nGrs, maxNHKLs, nScans))
                    omegas = np.fromfile(f"omegas_{nGrs}_{maxNHKLs}.bin", dtype=np.double, count=nGrs*maxNHKLs).reshape((nGrs, maxNHKLs))
                    grainSpots = np.fromfile(f"nrHKLs_{nGrs}.bin", dtype=np.int32, count=nGrs)
                except Exception as e:
                    logger.error(f"Error reading sino data: {str(e)}")
                    sys.exit(1)
                
                # Create directories
                os.makedirs('Sinos', exist_ok=True)
                os.makedirs('Thetas', exist_ok=True)
                
                # Save 4 sinogram variant TIFs per grain (raw, norm, abs, normabs)
                save_sinogram_variants(topdir, nGrs, maxNHKLs, nScans, grainSpots, omegas)
                
                # Reconstruct tomography using MIDAS_TOMO
                logger.info(f"Reconstructing tomography for {nGrs} grains using MIDAS_TOMO")
                os.makedirs('Tomo', exist_ok=True)
                
                # MIDAS_TOMO upscales detXdim to next power of 2 (reconDim).
                # Crop the center nScans×nScans region, aligning recon center
                # (reconDim/2) with the original grid center ((nScans-1)/2).
                from math import ceil, log2
                reconDim = 1 << int(ceil(log2(nScans))) if nScans > 1 else 1
                cropStart = reconDim // 2 - nScans // 2
                cropEnd = cropStart + nScans
                
                all_recons = np.zeros((nGrs, nScans, nScans))
                im_list = []
                
                for grNr in range(nGrs):
                    nSp = grainSpots[grNr]
                    thetas = omegas[grNr, :nSp]
                    
                    # Load the requested sinogram variant directly from the TIFF
                    sino_tif_fn = f'Sinos/sino_{sinoType}_grNr_{str(grNr).zfill(4)}.tif'
                    if os.path.exists(sino_tif_fn):
                        sino = np.array(Image.open(sino_tif_fn))
                    else:
                        logger.warning(f"Sinogram {sino_tif_fn} not found. Falling back to Sinos array.")
                        sino = np.transpose(Sinos[grNr, :nSp, :])
                    
                    # Save bare sino (for legacy compatibility) and thetas
                    Image.fromarray(np.transpose(Sinos[grNr, :nSp, :])).save(f'Sinos/sino_grNr_{str(grNr).zfill(4)}.tif')
                    np.savetxt(f'Thetas/thetas_grNr_{str(grNr).zfill(4)}.txt', thetas, fmt='%.6f')
                    
                    # Reconstruct: sino shape is (nScans, nSp), transpose to (nThetas, detXdim)
                    sino_for_tomo = sino.T  # (nSp, nScans) = (nThetas, detXdim)
                    recon_arr = run_tomo_from_sinos(
                        sino_for_tomo, 'Tomo', thetas,
                        shifts=0.0, filterNr=2, doLog=0,
                        extraPad=0, autoCentering=1, numCPUs=1, doCleanup=1)
                    recon_full = recon_arr[0, 0, :, :]  # (reconDim, reconDim)
                    recon = recon_full[cropStart:cropEnd, cropStart:cropEnd]  # (nScans, nScans)
                    all_recons[grNr, :, :] = recon
                    im_list.append(Image.fromarray(recon))
                    Image.fromarray(recon).save(f'Recons/recon_grNr_{str.zfill(str(grNr), 4)}.tif')
                
                # Create full reconstruction
                full_recon = np.max(all_recons, axis=0)
                logger.info("Finding orientation candidate at each location")
                
                max_id = np.argmax(all_recons, axis=0).astype(np.int32)
                max_id[full_recon == 0] = -1
                
                # Save max projection
                Image.fromarray(max_id).save('Recons/Full_recon_max_project_grID.tif')
                Image.fromarray(full_recon).save('Recons/Full_recon_max_project.tif')
                im_list[0].save('Recons/all_recons_together.tif', compression="tiff_deflate", save_all=True, append_images=im_list[1:])
                
                # Process unique orientations
                uniqueOrientations = np.genfromtxt(f'{topdir}/UniqueOrientations.csv', delimiter=' ')
                
                # Create spots to index file
                with open(f'{topdir}/SpotsToIndex.csv', 'w') as fSp:
                    for voxNr in range(nScans * nScans):
                        locX = voxNr % nScans - 1
                        locY = nScans - (voxNr // nScans + 1)
                        
                        if max_id[locY, locX] == -1:
                            continue
                            
                        orientThis = uniqueOrientations[max_id[locY, locX], 5:]
                        
                        unique_index_path = f'{topdir}/Output/UniqueIndexKeyOrientAll_voxNr_{str(voxNr).zfill(6)}.txt'
                        if os.path.isfile(unique_index_path):
                            with open(unique_index_path, 'r') as f:
                                lines = f.readlines()
                                
                            for line in lines:
                                orientInside = [float(val) for val in line.split()[4:]]
                                ang = rad2deg * GetMisOrientationAngleOM(orientThis, orientInside, sgnum)[0]
                                
                                if ang < maxang:
                                    lineSplit = line.split()
                                    outStr = f'{voxNr} {lineSplit[0]} {lineSplit[1]} {lineSplit[2]} {lineSplit[3]}\n'
                                    fSp.write(outStr)
                                    break
                
                # Create mic file if needed
                if not micFN:
                    logger.info("Creating dummy mic file for second indexing pass")
                    
                    fnSp = f'{topdir}/SpotsToIndex.csv'
                    if not os.path.exists(fnSp):
                        logger.error(f"SpotsToIndex.csv not found at {fnSp}")
                        sys.exit(1)
                        
                    spotsToIndex = np.genfromtxt(fnSp, delimiter=' ')
                    micFN = 'singleSolution.mic'
                    
                    with open(micFN, 'w') as micF:
                        micF.write("header\nheader\nheader\nheader\n")
                        
                        for spot in spotsToIndex:
                            voxNr = int(spot[0])
                            loc = int(spot[3])
                            
                            index_file = f'{topdir}/Output/IndexBest_voxNr_{str(voxNr).zfill(6)}.bin'
                            if not os.path.exists(index_file):
                                logger.warning(f"Index file not found: {index_file}")
                                continue
                                
                            data = np.fromfile(index_file, dtype=np.double, count=16, offset=loc)
                            xThis = data[11]
                            yThis = data[12]
                            omThis = data[2:11]
                            Euler = OrientMat2Euler(omThis)
                            
                            micF.write(f"0.0 0.0 0.0 {xThis:.6f} {yThis:.6f} 0.0 0.0 {Euler[0]:.6f} {Euler[1]:.6f} {Euler[2]:.6f} {omThis[0]:.6f} {omThis[1]:.6f} {omThis[2]:.6f} {omThis[3]:.6f} {omThis[4]:.6f} {omThis[5]:.6f} {omThis[6]:.6f} {omThis[7]:.6f} {omThis[8]:.6f}\n")
                    
                    # Update params file
                    with open(f'{topdir}/paramstest.txt', 'a') as paramsf:
                        paramsf.write(f'MicFile {topdir}/singleSolution.mic\n')
                    
                    # Move output directories
                    shutil.move(f'{topdir}/Output', f'{topdir}/fullOutput')
                    shutil.move(f'{topdir}/Results', f'{topdir}/fullResults')
                    
                    # Create new directories
                    Path(f'{topdir}/Output').mkdir(parents=True, exist_ok=True)
                    Path(f'{topdir}/Results').mkdir(parents=True, exist_ok=True)
                    
                    # Run indexing again
                    logger.info("Running indexing again with mic file")
                    
                    resIndex = []
                    for nodeNr in range(nNodes):
                        resIndex.append(indexscanning(topdir, numProcs, nScans, midas_path, blockNr=nodeNr, numBlocks=nNodes))
                        
                    outputIndex = [i.result() for i in resIndex]
                    
                    # Check for errors
                    for i, output in enumerate(outputIndex):
                        if output and "Failed" in output:
                            logger.error(f"Error in second indexing for node {i}: {output}")
                            sys.exit(1)
                    
                    # Run find single solution again
                    cmd = f"{os.path.join(midas_path, 'FF_HEDM/bin/findSingleSolutionPFRefactored')} {topdir} {sgnum} {maxang} {nScans} {numProcsLocal} {tol_ome} {tol_eta} {baseNameParamFN} {NormalizeIntensities} 1"
                    logger.info(f"Running findSingleSolutionPFRefactored again: {cmd}")
                    subprocess.call(cmd, cwd=topdir, shell=True)
                    
                    # Create spots to index file
                    with open(f'{topdir}/SpotsToIndex.csv', 'w') as f:
                        idData = np.fromfile(f'{topdir}/Output/UniqueIndexSingleKey.bin', dtype=np.uintp, count=nScans*nScans*5).reshape((-1, 5))
                        
                        for voxNr in range(nScans * nScans):
                            if idData[voxNr, 1] != 0:
                                f.write(f"{idData[voxNr, 0]} {idData[voxNr, 1]} {idData[voxNr, 2]} {idData[voxNr, 3]} {idData[voxNr, 4]}\n")
            else:
                # Create spots to index file for non-tomo case
                with open(f'{topdir}/SpotsToIndex.csv', 'w') as f:
                    idData = np.fromfile(f'{topdir}/Output/UniqueIndexSingleKey.bin', dtype=np.uintp, count=nScans*nScans*5).reshape((-1, 5))
                    
                    for voxNr in range(nScans * nScans):
                        if idData[voxNr, 1] != 0:
                            f.write(f"{idData[voxNr, 0]} {idData[voxNr, 1]} {idData[voxNr, 2]} {idData[voxNr, 3]} {idData[voxNr, 4]}\n")
            
            # Run refinement
            logger.info("Running refinement")
            os.makedirs('Results', exist_ok=True)
            
            resRefine = []
            for nodeNr in range(nNodes):
                resRefine.append(refinescanning(topdir, numProcs, midas_path, blockNr=nodeNr, numBlocks=nNodes))
                
            outputRefine = [i.result() for i in resRefine]
            
            # Check for errors in refinement
            for i, output in enumerate(outputRefine):
                if output and "Failed" in output:
                    logger.error(f"Error in refinement for node {i}: {output}")
                    
                    # Check error file
                    err_file = os.path.join(topdir, 'output', f'refining_err{i}.csv')
                    if os.path.exists(err_file):
                        with open(err_file, 'r') as f:
                            logger.error(f"Error file content: {f.read()}")
                    
                    sys.exit(1)
            
            # Filter final output
            logger.info(f"Filtering final output. Will be saved to {topdir}/Recons/microstrFull.csv and {topdir}/Recons/microstructure.hdf")
            
            # Get symmetries
            NrSym, Sym = MakeSymmetries(sgnum)
            
            # Process result files
            files2 = glob.glob(f'{topdir}/Results/*.csv')
            filesdata = np.zeros((len(files2), 43))
            i = 0
            info_arr = np.zeros((23, nScans * nScans))
            info_arr[:, :] = np.nan
            
            for fileN in files2:
                with open(fileN) as f:
                    voxNr = int(fileN.split('.')[-2].split('_')[-2])
                    _ = f.readline()
                    line = f.readline()
                    
                    if not line:
                        logger.warning(f"Empty result file: {fileN}")
                        continue
                        
                    data = line.split()
                    
                    for j in range(len(data)):
                        filesdata[i][j] = float(data[j])
                    
                    if isnan(filesdata[i][26]):
                        continue
                        
                    if filesdata[i][26] < 0 or filesdata[i][26] > 1.0000000001:
                        continue
                        
                    OM = filesdata[i][1:10]
                    quat = BringDownToFundamentalRegionSym(OrientMat2Quat(OM), NrSym, Sym)
                    filesdata[i][39:43] = quat
                    
                    # Store in info array
                    info_arr[:, voxNr] = filesdata[i][[0, -4, -3, -2, -1, 11, 12, 15, 16, 17, 18, 19, 20, 22, 23, 24, 26, 27, 28, 29, 31, 32, 35]]
                    
                    i += 1
            
            # Create header for output
            head = 'SpotID,O11,O12,O13,O21,O22,O23,O31,O32,O33,SpotID,x,y,z,SpotID,a,b,c,alpha,beta,gamma,SpotID,PosErr,OmeErr,InternalAngle,'
            head += 'Radius,Completeness,E11,E12,E13,E21,E22,E23,E31,E32,E33,Eul1,Eul2,Eul3,Quat1,Quat2,Quat3,Quat4'
            
            # Save output files
            np.savetxt(f'{topdir}/Recons/microstrFull.csv', filesdata, fmt='%.6f', delimiter=',', header=head)
            
            # Create HDF file
            with h5py.File(f'{topdir}/Recons/microstructure.hdf', 'w') as f:
                micstr = f.create_dataset(name='microstr', dtype=np.double, data=filesdata)
                micstr.attrs['Header'] = np.bytes_(head)
                
                # Process image data
                info_arr = info_arr.reshape((23, nScans, nScans))
                info_arr = np.flip(info_arr, axis=(1, 2))
                info_arr = info_arr.transpose(0, 2, 1)
                
                imgs = f.create_dataset(name='images', dtype=np.double, data=info_arr)
                imgs.attrs['Header'] = np.bytes_('ID,Quat1,Quat2,Quat3,Quat4,x,y,a,b,c,alpha,beta,gamma,posErr,omeErr,InternalAngle,Completeness,E11,E12,E13,E22,E23,E33')
        else:
            # Multiple solutions per voxel
            # Read Completeness from params file
            minConf = 0.0
            with open('paramstest.txt', 'r') as paramsf:
                lines = paramsf.readlines()
                for line in lines:
                    match = re.match(r"^\s*MinMatchesToAcceptFrac\s+([\d.]+)\s*;?\s*$", line)
                    if match:
                        try:
                            minConf = float(match.group(1))
                        except:
                            if line.strip().startswith('MinMatchesToAcceptFrac'):
                                print(f"Warning: Line starts with key but format is unexpected: {line.strip()}")
            cmd = f"{os.path.join(midas_path, 'FF_HEDM/bin/findMultipleSolutionsPF')} {topdir} {sgnum} {maxang} {nScans} {numProcsLocal} {minConf}"
            logger.info(f"Running findMultipleSolutionsPF: {cmd}")
            subprocess.call(cmd, shell=True, cwd=topdir)
            
            logger.info("Running refinement for all solutions found")
            
            # Run refinement
            resRefine = []
            for nodeNr in range(nNodes):
                resRefine.append(refinescanning(topdir, numProcs, midas_path, blockNr=nodeNr, numBlocks=nNodes))
                
            outputRefine = [inter.result() for inter in resRefine]
            
            # Check for errors
            for i, output in enumerate(outputRefine):
                if output and "Failed" in output:
                    logger.error(f"Error in refinement for node {i}: {output}")
                    sys.exit(1)
            
            # Process results
            NrSym, Sym = MakeSymmetries(sgnum)
            files2 = glob.glob(f'{topdir}/Results/*.csv')
            filesdata = np.zeros((len(files2), 43))
            i = 0
            
            for fileN in files2:
                with open(fileN) as f:
                    str1 = f.readline()
                    line = f.readline()
                    
                    if not line:
                        logger.warning(f"Empty result file: {fileN}")
                        continue
                        
                    data = line.split()
                    
                    for j in range(len(data)):
                        filesdata[i][j] = float(data[j])
                        
                    OM = filesdata[i][1:10]
                    quat = BringDownToFundamentalRegionSym(OrientMat2Quat(OM), NrSym, Sym)
                    filesdata[i][39:43] = quat
                    
                    i += 1
            
            # Create header and save
            head = 'SpotID,O11,O12,O13,O21,O22,O23,O31,O32,O33,SpotID,x,y,z,SpotID,a,b,c,alpha,beta,gamma,SpotID,PosErr,OmeErr,InternalAngle,Radius,Completeness,'
            head += 'E11,E12,E13,E21,E22,E23,E31,E32,E33,Eul1,Eul2,Eul3,Quat1,Quat2,Quat3,Quat4'
            
            np.savetxt('microstrFull.csv', filesdata, fmt='%.6f', delimiter=',', header=head)
        
        logger.info(f"All processing completed successfully. Time elapsed: {time.time() - startTime:.2f} seconds.")
        
    except Exception as e:
        logger.error(f"Unhandled exception in main: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()