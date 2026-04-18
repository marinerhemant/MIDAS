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
_tomo_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'TOMO')
if _tomo_dir not in sys.path:
    sys.path.insert(0, _tomo_dir)
from midas_tomo_python import run_tomo_from_sinos
# MLEM/OSEM reconstruction (alternative to FBP)
_utils_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'utils')
if _utils_dir not in sys.path:
    sys.path.insert(0, _utils_dir)
from mlem_recon import mlem as mlem_reconstruct, osem as osem_reconstruct
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
            out_fn = os.path.join(topdir, f'Sinos/sino_{label}_grNr_{grStr}.tif')
            Image.fromarray(sino).save(out_fn)
            if label == 'raw' and grNr < 3:
                logging.getLogger('pf_midas').info(
                    f"DEBUG save_sinogram_variants: grNr={grNr} label={label} "
                    f"shape={sino.shape} max={sino.max():.1f} fn={out_fn}")

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
                  omegaValues, minThresh, fStem, omegaFF, Ext, padding=6, scanStep=None):
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
    
    import midas_config
    midas_config.run_startup_checks()
    
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
            # Stream output: stdout to file, stderr to terminal (for tqdm progress bar)
            with open(outf_path, 'w') as f_out:
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=None,  # inherit terminal for tqdm progress bar
                    cwd=resFol,
                    bufsize=1,
                    universal_newlines=True
                )
                for line in process.stdout:
                    f_out.write(line)
                    if 'done:' in line or 'OutputZipName' in line or 'Processing' in line:
                        logger.info(f"[ZIP] {line.rstrip()}")
                process.wait()
            
            if process.returncode != 0:
                logger.error(f"ZIP generation failed with return code {process.returncode} for layer {layerNr}")
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
        effectiveStep = scanStep if scanStep is not None else nrFilesPerSweep
        thisStartNr = startNrFirstLayer + (layerNr - 1) * effectiveStep
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
            # Handle both '%YLab' (correct) and 'YLab' (old binaries where %Y was consumed by printf)
            ylab_col = '%YLab' if '%YLab' in dfAllF.columns else 'YLab'
            dfAllF.loc[dfAllF['GrainRadius'] > 0.001, ylab_col] += ypos
            dfAllF.loc[dfAllF['GrainRadius'] > 0.001, 'YOrig(NoWedgeCorr)'] += ypos
            dfAllF['Eta'] = CalcEtaAngleAll(dfAllF[ylab_col], dfAllF['ZLab'])
            dfAllF['Ttheta'] = rad2deg * np.arctan(np.linalg.norm(np.array([dfAllF[ylab_col], dfAllF['ZLab']]), axis=0) / Lsd)
            
            logger.info(f"Spots shape final for layer {layerNr}: {dfAllF.shape}")
            
            outFN2 = os.path.join(topdir, f'InputAllExtraInfoFittingAll{layerNr-1}.csv')
            t_st = time.time()
            
            # Fill NaN (from ragged C output lines) with 0 before writing
            dfAllF = dfAllF.fillna(0)
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
            # Use subprocess.run to capture return code
            result = subprocess.run(cmd_this, shell=True, stdout=f, stderr=f_err)
       
        # Always log stderr content for diagnostics
        if os.path.exists(f_err_path):
            with open(f_err_path, 'r') as ef:
                stderr_content = ef.read()
                if stderr_content.strip():
                    # Log all stderr (contains INFO/WARNING/ERROR from SaveBinDataScanning)
                    for line in stderr_content.strip().split('\n'):
                        if 'ERROR' in line:
                            logger.error(f"SaveBinDataScanning: {line}")
                        elif 'WARNING' in line:
                            logger.warning(f"SaveBinDataScanning: {line}")
                        else:
                            logger.info(f"SaveBinDataScanning: {line}")
        
        # Check return code
        if result.returncode != 0:
            logger.error(f"SaveBinDataScanning failed with return code {result.returncode}")
            logger.error(f"Check {f_err_path} for detailed error messages")
            return f"Failed to bin data (rc={result.returncode})"
        
        # Verify critical output files exist
        spots_file = os.path.join(resultDir, 'Spots.bin')
        if not os.path.exists(spots_file):
            logger.error(f"SaveBinDataScanning completed but Spots.bin not found at {spots_file}")
            logger.error(f"Check {f_err_path} for details")
            return "Failed to bin data: Spots.bin not created"
        elif os.path.getsize(spots_file) == 0:
            logger.error(f"Spots.bin exists but is 0 bytes at {spots_file}")
            return "Failed to bin data: Spots.bin is empty"
        else:
            logger.info(f"Spots.bin created successfully: {os.path.getsize(spots_file)} bytes")
        
        return "Successfully binned data"
        
    except Exception as e:
        logger.error(f"Exception in binData: {str(e)}")
        with open(f_err_path, 'a') as f_err:
            f_err.write(f"Exception: {str(e)}\n")
        return f"Failed with exception: {str(e)}"

@python_app
def indexscanning(resultDir, numProcs, num_scans, midas_path, blockNr=0, numBlocks=1, useGPU=0):
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
            indexer_bin = 'IndexerScanningGPU' if useGPU else 'IndexerScanningOMP'
            cmd_this = f"{os.path.join(midas_path, 'FF_HEDM/bin/' + indexer_bin)} paramstest.txt {blockNr} {numBlocks} {num_scans} {numProcs}"
            logger.info(f"Running {indexer_bin}: {cmd_this}")
            
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
def refinescanning(resultDir, numProcs, midas_path, blockNr=0, numBlocks=1, useGPU=0):
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
        
        refine_bin = 'FitOrStrainsScanningGPU' if useGPU else 'FitOrStrainsScanningOMP'
        cmd = f"{os.path.join(midas_path, 'FF_HEDM/bin/' + refine_bin)} paramstest.txt {blockNr} {numBlocks} {num_lines} {numProcs}"
        logger.info(f"Running {refine_bin}: {cmd}")
        
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
    parser.add_argument('-useGPU', type=int, required=False, default=0, help='Use GPU binaries (IndexerScanningGPU, FitOrStrainsScanningGPU) instead of OMP versions. Default: 0')
    parser.add_argument('-normalizeIntensities', type=int, required=False, default=2, help='Normalization mode for intensity in sinograms: 0=equivalent grain size, 1=powder-scaled, 2=integrated intensity (default), 3=raw sum intensity from PeaksFitting.')
    parser.add_argument('-convertFiles', type=int, required=False, default=1, help='If want to convert to zarr, if zarr files exist already, put to 0.')
    parser.add_argument('-runIndexing', type=int, required=False, default=1, help='If want to skip Indexing, put to 0.')
    parser.add_argument('-startScanNr', type=int, required=False, default=1, help='If you want to do partial peaksearch. Default: 1')
    parser.add_argument('-minThresh', type=int, required=False, default=-1, help='If you want to filter out peaks with intensity less than this number. -1 disables this. This is only used for filtering out peaksearch results for small peaks, peaks with maxInt smaller than this will be filtered out.')
    parser.add_argument('-sinoType', type=str, required=False, default='raw', choices=['raw', 'norm', 'abs', 'normabs'], help='Sinogram type to use for reconstruction (raw, norm, abs, normabs). Default: raw')
    parser.add_argument('-sinoSource', type=str, required=False, default='tolerance', choices=['indexing', 'tolerance'], help='Sinogram spot source: tolerance=match all spots by angular tolerance (default), indexing=use only spots from per-voxel indexing results (cleaner).')
    parser.add_argument('-reconMethod', type=str, required=False, default='fbp', choices=['fbp', 'mlem', 'osem'], help='Sinogram reconstruction method: fbp=filtered back-projection via gridrec (default), mlem=Maximum Likelihood EM (handles sparse/missing angles natively), osem=Ordered Subsets EM (accelerated MLEM).')
    parser.add_argument('-mlemIter', type=int, required=False, default=50, help='Number of MLEM/OSEM iterations (only used when -reconMethod is mlem or osem). Default: 50.')
    parser.add_argument('-osemSubsets', type=int, required=False, default=4, help='Number of ordered subsets for OSEM (only used when -reconMethod is osem). Default: 4.')
    parser.add_argument('-useEM', type=int, required=False, default=0,
                        help='Use EM spot-ownership for soft sinogram generation (requires doTomo=1). Default: 0 (off).')
    parser.add_argument('-emIter', type=int, required=False, default=20,
                        help='Number of EM iterations. Default: 20.')
    parser.add_argument('-emSigmaInit', type=float, required=False, default=0.1,
                        help='Initial sigma for EM Gaussian kernel (radians, ~6 degrees). Default: 0.1.')
    parser.add_argument('-emSigmaMin', type=float, required=False, default=0.005,
                        help='Minimum sigma for EM annealing floor (radians). Default: 0.005.')
    parser.add_argument('-emSigmaDecay', type=float, required=False, default=0.85,
                        help='Sigma decay factor per EM iteration. Default: 0.85.')
    parser.add_argument('-emRefineOrientations', type=int, required=False, default=1,
                        help='Whether EM M-step refines grain orientations. 0=E-step only, 1=full EM (default).')
    parser.add_argument('-emOptSteps', type=int, required=False, default=5,
                        help='Gradient steps per EM M-step per grain. Default: 5.')
    parser.add_argument('-emLR', type=float, required=False, default=0.005,
                        help='Learning rate for EM M-step Adam optimizer. Default: 0.005.')
    parser.add_argument('-resume', type=str, required=False, default='',
                        help='Path to a pipeline H5 file to resume from. Auto-detects the first incomplete stage.')
    parser.add_argument('-restartFrom', type=str, required=False, default='',
                        help='Stage name to restart from (re-runs all stages from that point). Valid: hkl, peak_search, merge, params_rewrite, indexing, refinement, find_multiple_solutions, consolidation')
    parser.add_argument('-skipValidation', action='store_true',
                        help='Skip midas-params preflight validation.')
    parser.add_argument('-strictValidation', action='store_true',
                        help='Exit on parameter-file validation errors (default: warn and continue).')
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
    useGPU = args.useGPU
    ConvertFiles = args.convertFiles
    runIndexing = args.runIndexing
    NormalizeIntensities = args.normalizeIntensities
    startScanNr = args.startScanNr
    minThresh = args.minThresh
    micFN = args.micFN
    grainsFN = args.grainsFN
    sinoType = args.sinoType
    sinoSource = args.sinoSource
    sinoMode = 1 if sinoSource == 'indexing' else 0
    useEM = args.useEM
    # Ensure command line file arguments are absolute paths before a potential directory change
    param_path = os.path.abspath(args.paramFile)
    param_dir = os.path.dirname(param_path)

    # --- midas-params preflight validation (soft dependency) ---
    try:
        from midas_params.hook import preflight_validate as _preflight
    except ImportError:
        _preflight = None
    if _preflight is not None:
        if not _preflight(
            param_file=param_path, pipeline="pf",
            skip=args.skipValidation, strict=args.strictValidation,
        ):
            sys.exit(1)

    # --- Auto-compute smarter runtime defaults for -numFrameChunks and
    #     -preProcThresh when left at the -1 sentinel. ---
    try:
        from midas_params.hook import resolve_runtime_defaults
        nchunks, preproc = resolve_runtime_defaults(
            param_file=param_path,
            num_frame_chunks=nchunks,
            pre_proc_thresh=preproc,
            n_cpus=numProcs,
        )
    except ImportError:
        pass
    if micFN and not os.path.isabs(micFN):
        micFN = os.path.abspath(micFN)
    if grainsFN and not os.path.isabs(grainsFN):
        grainsFN = os.path.abspath(grainsFN)
    if omegaFile and not os.path.isabs(omegaFile):
        omegaFile = os.path.abspath(omegaFile)
    if args.resume and not os.path.isabs(args.resume):
        args.resume = os.path.abspath(args.resume)

    # Use current directory if no result directory specified
    if not topdir:
        topdir = os.getcwd()
    topdir = os.path.abspath(topdir)
    
    logger.info(f'Working directory: {topdir}')
    logDir = os.path.join(topdir, 'output')
    
    # Create directories
    os.makedirs(topdir, exist_ok=True)
    os.makedirs(logDir, exist_ok=True)
    
    # Copy parameter file and positions.csv to result directory if it's different
    if topdir != param_dir:
        logger.info(f"Copying parameter file and positions.csv from {param_dir} to {topdir}")
        if os.path.exists(param_path):
            shutil.copy2(param_path, os.path.join(topdir, os.path.basename(param_path)))
            baseNameParamFN = os.path.basename(param_path)
        else:
            logger.error(f"Parameter file {param_path} not found.")
            sys.exit(1)
            
        positions_source = os.path.join(param_dir, 'positions.csv')
        if os.path.exists(positions_source):
            shutil.copy2(positions_source, os.path.join(topdir, 'positions.csv'))
        else:
            logger.warning(f"positions.csv not found in {param_dir}. Downstream steps may fail.")
    else:
        # If they are in the same directory, baseNameParamFN should just be the base name
        baseNameParamFN = os.path.basename(param_path)
            
    # Change into topdir early so outputs like hkls.csv and logs fall in resultDir
    if os.getcwd() != topdir:
        os.chdir(topdir)
    
    # Get MIDAS installation directory dynamically
    midas_path = get_installation_dir()
    logger.info(f"Using MIDAS installation directory: {midas_path}")
    
    # Import required modules using midas_path
    utils_dir = os.path.join(midas_path, 'utils')
    v7_dir = os.path.join(midas_path, 'FF_HEDM/v7')
    sys.path.insert(0, utils_dir)
    sys.path.insert(0, v7_dir)
    
    from version import version_string, stamp_h5
    from pipeline_state import PipelineH5, find_resume_stage, load_resume_info

    PF_STAGE_ORDER = [
        'hkl', 'peak_search', 'merge', 'params_rewrite', 'indexing',
        'refinement', 'find_multiple_solutions', 'consolidation'
    ]

    # --- Resume handling ---
    resume_from_stage = ''
    if args.resume:
        if not os.path.exists(args.resume):
            logger.error(f"Resume H5 not found: {args.resume}")
            sys.exit(1)
        resume_from_stage = find_resume_stage(args.resume, PF_STAGE_ORDER)
        if resume_from_stage:
            info = load_resume_info(args.resume)
            logger.info(f"Resuming from stage '{resume_from_stage}'. "
                        f"Completed: {info['completed_stages']}")
        else:
            logger.info("All stages complete. Re-running consolidation.")
            resume_from_stage = 'consolidation'
    elif args.restartFrom:
        if args.restartFrom not in PF_STAGE_ORDER:
            logger.error(f"Invalid restart stage '{args.restartFrom}'. Valid: {PF_STAGE_ORDER}")
            sys.exit(1)
        resume_from_stage = args.restartFrom
        logger.info(f"Restarting from explicit stage: {resume_from_stage}")

    _skip_before = -1
    if resume_from_stage and resume_from_stage in PF_STAGE_ORDER:
        _skip_before = PF_STAGE_ORDER.index(resume_from_stage)
    def _should_run(stage_name):
        if _skip_before < 0:
            return True
        return PF_STAGE_ORDER.index(stage_name) >= _skip_before if stage_name in PF_STAGE_ORDER else True
    logger.info(version_string())
    
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
        param_text = ''.join(paramContents)

        # Initialize consolidated H5
        h5_path = os.path.join(topdir, 'Recons', 'microstructure_pf.h5')
        os.makedirs(os.path.join(topdir, 'Recons'), exist_ok=True)
        ph5 = PipelineH5(h5_path, 'pf_midas', vars(args), param_text)
        ph5.__enter__()
        
        # Initialize variables
        RingNrs = []
        nMerges = 0
        maxang = 1
        tol_ome = 1
        tol_eta = 1
        omegaFN = ''
        omegaFF = -1
        padding = 6
        scanStep = None  # Will default to nrFilesPerSweep if not set
        
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
            elif line.startswith('ScanStep'):
                scanStep = int(line.split()[1])
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
        if _should_run('hkl'):
            cmd = f"{os.path.join(midas_path, 'FF_HEDM/bin/GetHKLList')} {baseNameParamFN}"
            logger.info(f"Running GetHKLList: {cmd}")
            subprocess.call(cmd, shell=True)
        else:
            logger.info("RESUME: skipping hkl stage")
        
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
        if _should_run('hkl'):
            ph5.mark('hkl')
            ph5.write_dataset('parameters/RingNrs', np.array(RingNrs, dtype=np.int32))
            ph5.write_dataset('parameters/RingRadii', np.array(rads))
            ph5.write_dataset('parameters/sgnum', sgnum)
            ph5.write_dataset('parameters/nScans', nScans)
            ph5.write_dataset('parameters/BeamSize', BeamSize)
            ph5.write_dataset('parameters/topdir', topdir)
            if grainsFN:
                ph5.write_dataset('parameters/grainsFN', grainsFN)
        
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
        if _should_run('peak_search'):
            if doPeakSearch == 1 or doPeakSearch == -1:
                logger.info(f"Starting peak search for {nScans} scans starting from {startScanNr}")
                
                # Use parsl to run in parallel
                res = []
                for layerNr in range(startScanNr, nScans + 1):
                    res.append(parallel_peaks(
                        layerNr, positions, startNrFirstLayer, nrFilesPerSweep, topdir,
                        paramContents, baseNameParamFN, ConvertFiles, nchunks, preproc,
                        midas_path, doPeakSearch, numProcs, startNr, endNr, Lsd, NormalizeIntensities,
                        omegaValues, minThresh, fStem, omegaFF, Ext, padding, scanStep
                    ))
                
                # Wait for all tasks to complete
                outputs = [i.result() for i in res]
                
                # Check for errors in outputs
                for i, output in enumerate(outputs):
                    if output and "Failed" in output:
                        logger.error(f"Error in peak search for layer {startScanNr + i}: {output}")
                        
                        # Check error files for this layer
                        layerNr = startScanNr + i
                        effectiveStep = scanStep if scanStep is not None else nrFilesPerSweep
                        thisStartNr = startNrFirstLayer + (layerNr - 1) * effectiveStep
                        folderName = str(thisStartNr)
                        thisDir = os.path.join(topdir, folderName)
                        err_file = os.path.join(thisDir, 'output', 'processing_err0.csv')
                        
                        if os.path.exists(err_file):
                            with open(err_file, 'r') as f:
                                logger.error(f"Error file content: {f.read()}")
                        
                        sys.exit(1)
                
                logger.info(f'Peak search completed on {nNodes} nodes.')
            else:
                logger.info("Peak search skipped (doPeakSearch=0)")
                if nMerges != 0:
                    for layerNr in range(0, nMerges * (nScans // nMerges)):
                        if os.path.exists(f'original_InputAllExtraInfoFittingAll{layerNr}.csv'):
                            shutil.move(
                                f'original_InputAllExtraInfoFittingAll{layerNr}.csv',
                                f'InputAllExtraInfoFittingAll{layerNr}.csv'
                            )
            ph5.mark('peak_search')
            
            # Ingest peak search results into H5 (works whether peaksearch
            # was actually run or skipped via -doPeakSearch 0)
            spots_per_scan = []
            total_spots = 0
            for scanNr in range(nScans):
                csv_fn = os.path.join(topdir, f'InputAllExtraInfoFittingAll{scanNr}.csv')
                if os.path.exists(csv_fn):
                    try:
                        df = pd.read_csv(csv_fn, delimiter=' ', skipinitialspace=True)
                        n = len(df)
                        spots_per_scan.append(n)
                        total_spots += n
                    except Exception as e:
                        logger.warning(f"Could not read {csv_fn}: {e}")
                        spots_per_scan.append(0)
                else:
                    spots_per_scan.append(0)
            ph5.write_dataset('peak_search/spots_per_scan', np.array(spots_per_scan, dtype=np.int64))
            ph5.write_dataset('peak_search/total_spots', total_spots)
            ph5.write_dataset('peak_search/doPeakSearch', doPeakSearch)
            logger.info(f"Peak search summary: {total_spots} total spots across {nScans} scans")
        else:
            logger.info("RESUME: skipping peak_search stage")
        
        # Handle merges
        if nMerges != 0:
            if _should_run('merge'):
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
                ph5.mark('merge')
                ph5.write_dataset('merge_info/was_merged', True)
                ph5.write_dataset('merge_info/nMerges', nMerges)
                ph5.write_dataset('merge_info/post_merge_nScans', nScans)
                ph5.write_dataset('merge_info/post_merge_BeamSize', BeamSize)
            else:
                logger.info("RESUME: skipping merge stage")
                # Still update nScans/BeamSize since downstream stages need them
                nScans = int(floor(nScans / nMerges))
                BeamSize *= nMerges
                with open(os.path.join(topdir, 'positions.csv'), 'r') as f:
                    positions = f.readlines()
        
        # Prepare for indexing and refinement
        os.chdir(topdir)
        Path(os.path.join(topdir, 'Output')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(topdir, 'Results')).mkdir(parents=True, exist_ok=True)
        
        # Update parameters file
        if _should_run('params_rewrite'):
            with open('paramstest.txt', 'r') as paramsf:
                lines = paramsf.readlines()
                
            with open('paramstest.txt', 'w') as paramsf:
                for line in lines:
                    if any(line.startswith(x) for x in ['RingNumbers', 'MarginRadius', 'RingRadii', 'RingToIndex', 'BeamSize', 'px']):
                        continue
                    if line.startswith('MicFile'):
                        continue
                    if line.startswith('GrainsFile'):
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
                
                if micFN:
                    paramsf.write(f'MicFile {micFN}\n')
                if grainsFN:
                    paramsf.write(f'GrainsFile {grainsFN}\n')
                    logger.info(f"Added GrainsFile to paramstest.txt: {grainsFN}")
            ph5.mark('params_rewrite')
            ph5.write_dataset('parameters/nScans_final', nScans)
            ph5.write_dataset('parameters/BeamSize_final', BeamSize)
        else:
            logger.info("RESUME: skipping params_rewrite stage")
        
        # Determine execution path
        if oneSolPerVox == 1:
            if doTomo == 1:
                exec_path = 'single_tomo'
            else:
                exec_path = 'single_no_tomo'
        else:
            exec_path = 'multi_solution'
        ph5.write_dataset('parameters/execution_path', exec_path)
        
        # Run indexing if requested
        if _should_run('indexing'):
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
                    resIndex.append(indexscanning(topdir, numProcs, nScans, midas_path, blockNr=nodeNr, numBlocks=nNodes, useGPU=useGPU))
                    
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
            else:
                logger.info("Indexing skipped (runIndexing=0)")
            ph5.mark('indexing')
        else:
            logger.info("RESUME: skipping indexing stage")
        
        # Handle single solution per voxel
        if not _should_run('refinement'):
            logger.info("RESUME: skipping refinement and downstream stages")
        elif oneSolPerVox == 1:
            # Prepare for tomography if requested
            if doTomo == 1:
                # Remove existing sinos
                for pattern in ["sinos_*.bin", "omegas_*.bin", "nrHKLs_*.bin"]:
                    sinoFNs = glob.glob(pattern)
                    for sinoF in sinoFNs:
                        os.remove(sinoF)
                
                # Remove existing directories
                for dirn in ['Sinos', 'Thetas']:
                    if os.path.isdir(dirn):
                        shutil.rmtree(dirn)
                # Clean Recons/ contents but keep the directory and the pipeline H5
                if os.path.isdir('Recons'):
                    for entry in os.listdir('Recons'):
                        if entry == 'microstructure_pf.h5':
                            continue
                        p = os.path.join('Recons', entry)
                        if os.path.isdir(p):
                            shutil.rmtree(p)
                        else:
                            os.remove(p)
                
                # Move result directories if they exist
                for dirn in ['fullResults', 'fullOutput']:
                    if os.path.isdir(dirn):
                        if os.path.isdir(dirn[4:]):
                            shutil.rmtree(dirn[4:])
                        shutil.move(dirn, dirn[4:])
            
            # Run find single solution
            cmd = f"{os.path.join(midas_path, 'FF_HEDM/bin/findSingleSolutionPFRefactored')} {topdir} {sgnum} {maxang} {nScans} {numProcsLocal} {tol_ome} {tol_eta} {baseNameParamFN} {NormalizeIntensities} 1 {sinoMode}"
            logger.info(f"Running findSingleSolutionPFRefactored: {cmd}")
            result = subprocess.call(cmd, cwd=topdir, shell=True)
            
            if result != 0:
                logger.error("Error in findSingleSolutionPFRefactored")
                sys.exit(1)
                
            os.makedirs('Recons', exist_ok=True)
            
            # Create spots to index file from UniqueIndexSingleKey.bin
            with open(f'{topdir}/SpotsToIndex.csv', 'w') as f:
                idData = np.fromfile(f'{topdir}/Output/UniqueIndexSingleKey.bin', dtype=np.uintp, count=nScans*nScans*5).reshape((-1, 5))
                for voxNr in range(nScans * nScans):
                    if idData[voxNr, 1] != 0:
                        f.write(f"{idData[voxNr, 0]} {idData[voxNr, 1]} {idData[voxNr, 2]} {idData[voxNr, 3]} {idData[voxNr, 4]}\n")

            # EM spot-ownership: pre-tomo refinement + soft sinograms
            if useEM == 1 and doTomo == 1:
                logger.info("=== EM Spot-Ownership Pipeline ===")

                # Step 1: Pre-tomo refinement to get better initial orientations
                logger.info("Running pre-tomo refinement for EM initialization")

                # Save first-pass Output/Results before refinement overwrites
                for dirn in ['Results']:
                    if os.path.isdir(dirn):
                        shutil.rmtree(dirn)
                    os.makedirs(dirn, exist_ok=True)

                resRefine0 = []
                for nodeNr in range(nNodes):
                    resRefine0.append(refinescanning(topdir, numProcs, midas_path,
                                                      blockNr=nodeNr, numBlocks=nNodes,
                                                      useGPU=useGPU))
                outputRefine0 = [i.result() for i in resRefine0]
                for i, output in enumerate(outputRefine0):
                    if output and "Failed" in output:
                        logger.warning(f"Pre-tomo refinement warning for node {i}: {output}")

                logger.info("Pre-tomo refinement complete. Updating grain orientations.")

                # Step 2: Re-derive unique grain orientations from refined results
                from em_pf_integration import update_unique_orientations_from_refinement
                update_unique_orientations_from_refinement(topdir, nScans)

                # Step 3: Run EM spot-ownership
                logger.info("Running EM spot-ownership for weighted sinograms")
                from em_pf_integration import run_em_spot_ownership

                # EM re-weights the existing sinograms in-place
                # (preserves C code's sinogram structure, just adjusts intensities)
                run_em_spot_ownership(
                    topdir=topdir,
                    n_scans=nScans,
                    n_iter=getattr(args, 'emIter', 20),
                    sigma_init=getattr(args, 'emSigmaInit', 0.1),
                    sigma_min=getattr(args, 'emSigmaMin', 0.005),
                    sigma_decay=getattr(args, 'emSigmaDecay', 0.85),
                    tol_ome_override=tol_ome,
                    tol_eta_override=tol_eta,
                    n_opt_steps=getattr(args, 'emOptSteps', 5),
                    lr=getattr(args, 'emLR', 0.005),
                    refine_orientations=bool(getattr(args, 'emRefineOrientations', 1)),
                    use_refined_orientations=True,
                )

                logger.info("EM complete. Proceeding with tomo reconstruction on EM sinograms.")

            # Run tomography if requested
            if doTomo == 1:
                # Find sino file
                sinoFNs = glob.glob("sinos_*.bin")
                # Filter out variant files (sinos_raw_*, sinos_norm_*, etc.)
                # The base file has format sinos_N_M_S.bin (4 parts when split by _)
                sinoFNs = [f for f in sinoFNs if f.count('_') == 3 and f.split('_')[1].isdigit()]
                if not sinoFNs:
                    logger.error("No base sino file found (expected sinos_N_M_S.bin)")
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
                
                # Load the requested sinogram variant from its binary file
                sinoVariantFNs = glob.glob(f"sinos_{sinoType}_*.bin")
                sinoVariantFNs = [f for f in sinoVariantFNs
                                  if f.count('_') == 4 and f.split('_')[2].isdigit()]
                if sinoVariantFNs:
                    SinosVariant = np.fromfile(sinoVariantFNs[0], dtype=np.double,
                                               count=nGrs*maxNHKLs*nScans
                                               ).reshape((nGrs, maxNHKLs, nScans))
                    logger.info(f"Loaded sinogram variant '{sinoType}' from {sinoVariantFNs[0]}")
                else:
                    logger.warning(f"Sinogram variant sinos_{sinoType}_*.bin not found. "
                                   f"Using main sinogram.")
                    SinosVariant = Sinos

                all_recons = np.zeros((nGrs, nScans, nScans))
                im_list = []

                for grNr in range(nGrs):
                    nSp = grainSpots[grNr]
                    thetas = omegas[grNr, :nSp]

                    # Read sinogram from binary array (consistent grain ordering)
                    sino = np.transpose(SinosVariant[grNr, :nSp, :])  # (nScans, nSp)

                    # Save TIFs for visualization and thetas
                    Image.fromarray(sino).save(f'Sinos/sino_{sinoType}_grNr_{str(grNr).zfill(4)}.tif')
                    Image.fromarray(np.transpose(Sinos[grNr, :nSp, :])).save(f'Sinos/sino_grNr_{str(grNr).zfill(4)}.tif')
                    np.savetxt(f'Thetas/thetas_grNr_{str(grNr).zfill(4)}.txt', thetas, fmt='%.6f')

                    # Reconstruct: sino shape is (nScans, nSp), transpose to (nThetas, detXdim)
                    sino_for_tomo = sino.T  # (nSp, nScans) = (nThetas, detXdim)
                    reconMethod = getattr(args, 'reconMethod', 'fbp')
                    if reconMethod == 'mlem':
                        mlemIter = getattr(args, 'mlemIter', 50)
                        recon = mlem_reconstruct(sino_for_tomo, thetas, n_iter=mlemIter)
                        recon = recon[:nScans, :nScans].T  # transpose to match FBP spatial convention
                    elif reconMethod == 'osem':
                        mlemIter = getattr(args, 'mlemIter', 50)
                        osemSubsets = getattr(args, 'osemSubsets', 4)
                        recon = osem_reconstruct(sino_for_tomo, thetas, n_iter=mlemIter, n_subsets=osemSubsets)
                        recon = recon[:nScans, :nScans].T  # transpose to match FBP spatial convention
                    else:  # fbp (default)
                        recon_arr = run_tomo_from_sinos(
                            sino_for_tomo, 'Tomo', thetas,
                            shifts=0.0, filterNr=2, doLog=0,
                            extraPad=0, autoCentering=1, numCPUs=1, doCleanup=1)
                        recon_full = recon_arr[0, 0, :, :]  # (reconDim, reconDim)
                        recon = recon_full[cropStart:cropEnd, cropStart:cropEnd]  # (nScans, nScans)
                    recon = recon.T  # transpose to match voxel grid convention
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

                # Generate mic file from tomo grain map for re-indexing.
                # The mic file seeds IndexerScanningOMP with the tomo-assigned
                # orientation at each voxel, so it finds the matched spots for
                # that specific grain at each position.
                pos_vals = np.loadtxt(f'{topdir}/positions.csv')
                pos_sorted = np.sort(pos_vals)

                micFNTomo = f'{topdir}/singleSolution.mic'
                with open(micFNTomo, 'w') as micF:
                    micF.write("header\nheader\nheader\nheader\n")
                    for voxNr in range(nScans * nScans):
                        row = voxNr // nScans
                        col = voxNr % nScans
                        if max_id[row, col] == -1:
                            continue
                        xThis = pos_sorted[row]
                        yThis = pos_sorted[col]
                        omThis = uniqueOrientations[max_id[row, col], 5:]
                        Euler = OrientMat2Euler(np.array(omThis))
                        micF.write(f"0.0 0.0 0.0 {xThis:.6f} {yThis:.6f} 0.0 0.0 "
                                   f"{Euler[0]:.6f} {Euler[1]:.6f} {Euler[2]:.6f} 0.0 0.0\n")

                # Add MicFile to paramstest.txt
                with open(f'{topdir}/paramstest.txt', 'a') as paramsf:
                    paramsf.write(f'MicFile {micFNTomo}\n')

                # Save first-pass Output and Results, create fresh directories
                for dirn in ['Output', 'Results']:
                    fullDirn = f'full{dirn}'
                    if os.path.isdir(fullDirn):
                        shutil.rmtree(fullDirn)
                    shutil.move(dirn, fullDirn)
                    os.makedirs(dirn, exist_ok=True)

                # Re-run indexing with mic-seeded orientations
                logger.info("Re-running indexing with tomo-seeded mic file")
                resIndex = []
                for nodeNr in range(nNodes):
                    resIndex.append(indexscanning(topdir, numProcs, nScans, midas_path,
                                                  blockNr=nodeNr, numBlocks=nNodes, useGPU=useGPU))
                outputIndex = [i.result() for i in resIndex]
                for i, output in enumerate(outputIndex):
                    if output and "Failed" in output:
                        logger.error(f"Error in tomo-seeded indexing for node {i}: {output}")
                        sys.exit(1)

                # Build SpotsToIndex.csv directly from new IndexBest_all.bin
                # (mic-seeded indexing produces 1 solution per voxel, solIndex=0)
                consol_file = f'{topdir}/Output/IndexBest_all.bin'
                with open(consol_file, 'rb') as cf:
                    nVoxels = np.frombuffer(cf.read(4), dtype=np.int32)[0]
                    nSolArr = np.frombuffer(cf.read(4 * nVoxels), dtype=np.int32)
                    offArr = np.frombuffer(cf.read(8 * nVoxels), dtype=np.int64)
                    headerSize = 4 + 4 * nVoxels + 8 * nVoxels
                    allData = np.frombuffer(cf.read(), dtype=np.double)

                with open(f'{topdir}/SpotsToIndex.csv', 'w') as fSp:
                    nWritten = 0
                    for voxNr in range(nVoxels):
                        if nSolArr[voxNr] == 0:
                            continue
                        dataOffset = int((offArr[voxNr] - headerSize) // 8)
                        row = allData[dataOffset:dataOffset + 16]
                        spotID = int(row[0])
                        nExpected = int(row[14])
                        nMatched = int(row[15])
                        fSp.write(f"{voxNr} {spotID} {nMatched} {nExpected} 0\n")
                        nWritten += 1
                logger.info(f"Built SpotsToIndex.csv from tomo-seeded indexing: "
                            f"{nWritten}/{nVoxels} voxels")

                # Remove MicFile line from paramstest.txt (so refinement doesn't re-seed)
                with open(f'{topdir}/paramstest.txt', 'r') as f:
                    paramLines = f.readlines()
                with open(f'{topdir}/paramstest.txt', 'w') as f:
                    for line in paramLines:
                        if not line.startswith('MicFile'):
                            f.write(line)
                
            # Run refinement (clear stale results so consolidation only sees this run)
            logger.info("Running refinement")
            if os.path.isdir('Results'):
                shutil.rmtree('Results')
            os.makedirs('Results', exist_ok=True)
            
            resRefine = []
            for nodeNr in range(nNodes):
                resRefine.append(refinescanning(topdir, numProcs, midas_path, blockNr=nodeNr, numBlocks=nNodes, useGPU=useGPU))
                
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
            
            # Create legacy HDF file for backward compatibility
            with h5py.File(f'{topdir}/Recons/microstructure.hdf', 'w') as f:
                stamp_h5(f)
                micstr = f.create_dataset(name='microstr', dtype=np.double, data=filesdata)
                micstr.attrs['Header'] = np.bytes_(head)
                
                # Process image data
                info_arr = info_arr.reshape((23, nScans, nScans))
                info_arr = np.flip(info_arr, axis=(1, 2))
                info_arr = info_arr.transpose(0, 2, 1)
                
                imgs = f.create_dataset(name='images', dtype=np.double, data=info_arr)
                imgs.attrs['Header'] = np.bytes_('ID,Quat1,Quat2,Quat3,Quat4,x,y,a,b,c,alpha,beta,gamma,posErr,omeErr,InternalAngle,Completeness,E11,E12,E13,E22,E23,E33')
            
            # Write to consolidated H5
            ph5.mark('refinement')
            ph5.write_dataset('voxels/microstr', filesdata)
            ph5.h5['voxels/microstr'].attrs['Header'] = np.bytes_(head)
            
            info_arr_final = info_arr.reshape((23, nScans, nScans))
            info_arr_final = np.flip(info_arr_final, axis=(1, 2))
            info_arr_final = info_arr_final.transpose(0, 2, 1)
            ph5.write_dataset('images/data', info_arr_final)
            ph5.h5['images/data'].attrs['Header'] = np.bytes_('ID,Quat1,Quat2,Quat3,Quat4,x,y,a,b,c,alpha,beta,gamma,posErr,omeErr,InternalAngle,Completeness,E11,E12,E13,E22,E23,E33')
            
            # Extract per-voxel structured data
            valid = ~np.isnan(filesdata[:, 26]) & (filesdata[:, 26] >= 0) & (filesdata[:, 26] <= 1.0000001)
            valid_data = filesdata[valid]
            if valid_data.size > 0:
                ph5.write_dataset('voxels/position', valid_data[:, 11:14])
                ph5.write_dataset('voxels/orientation_matrix', valid_data[:, 1:10].reshape(-1, 3, 3))
                ph5.write_dataset('voxels/euler_angles', valid_data[:, 35:38])
                ph5.write_dataset('voxels/quaternion', valid_data[:, 39:43])
                ph5.write_dataset('voxels/lattice_params', valid_data[:, 15:21])
                ph5.write_dataset('voxels/strain', valid_data[:, 27:36].reshape(-1, 3, 3))
                ph5.write_dataset('voxels/completeness', valid_data[:, 26])
                ph5.write_dataset('voxels/pos_error', valid_data[:, 22])
                ph5.write_dataset('voxels/ome_error', valid_data[:, 23])
                ph5.write_dataset('voxels/internal_angle', valid_data[:, 24])
                ph5.write_dataset('voxels/radius', valid_data[:, 25])
            
            ph5.mark('consolidation')
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
                resRefine.append(refinescanning(topdir, numProcs, midas_path, blockNr=nodeNr, numBlocks=nNodes, useGPU=useGPU))
                
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

            # Write multi-solution results to consolidated H5
            ph5.mark('find_multiple_solutions')
            ph5.write_dataset('voxels/microstr', filesdata)
            ph5.h5['voxels/microstr'].attrs['Header'] = np.bytes_(head)

            valid = filesdata[:, :1].ravel() > 0  # has valid SpotID
            valid_data = filesdata[valid]
            if valid_data.size > 0:
                ph5.write_dataset('voxels/position', valid_data[:, 11:14])
                ph5.write_dataset('voxels/orientation_matrix', valid_data[:, 1:10].reshape(-1, 3, 3))
                ph5.write_dataset('voxels/euler_angles', valid_data[:, 35:38])
                ph5.write_dataset('voxels/quaternion', valid_data[:, 39:43])
                ph5.write_dataset('voxels/lattice_params', valid_data[:, 15:21])
                ph5.write_dataset('voxels/strain', valid_data[:, 27:36].reshape(-1, 3, 3))
                ph5.write_dataset('voxels/completeness', valid_data[:, 26])
            ph5.mark('consolidation')
        
        logger.info(f"All processing completed successfully. Time elapsed: {time.time() - startTime:.2f} seconds.")
        ph5.__exit__(None, None, None)
        
    except Exception as e:
        logger.error(f"Unhandled exception in main: {str(e)}")
        logger.error(traceback.format_exc())
        try:
            ph5.__exit__(type(e), e, e.__traceback__)
        except Exception:
            pass
        sys.exit(1)

if __name__ == "__main__":
    main()