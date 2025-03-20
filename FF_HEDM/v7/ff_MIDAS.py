#!/usr/bin/env python

import parsl
import subprocess
import sys, os
import time
import argparse
import signal
import shutil
import re
import logging
import numpy as np
from typing import Optional, Dict, List, Tuple, Any, Union

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('MIDAS')

# Set paths
utilsDir = os.path.expanduser('~/opt/MIDAS/utils/')
v7Dir = os.path.expanduser('~/opt/MIDAS/FF_HEDM/v7/')
sys.path.insert(0, utilsDir)
sys.path.insert(0, v7Dir)
from parsl.app.app import python_app
pytpath = sys.executable

def run_command(cmd: str, working_dir: str, out_file: str, err_file: str) -> int:
    """Run a shell command and check for errors.
    
    Args:
        cmd: Command to run
        working_dir: Directory to run command in
        out_file: Path to save stdout
        err_file: Path to save stderr
        
    Returns:
        Return code from the command
    
    Raises:
        RuntimeError: If command fails
    """
    logger.info(f"Running: {cmd}")
    
    with open(out_file, 'w') as f_out, open(err_file, 'w') as f_err:
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=f_out, 
            stderr=f_err, 
            cwd=working_dir,
            env=get_midas_env()
        )
        returncode = process.wait()
        
    # Check if command failed
    if returncode != 0:
        with open(err_file, 'r') as f:
            error_content = f.read()
        error_msg = f"Command failed with return code {returncode}:\n{cmd}\nError output:\n{error_content}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
        
    return returncode

def get_midas_env() -> Dict[str, str]:
    """Get the environment variables for MIDAS."""
    env = dict(os.environ)
    midas_path = os.path.expanduser("~/.MIDAS")
    libpth = os.environ.get('LD_LIBRARY_PATH', '')
    env['LD_LIBRARY_PATH'] = f'{midas_path}/BLOSC/lib64:{midas_path}/FFTW/lib:{midas_path}/HDF5/lib:{midas_path}/LIBTIFF/lib:{midas_path}/LIBZIP/lib64:{midas_path}/NLOPT/lib:{midas_path}/ZLIB/lib:{libpth}'
    return env

def generateZip(
    resFol: str,
    pfn: str,
    layerNr: int,
    dfn: str = '',
    dloc: str = '',
    nchunks: int = -1,
    preproc: int = -1,
    outf: str = 'ZipOut.txt',
    errf: str = 'ZipErr.txt'
) -> Optional[str]:
    """Generate ZIP file from data.
    
    Args:
        resFol: Result folder
        pfn: Parameter file name
        layerNr: Layer number
        dfn: Data file name
        dloc: Data location
        nchunks: Number of frame chunks
        preproc: Pre-processing threshold
        outf: Output file name
        errf: Error file name
        
    Returns:
        ZIP file name if successful, None otherwise
    """
    cmd = f"{pytpath} {os.path.expanduser('~/opt/MIDAS/utils/ffGenerateZip.py')} -resultFolder {resFol} -paramFN {pfn} -LayerNr {layerNr}"
    
    if dfn:
        cmd += f' -dataFN {dfn}'
    if dloc:
        cmd += f' -dataLoc {dloc}'
    if nchunks != -1:
        cmd += f' -numFrameChunks {nchunks}'
    if preproc != -1:
        cmd += f' -preProcThresh {preproc}'
        
    outf_path = f"{resFol}/output/{outf}"
    errf_path = f"{resFol}/output/{errf}"
    
    try:
        run_command(cmd, resFol, outf_path, errf_path)
        
        with open(outf_path, 'r') as f:
            lines = f.readlines()
            
        if lines and lines[-1].startswith('OutputZipName'):
            return lines[-1].split()[1]
        else:
            logger.error("Could not find OutputZipName in the output")
            return None
    except Exception as e:
        logger.error(f"Failed to generate ZIP: {e}")
        return None

@python_app
def peaks(resultDir: str, zipFN: str, numProcs: int, blockNr: int = 0, numBlocks: int = 1) -> None:
    """Run peak search.
    
    Args:
        resultDir: Result directory
        zipFN: ZIP file name
        numProcs: Number of processors
        blockNr: Block number
        numBlocks: Number of blocks
    """
    import subprocess
    import os
    import sys
    
    # Set up environment
    env = dict(os.environ)
    midas_path = os.path.expanduser("~/.MIDAS")
    libpth = os.environ.get('LD_LIBRARY_PATH', '')
    env['LD_LIBRARY_PATH'] = f'{midas_path}/BLOSC/lib64:{midas_path}/FFTW/lib:{midas_path}/HDF5/lib:{midas_path}/LIBTIFF/lib:{midas_path}/LIBZIP/lib64:{midas_path}/NLOPT/lib:{midas_path}/ZLIB/lib:{libpth}'
    
    outfile = f'{resultDir}/output/peaksearch_out{blockNr}.csv'
    errfile = f'{resultDir}/output/peaksearch_err{blockNr}.csv'
    
    with open(outfile, 'w') as f, open(errfile, 'w') as f_err:
        cmd = f"{os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/PeaksFittingOMPZarr')} {zipFN} {blockNr} {numBlocks} {numProcs}"
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            env=env, 
            stdout=f, 
            stderr=f_err, 
            cwd=resultDir
        )
        returncode = process.wait()
        
        if returncode != 0:
            f_err.flush()
            with open(errfile, 'r') as err_reader:
                error_content = err_reader.read()
            error_msg = f"Peak search failed with return code {returncode}. Error output:\n{error_content}"
            print(error_msg, file=sys.stderr)
            raise RuntimeError(error_msg)

@python_app
def index(resultDir: str, numProcs: int, blockNr: int = 0, numBlocks: int = 1) -> None:
    """Run indexing.
    
    Args:
        resultDir: Result directory
        numProcs: Number of processors
        blockNr: Block number
        numBlocks: Number of blocks
    """
    import subprocess
    import os
    import sys
    
    os.chdir(resultDir)
    
    # Set up environment
    env = dict(os.environ)
    midas_path = os.path.expanduser("~/.MIDAS")
    libpth = os.environ.get('LD_LIBRARY_PATH', '')
    env['LD_LIBRARY_PATH'] = f'{midas_path}/BLOSC/lib64:{midas_path}/FFTW/lib:{midas_path}/HDF5/lib:{midas_path}/LIBTIFF/lib:{midas_path}/LIBZIP/lib64:{midas_path}/NLOPT/lib:{midas_path}/ZLIB/lib:{libpth}'
    
    # Count lines in SpotsToIndex.csv
    try:
        with open("SpotsToIndex.csv", "r") as f:
            num_lines = len(f.readlines())
    except Exception as e:
        error_msg = f"Failed to read SpotsToIndex.csv: {e}"
        print(error_msg, file=sys.stderr)
        raise RuntimeError(error_msg)
    
    outfile = f'{resultDir}/output/indexing_out{blockNr}.csv'
    errfile = f'{resultDir}/output/indexing_err{blockNr}.csv'
    
    with open(outfile, 'w') as f, open(errfile, 'w') as f_err:
        cmd = f"{os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/IndexerOMP')} paramstest.txt {blockNr} {numBlocks} {num_lines} {numProcs}"
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            env=env, 
            stdout=f, 
            stderr=f_err, 
            cwd=resultDir
        )
        returncode = process.wait()
        
        if returncode != 0:
            f_err.flush()
            with open(errfile, 'r') as err_reader:
                error_content = err_reader.read()
            error_msg = f"Indexing failed with return code {returncode}. Error output:\n{error_content}"
            print(error_msg, file=sys.stderr)
            raise RuntimeError(error_msg)

@python_app
def refine(resultDir: str, numProcs: int, blockNr: int = 0, numBlocks: int = 1) -> None:
    """Run refinement.
    
    Args:
        resultDir: Result directory
        numProcs: Number of processors
        blockNr: Block number
        numBlocks: Number of blocks
    """
    import subprocess
    import os
    import sys
    
    os.chdir(resultDir)
    
    # Set up environment
    env = dict(os.environ)
    midas_path = os.path.expanduser("~/.MIDAS")
    libpth = os.environ.get('LD_LIBRARY_PATH', '')
    env['LD_LIBRARY_PATH'] = f'{midas_path}/BLOSC/lib64:{midas_path}/FFTW/lib:{midas_path}/HDF5/lib:{midas_path}/LIBTIFF/lib:{midas_path}/LIBZIP/lib64:{midas_path}/NLOPT/lib:{midas_path}/ZLIB/lib:{libpth}'
    
    # Count lines in SpotsToIndex.csv
    try:
        with open("SpotsToIndex.csv", "r") as f:
            num_lines = len(f.readlines())
    except Exception as e:
        error_msg = f"Failed to read SpotsToIndex.csv: {e}"
        print(error_msg, file=sys.stderr)
        raise RuntimeError(error_msg)
    
    outfile = f'{resultDir}/output/refining_out{blockNr}.csv'
    errfile = f'{resultDir}/output/refining_err{blockNr}.csv'
    
    with open(outfile, 'w') as f, open(errfile, 'w') as f_err:
        cmd = f"{os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/FitPosOrStrainsOMP')} paramstest.txt {blockNr} {numBlocks} {num_lines} {numProcs}"
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            env=env, 
            stdout=f, 
            stderr=f_err, 
            cwd=resultDir
        )
        returncode = process.wait()
        
        if returncode != 0:
            f_err.flush()
            with open(errfile, 'r') as err_reader:
                error_content = err_reader.read()
            error_msg = f"Refinement failed with return code {returncode}. Error output:\n{error_content}"
            print(error_msg, file=sys.stderr)
            raise RuntimeError(error_msg)

# Signal handler for cleanup
default_handler = None

def handler(num, frame):
    """Handle Ctrl+C by cleaning up and exiting."""
    try:
        subprocess.call("rm -rf /dev/shm/*.bin", shell=True)
        logger.info("Ctrl-C was pressed, cleaning up.")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
    finally:
        return default_handler(num, frame)


class MyParser(argparse.ArgumentParser):
    """Custom argument parser with better error handling."""
    def error(self, message):
        """Print error message and exit."""
        sys.stderr.write(f'error: {message}\n')
        self.print_help()
        sys.exit(2)


def update_parameter_file(psFN: str, updates: Dict[str, str]) -> None:
    """Update parameter file with new values.
    
    Args:
        psFN: Parameter file name
        updates: Dictionary of parameter names and values to update
    """
    try:
        psContents = open(psFN, 'r').readlines()
        with open(psFN, 'w') as psF:
            for line in psContents:
                parameter = line.strip().split(' ')[0] if ' ' in line else ''
                if parameter in updates:
                    psF.write(f"{parameter} {updates[parameter]}\n")
                else:
                    psF.write(line)
    except Exception as e:
        logger.error(f"Failed to update parameter file: {e}")
        raise


def main():
    """Main function to process data."""
    # Set up signal handler
    global default_handler
    default_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handler)
    
    # Set up argument parser
    parser = MyParser(
        description='Far-field HEDM analysis using MIDAS. V7.0.0, contact hsharma@anl.gov', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add arguments
    parser.add_argument('-resultFolder', type=str, required=False, default='', 
                        help='Folder where you want to save results. If nothing is provided, it will default to the current folder.')
    parser.add_argument('-paramFN', type=str, required=False, default='', 
                        help='Parameter file name. Provide either paramFN and/or dataFN.')
    parser.add_argument('-dataFN', type=str, required=False, default='', 
                        help='Data file name. This is if you have either h5 or zip files. Provide either paramFN and/or dataFN (in case zip exists).')
    parser.add_argument('-nCPUs', type=int, required=False, default=10, 
                        help='Number of CPU cores to use if running locally.')
    parser.add_argument('-machineName', type=str, required=False, default='local', 
                        help='Machine name for execution, local, orthrosnew, orthrosall, umich, marquette, purdue.')
    parser.add_argument('-numFrameChunks', type=int, required=False, default=-1, 
                        help='If low on RAM, it can process parts of the dataset at the time. -1 will disable.')
    parser.add_argument('-preProcThresh', type=int, required=False, default=-1, 
                        help='If want to save the dark corrected data, then put to whatever threshold wanted above dark. -1 will disable. 0 will just subtract dark. Negative values will be reset to 0.')
    parser.add_argument('-nNodes', type=int, required=False, default=-1, 
                        help='Number of nodes for execution, omit if want to automatically select.')
    parser.add_argument('-fileName', type=str, required=False, default='', 
                        help='If you specify a fileName, this will run just that file. If you provide this, it will override startLayerNr and endLayerNr')
    parser.add_argument('-startLayerNr', type=int, required=False, default=1, 
                        help='Start LayerNr to process')
    parser.add_argument('-endLayerNr', type=int, required=False, default=1, 
                        help='End LayerNr to process')
    parser.add_argument('-convertFiles', type=int, required=False, default=1, 
                        help='If want to convert to zarr, if zarr files exist already, put to 0.')
    parser.add_argument('-peakSearchOnly', type=int, required=False, default=0, 
                        help='If want to do peakSearchOnly, nothing more, put to 1.')
    parser.add_argument('-doPeakSearch', type=int, required=False, default=1, 
                        help="If don't want to do peakSearch, put to 0.")
    parser.add_argument('-provideInputAll', type=int, required=False, default=0, 
                        help="If want to provide InputAllExtraInfoFittingAll.csv, put to 1. MUST provide all the parameters in the paramFN. The resultFolder must exist and contain the InputAlExtraInfoFittingAll.csv")
    parser.add_argument('-rawDir', type=str, required=False, default='', 
                        help='If want override the rawDir in the Parameter file.')
    
    # Parse arguments
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        logger.error("MUST PROVIDE EITHER paramFN or dataFN")
        sys.exit(1)
        
    args, unparsed = parser.parse_known_args()
    
    # Set variables from arguments
    resultDir = args.resultFolder
    psFN = args.paramFN
    dataFN = args.dataFN
    numProcs = args.nCPUs
    machineName = args.machineName
    nNodes = args.nNodes
    nchunks = args.numFrameChunks
    preproc = args.preProcThresh
    startLayerNr = args.startLayerNr
    endLayerNr = args.endLayerNr
    ConvertFiles = args.convertFiles
    peakSearchOnly = args.peakSearchOnly
    DoPeakSearch = args.doPeakSearch
    rawDir = args.rawDir
    inpFileName = args.fileName
    ProvideInputAll = args.provideInputAll
    
    # Handle input file name
    if len(inpFileName) > 1 and len(dataFN) < 1 and '.h5' in inpFileName:
        dataFN = inpFileName
        
    # Set number of nodes
    if nNodes == -1:
        nNodes = 1
        
    # Set defaults if neither paramFN nor dataFN is provided
    if not psFN and not dataFN:
        logger.error("Either paramFN or dataFN must be provided")
        sys.exit(1)
        
    # Update raw directory if provided
    if len(rawDir) > 1 and psFN:
        try:
            psContents = open(psFN, 'r').readlines()
            updates = {}
            
            for line in psContents:
                if line.startswith('OverAllRingToIndex'):
                    ring2Index = float(line.split(' ')[1])
                if line.startswith('MinOmeSpotIDsToIndex'):
                    min2Index = float(line.split(' ')[1])
                if line.startswith('MaxOmeSpotIDsToIndex'):
                    max2Index = float(line.split(' ')[1])
                    
            # Update RawFolder and Dark
            updates['RawFolder'] = rawDir
            
            # Find the dark file name and update its path
            for line in psContents:
                if line.startswith('Dark'):
                    darkName = line.strip().split(' ')[1].split('/')[-1]
                    updates['Dark'] = f'{rawDir}/{darkName}'
                    break
                    
            update_parameter_file(psFN, updates)
            
        except Exception as e:
            logger.error(f"Failed to update raw directory: {e}")
            sys.exit(1)
    
    # Read parameter file parameters
    if psFN:
        try:
            psContents = open(psFN, 'r').readlines()
            for line in psContents:
                if line.startswith('OverAllRingToIndex'):
                    ring2Index = float(line.split(' ')[1])
                if line.startswith('MinOmeSpotIDsToIndex'):
                    min2Index = float(line.split(' ')[1])
                if line.startswith('MaxOmeSpotIDsToIndex'):
                    max2Index = float(line.split(' ')[1])
        except Exception as e:
            logger.error(f"Failed to read parameter file: {e}")
            sys.exit(1)
    
    # Update parameters for specific input file
    if len(inpFileName) > 1:
        try:
            ext = '.' + '.'.join(inpFileName.split('_')[-1].split('.')[1:])
            filestem = '_'.join(inpFileName.split('_')[:-1])
            fileNr = int(inpFileName.split('_')[-1].split('.')[0])
            startLayerNr = fileNr
            endLayerNr = fileNr
            padding = len(inpFileName.split('_')[-1].split('.')[0])
            inpFSTM = inpFileName.split('.')[0]
            output_dir_stem = f'analysis_{inpFSTM}'
            
            updates = {
                'Ext': ext,
                'FileStem': filestem,
                'StartFileNrFirstLayer': '1'
            }
            
            update_parameter_file(psFN, updates)
            
        except Exception as e:
            logger.error(f"Failed to update parameters for input file: {e}")
            sys.exit(1)
    
    # Set up environment
    env = get_midas_env()
    
    # Set up result directory
    if len(resultDir) == 0 or resultDir == '.':
        resultDir = os.getcwd()
    if resultDir[0] == '~':
        resultDir = os.path.expanduser(resultDir)
    if resultDir[0] != '/':
        resultDir = os.getcwd() + '/' + resultDir
        
    os.makedirs(resultDir, exist_ok=True)
    os.environ['MIDAS_SCRIPT_DIR'] = resultDir
    
    # Load configuration based on machine name
    try:
        if machineName == 'local':
            nNodes = 1
            import localConfig
            parsl.load(config=localConfig.localConfig)
        elif machineName == 'orthrosnew':
            numProcs = 32
            nNodes = 11
            import orthrosAllConfig
            parsl.load(config=orthrosAllConfig.orthrosNewConfig)
        elif machineName == 'orthrosall':
            numProcs = 64
            nNodes = 5
            import orthrosAllConfig
            parsl.load(config=orthrosAllConfig.orthrosAllConfig)
        elif machineName == 'umich':
            numProcs = 36
            os.environ['nNodes'] = str(nNodes)
            import uMichConfig
            parsl.load(config=uMichConfig.uMichConfig)
        elif machineName == 'marquette':
            numProcs = 36
            os.environ['nNodes'] = str(nNodes)
            import marquetteConfig
            parsl.load(config=marquetteConfig.marquetteConfig)
        elif machineName == 'purdue':
            numProcs = 128
            os.environ['nNodes'] = str(nNodes)
            import purdueConfig
            parsl.load(config=purdueConfig.purdueConfig)
        else:
            logger.error(f"Unknown machine name: {machineName}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Run for each layer
    origDir = os.getcwd()
    topResDir = resultDir
    
    for layerNr in range(startLayerNr, endLayerNr + 1):
        try:
            if len(inpFileName) <= 1:
                output_dir_stem = f'LayerNr_{layerNr}/'
            resultDir = f'{topResDir}/{output_dir_stem}'
            logger.info(f"Processing Layer Nr: {layerNr}, results will be saved in {resultDir}")
            
            logDir = resultDir + '/output'
            os.makedirs(resultDir, exist_ok=True)
            shutil.copy2(psFN, resultDir)
            os.makedirs(logDir, exist_ok=True)
            
            t0 = time.time()
            
            # Process based on input type
            if ProvideInputAll == 0:
                if ConvertFiles == 1:
                    if len(dataFN) > 0:
                        logger.info("Generating combined MIDAS file from HDF and ps files.")
                    else:
                        logger.info("Generating combined MIDAS file from GE and ps files.")
                    outFStem = generateZip(resultDir, psFN, layerNr, dfn=dataFN, nchunks=nchunks, preproc=preproc)
                    if not outFStem:
                        logger.error("Failed to generate ZIP file")
                        sys.exit(1)
                else:
                    if len(dataFN) > 0:
                        outFStem = f'{resultDir}/{dataFN}'
                        if not os.path.exists(outFStem):
                            shutil.copy2(dataFN, resultDir)
                    else:
                        # Extract file information from parameter file
                        psContents = open(psFN).readlines()
                        fStem = None
                        startFN = None
                        NrFilerPerLayer = None
                        
                        for line in psContents:
                            if line.startswith('FileStem '):
                                fStem = line.split()[1]
                            if line.startswith('StartFileNrFirstLayer '):
                                startFN = int(line.split()[1])
                            if line.startswith('NrFilesPerSweep '):
                                NrFilerPerLayer = int(line.split()[1])
                                
                        if not all([fStem, startFN, NrFilerPerLayer]):
                            logger.error("Missing required parameters in parameter file")
                            sys.exit(1)
                            
                        thisFileNr = startFN + (layerNr - 1) * NrFilerPerLayer
                        outFStem = f'{resultDir}/{fStem}_{str(thisFileNr).zfill(6)}.MIDAS.zip'
                        
                        if not os.path.exists(outFStem) and dataFN:
                            shutil.copy2(dataFN, resultDir)
                            
                    # Update zarr dataset
                    cmdUpd = f'{pytpath} {os.path.expanduser("~/opt/MIDAS/utils/updateZarrDset.py")} -fn {os.path.basename(outFStem)} -folder {resultDir} -keyToUpdate analysis/process/analysis_parameters/ResultFolder -updatedValue {resultDir}/'
                    logger.info(cmdUpd)
                    
                    try:
                        subprocess.check_call(cmdUpd, shell=True)
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Failed to update zarr dataset: {e}")
                        sys.exit(1)
                        
                logger.info(f"Generating HKLs. Time till now: {time.time() - t0} seconds.")
                
                try:
                    f_hkls_out = f'{logDir}/hkls_out.csv'
                    f_hkls_err = f'{logDir}/hkls_err.csv'
                    cmd = f"{os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/GetHKLListZarr')} {outFStem}"
                    run_command(cmd, resultDir, f_hkls_out, f_hkls_err)
                except Exception as e:
                    logger.error(f"Failed to generate HKLs: {e}")
                    sys.exit(1)
            else:
                os.chdir(resultDir)
                logger.info(f"Generating HKLs. Time till now: {time.time() - t0} seconds.")
                
                try:
                    f_hkls_out = f'{logDir}/hkls_out.csv'
                    f_hkls_err = f'{logDir}/hkls_err.csv'
                    cmd = f"{os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/GetHKLList')} {psFN}"
                    run_command(cmd, resultDir, f_hkls_out, f_hkls_err)
                except Exception as e:
                    logger.error(f"Failed to generate HKLs: {e}")
                    sys.exit(1)
                    
            # Create temporary directory
            os.makedirs(f'{resultDir}/Temp', exist_ok=True)
            
            # Process peaks if required
            if ProvideInputAll == 0:
                if DoPeakSearch == 1:
                    logger.info(f"Doing PeakSearch. Time till now: {time.time() - t0} seconds.")
                    
                    try:
                        res = []
                        for nodeNr in range(nNodes):
                            res.append(peaks(resultDir, outFStem, numProcs, blockNr=nodeNr, numBlocks=nNodes))
                        outputs = [i.result() for i in res]
                        logger.info(f"PeakSearch done. Time till now: {time.time() - t0}")
                    except Exception as e:
                        logger.error(f"Failed during peak search: {e}")
                        sys.exit(1)
                else:
                    logger.info("Peaksearch results were supplied. Skipping peak search.")
                    
                if peakSearchOnly == 1:
                    continue
                    
                logger.info("Merging peaks.")
                
                try:
                    f_merge_out = f'{logDir}/merge_overlaps_out.csv'
                    f_merge_err = f'{logDir}/merge_overlaps_err.csv'
                    cmd = f"{os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/MergeOverlappingPeaksAllZarr')} {outFStem}"
                    run_command(cmd, resultDir, f_merge_out, f_merge_err)
                except Exception as e:
                    logger.error(f"Failed to merge peaks: {e}")
                    sys.exit(1)
                
                logger.info(f"Calculating Radii. Time till now: {time.time() - t0}")
                
                try:
                    f_radius_out = f'{logDir}/calc_radius_out.csv'
                    f_radius_err = f'{logDir}/calc_radius_err.csv'
                    cmd = f"{os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/CalcRadiusAllZarr')} {outFStem}"
                    run_command(cmd, resultDir, f_radius_out, f_radius_err)
                except Exception as e:
                    logger.error(f"Failed to calculate radii: {e}")
                    sys.exit(1)
                
                logger.info(f"Transforming data. Time till now: {time.time() - t0}")
                
                try:
                    f_setup_out = f'{logDir}/fit_setup_out.csv'
                    f_setup_err = f'{logDir}/fit_setup_err.csv'
                    cmd = f"{os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/FitSetupZarr')} {outFStem}"
                    run_command(cmd, resultDir, f_setup_out, f_setup_err)
                except Exception as e:
                    logger.error(f"Failed to transform data: {e}")
                    sys.exit(1)
            else:
                # Handle InputAll data
                try:
                    os.chdir(resultDir)
                    shutil.copy2(f'{topResDir}/InputAllExtraInfoFittingAll.csv', f'{resultDir}/InputAll.csv')
                    shutil.copy2(f'{topResDir}/InputAllExtraInfoFittingAll.csv', f'{resultDir}/.')
                    
                    # Process spots
                    sps = np.genfromtxt(f'{resultDir}/InputAll.csv', skip_header=1)
                    sps_filt = sps[sps[:,5] == ring2Index,:]
                    
                    if len(sps_filt.shape) < 2:
                        error_msg = "No IDs could be identified for indexing due to no spots present for ring2index. Check param file and data"
                        logger.error(error_msg)
                        sys.exit(1)
                        
                    sps_filt2 = sps_filt[sps_filt[:,2] >= min2Index,:]
                    
                    if len(sps_filt2.shape) < 2:
                        error_msg = "No IDs could be identified for indexing due to no spots more than minOmeSpotsToIndex. Check param file and data"
                        logger.error(error_msg)
                        sys.exit(1)
                        
                    sps_filt3 = sps_filt2[sps_filt2[:,2] <= max2Index,:]
                    
                    if len(sps_filt3.shape) < 2:
                        error_msg = "No IDs could be identified for indexing due to no spots more than minOmeSpotsToIndex. Check param file and data"
                        logger.error(error_msg)
                        sys.exit(1)
                        
                    IDs = sps_filt3[:,4].astype(np.int32)
                    np.savetxt(f'{resultDir}/SpotsToIndex.csv', IDs, fmt="%d")
                    
                    # Copy and update paramstest.txt
                    shutil.copy2(f'{topResDir}/{psFN}', f'{resultDir}/paramstest.txt')                    
                    ringNrs = []
                    with open(f'{resultDir}/paramstest.txt', 'r') as f:
                        lines = f.readlines()
                    for line in lines:
                        if line.startswith('RingThresh '):
                            ringNrs.append(int(line.split(' ')[1]))
                    # What we need extra: RingRadii and RingNumbers, first read hkls.csv
                    ringRads = np.zeros((len(ringNrs),2))
                    hkls = np.genfromtxt(f'{resultDir}/hkls.csv',skip_header=1)
                    unq, locs = np.unique(hkls[:,4],return_index=True)
                    for rN in range(len(ringNrs)):
                        ringNr = ringNrs[rN]
                        for tp in range(len(unq)):
                            if ringNr == int(unq[tp]):
                                ringRads[rN] = np.array([ringNr,hkls[locs[tp],-1]])
                        
                    with open(f'{resultDir}/paramstest.txt', 'w') as paramstestF:
                        for nr in range(len(ringRads)):
                            paramstestF.write(f'RingRadii {ringRads[nr,1]}\n')
                            paramstestF.write(f'RingNumbers {int(ringRads[nr,0])}\n')
                        paramstestF.write(f'OutputFolder {resultDir}/Output\n')
                        paramstestF.write(f'ResultFolder {resultDir}/Results\n')
                        paramstestF.write('SpotsFileName InputAll.csv')
                        paramstestF.write('IDsFileName SpotsToIndex.csv')
                        paramstestF.write('RefinementFileName InputAllExtraInfoFittingAll.csv')
                        for line in lines:
                            paramstestF.write(line)
                                
                    os.makedirs(f'{resultDir}/Output', exist_ok=True)
                    os.makedirs(f'{resultDir}/Results', exist_ok=True)
                except Exception as e:
                    logger.error(f"Failed to process InputAll data: {e}")
                    sys.exit(1)

            # Change to result directory and bin data
            os.chdir(resultDir)
            logger.info(f"Binning data. Time till now: {time.time() - t0}, workingdir: {resultDir}")
            
            try:
                f_bin_out = f'{logDir}/binning_out.csv'
                f_bin_err = f'{logDir}/binning_err.csv'
                cmd = f"{os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/SaveBinData')}"
                run_command(cmd, resultDir, f_bin_out, f_bin_err)
            except Exception as e:
                logger.error(f"Failed to bin data: {e}")
                sys.exit(1)
                
            # Run indexing
            logger.info(f"Indexing. Time till now: {time.time() - t0}")
            
            try:
                resIndex = []
                for nodeNr in range(nNodes):
                    resIndex.append(index(resultDir, numProcs, blockNr=nodeNr, numBlocks=nNodes))
                outputIndex = [i.result() for i in resIndex]
            except Exception as e:
                logger.error(f"Failed during indexing: {e}")
                sys.exit(1)
                
            # Run refinement
            logger.info(f"Refining. Time till now: {time.time() - t0}")
            
            try:
                resRefine = []
                for nodeNr in range(nNodes):
                    resRefine.append(refine(resultDir, numProcs, blockNr=nodeNr, numBlocks=nNodes))
                outputRefine = [i.result() for i in resRefine]
            except Exception as e:
                logger.error(f"Failed during refinement: {e}")
                sys.exit(1)
                
            # Clean up shared memory
            try:
                subprocess.call("rm -rf /dev/shm/*.bin", shell=True)
            except Exception as e:
                logger.error(f"Failed to clean up shared memory: {e}")
                
            # Process grains
            logger.info(f"Making grains list. Time till now: {time.time() - t0}")
            
            try:
                f_grains_out = f'{logDir}/process_grains_out.csv'
                f_grains_err = f'{logDir}/process_grains_err.csv'
                
                if ProvideInputAll == 0:
                    cmd = f"{os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/ProcessGrainsZarr')} {outFStem}"
                else:
                    cmd = f"{os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/ProcessGrains')} {resultDir}/paramstest.txt"
                    
                run_command(cmd, resultDir, f_grains_out, f_grains_err)
            except Exception as e:
                logger.error(f"Failed to process grains: {e}")
                sys.exit(1)
                
            logger.info(f"Done Layer {layerNr}. Total time elapsed: {time.time() - t0}")
            os.chdir(origDir)
            
        except Exception as e:
            logger.error(f"Failed to process layer {layerNr}: {e}")
            sys.exit(1)
    
    logger.info("All layers processed successfully")


if __name__ == "__main__":
    main()