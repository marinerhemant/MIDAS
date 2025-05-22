#!/usr/bin/env python

import time
import parsl
import subprocess
import sys
import os
import argparse
import signal
import shutil
import logging
import numpy as np
from multiprocessing import Pool
from typing import List, Optional, Dict, Any, Union, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('MIDAS-NF')

# Set paths
utilsDir = os.path.expanduser('~/opt/MIDAS/utils/')
v7Dir = os.path.expanduser('~/opt/MIDAS/NF_HEDM/v7/')
binDir = os.path.expanduser('~/opt/MIDAS/NF_HEDM/bin/')
sys.path.insert(0, utilsDir)
sys.path.insert(0, v7Dir)
from parsl.app.app import python_app
pytpath = sys.executable

def median(psFN: str, distanceNr: int, logDir: str, resultFolder: str) -> None:
    """Run median calculation remotely.
    
    Args:
        psFN: Parameter file name
        distanceNr: Distance number
        logDir: Log directory
        resultFolder: Result folder
    """
    import os
    import subprocess
    f = open(f'{logDir}/median{distanceNr}_out.csv', 'w')
    f_err = open(f'{logDir}/median{distanceNr}_err.csv', 'w')
    cmd = os.path.expanduser("~/opt/MIDAS/NF_HEDM/bin/MedianImageLibTiff") + f' {psFN} {distanceNr}'
    f.write(cmd)
    subprocess.call(cmd, shell=True, stdout=f, stderr=f_err, cwd=resultFolder)
    f.close()
    f_err.close()

def median_local(distanceNr: int) -> int:
    """Run median calculation locally.
    
    Args:
        distanceNr: Distance number
        
    Returns:
        Success flag (1 for success)
    """
    import os
    import subprocess
    f = open(f'{logDir}/median{distanceNr}_out.csv', 'w')
    f_err = open(f'{logDir}/median{distanceNr}_err.csv', 'w')
    cmd = os.path.expanduser("~/opt/MIDAS/NF_HEDM/bin/MedianImageLibTiff") + f' {psFN} {distanceNr}'
    f.write(cmd)
    subprocess.call(cmd, shell=True, stdout=f, stderr=f_err, cwd=resultFolder)
    f.close()
    f_err.close()
    return 1

def image(psFN: str, nodeNr: int, nNodes: int, numProcs: int, logDir: str, resultFolder: str) -> None:
    """Run image processing.
    
    Args:
        psFN: Parameter file name
        nodeNr: Node number
        nNodes: Total number of nodes
        numProcs: Number of processors
        logDir: Log directory
        resultFolder: Result folder
    """
    import os
    import subprocess
    f = open(f'{logDir}/image{nodeNr}_out.csv', 'w')
    f_err = open(f'{logDir}/image{nodeNr}_err.csv', 'w')
    cmd = os.path.expanduser("~/opt/MIDAS/NF_HEDM/bin/ImageProcessingLibTiffOMP") + f' {psFN} {nodeNr} {nNodes} {numProcs}'
    f.write(cmd)
    subprocess.call(cmd, shell=True, stdout=f, stderr=f_err, cwd=resultFolder)
    f.close()
    f_err.close()

def fit(psFN: str, nodeNr: int, nNodes: int, numProcs: int, logDir: str, resultFolder: str) -> None:
    """Run orientation fitting.
    
    Args:
        psFN: Parameter file name
        nodeNr: Node number
        nNodes: Total number of nodes
        numProcs: Number of processors
        logDir: Log directory
        resultFolder: Result folder
    """
    import os
    import subprocess
    f = open(f'{logDir}/fit{nodeNr}_out.csv', 'w')
    f_err = open(f'{logDir}/fit{nodeNr}_err.csv', 'w')
    cmd = os.path.expanduser("~/opt/MIDAS/NF_HEDM/bin/FitOrientationOMP") + f' {psFN} {nodeNr} {nNodes} {numProcs}'
    f.write(cmd)
    subprocess.call(cmd, shell=True, stdout=f, stderr=f_err, cwd=resultFolder)
    f.close()
    f_err.close()

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
        f_out.write(cmd + "\n")
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=f_out, 
            stderr=f_err, 
            cwd=working_dir
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

# Signal handler for cleanup
default_handler = None

def handler(num, frame):
    """Handle Ctrl+C by cleaning up and exiting."""
    try:
        subprocess.call("rm -rf /dev/shm/*.bin", shell=True)
        logger.info("Ctrl-C was pressed, cleaning up.")
        # Add parsl cleanup
        parsl.dfk().cleanup()
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
    finally:
        return default_handler(num, frame)

def check_shared_memory_files() -> bool:
    """Check for existing /dev/shm/*.bin files from other users.
    
    Returns:
        True if no conflicting files found, False otherwise
    """
    try:
        # Get current user
        current_user = os.environ.get('USER', subprocess.check_output("whoami", shell=True).decode().strip())
        
        # Find all .bin files in /dev/shm
        bin_files = subprocess.check_output("ls -l /dev/shm/*.bin 2>/dev/null || true", shell=True).decode()
        
        # Check if there are any bin files not owned by current user
        if bin_files:
            other_user_files = False
            for line in bin_files.splitlines():
                if line and current_user not in line:
                    other_user_files = True
                    break
            
            if other_user_files:
                logger.error("Detected /dev/shm/*.bin files created by another user.")
                logger.error("These files may cause conflicts with the current process.")
                logger.error("Please have the other user clean up their /dev/shm/*.bin files or restart the system.")
                return False
    except Exception as e:
        logger.warning(f"Could not check for existing bin files: {e}")
    
    return True

class MyParser(argparse.ArgumentParser):
    """Custom argument parser with better error handling."""
    def error(self, message):
        """Print error message and exit."""
        sys.stderr.write(f'error: {message}\n')
        self.print_help()
        sys.exit(2)

def main():
    """Main function to process data."""
    global default_handler, psFN, logDir, resultFolder
    
    # Start timing
    t0 = time.time()
    
    # Check for existing shared memory files from other users
    if not check_shared_memory_files():
        sys.exit(1)
    
    # Set up signal handler
    default_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handler)
    
    # Set up argument parser
    parser = MyParser(
        description='''Near-field HEDM analysis using MIDAS. V7.0.0, contact hsharma@anl.gov
                      The machine MUST have write access to the DataDirectory.
                      The data is constructed from the parameter file as follows: DataDirectory/OrigFileName_XXXX.tif
                      nNodes or nCPUs must exceed number of distances.
                      ''', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add arguments
    parser.add_argument('-paramFN', type=str, required=False, default='', 
                        help='Parameter file name. Provide either paramFN and/or dataFN.')
    parser.add_argument('-nCPUs', type=int, required=False, default=10, 
                        help='Number of CPU cores to use if running locally.')
    parser.add_argument('-machineName', type=str, required=False, default='local', 
                        help='Machine name for execution, local, orthrosnew, orthrosall, umich, marquette, purdue.')
    parser.add_argument('-nNodes', type=int, required=False, default=1, 
                        help='Number of nodes for execution, omit if want to automatically select.')
    parser.add_argument('-startLayerNr', type=int, required=False, default=1, 
                        help='NOT IMPLEMENTED YET!!! Start LayerNr to process.')
    parser.add_argument('-endLayerNr', type=int, required=False, default=1, 
                        help='NOT IMPLEMENTED YET!!! End LayerNr to process. If Start and End'+
                            ' LayerNrs are equal, it will only process 1 layer, else will process multiple layers.')
    parser.add_argument('-ffSeedOrientations', type=int, required=False, default=0, 
                        help='If want to use seedOrientations from far-field results, provide 1 else 0. If 1, paramFN must have a parameter GrainsFile '+
                            'pointing to the location of the Grains.csv file. NEXT PART NOT IMPLEMENTED YET!!!!!'+
                            ' If put to 1, you can add the following parameters: FinalGridSize, FinalEdgeLength,'+
                            ' FullSeedFile, MinConfidenceLowerBound and MinConfidence to rerun analysis with all possible orientations.')
    parser.add_argument('-doImageProcessing', type=int, required=False, default=1, 
                        help='If want do ImageProcessing, put to 1, else 0. This is only for single layer processing.')
    
    # Parse arguments
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        logger.error("MUST PROVIDE paramFN")
        sys.exit(1)
        
    args, unparsed = parser.parse_known_args()
    
    # Set variables from arguments
    psFN = args.paramFN
    numProcs = args.nCPUs
    machineName = args.machineName
    nNodes = args.nNodes
    ffSeedOrientations = args.ffSeedOrientations
    doImageProcessing = args.doImageProcessing
    
    # Parse parameter file
    try:
        lines = open(psFN).readlines()
        tomoFN = ''
        GridMask = []
        
        for line in lines:
            if line.startswith('DataDirectory '):
                resultFolder = line.split(' ')[1].rstrip()
            elif line.startswith('OrigFileName '):
                origFileName = line.split(' ')[1].rstrip()
            elif line.startswith('RawStartNr '):
                firstNr = int(line.split(' ')[1])
            elif line.startswith('nDistances '):
                nDistances = int(line.split(' ')[1])
            elif line.startswith('ReducedFileName '):
                reducedName = line.split(' ')[1].rstrip()
            elif line.startswith('GrainsFile '):
                grainsFile = line.split(' ')[1].rstrip()
            elif line.startswith('SeedOrientations '):
                seedOrientations = line.split(' ')[1].rstrip()
            elif line.startswith('TomoImage '):
                tomoFN = line.split(' ')[1].rstrip()
            elif line.startswith('TomoPixelSize '):
                tomoPx = float(line.split(' ')[1])
            elif line.startswith('GridMask '):
                GridMask = [float(line.split(' ')[i+1]) for i in range(4)]
    except Exception as e:
        logger.error(f"Failed to parse parameter file: {e}")
        sys.exit(1)
    
    # Set up environment
    os.environ['MIDAS_SCRIPT_DIR'] = resultFolder
    
    # Load configuration based on machine name
    try:
        if machineName == 'local':
            nNodes = 1
            from localConfig import localConfig
            parsl.load(config=localConfig)
        elif machineName == 'orthrosnew':
            numProcs = 32
            nNodes = 11
            from orthrosAllConfig import orthrosNewConfig
            parsl.load(config=orthrosNewConfig)
        elif machineName == 'orthrosall':
            numProcs = 64
            nNodes = 5
            from orthrosAllConfig import orthrosAllConfig
            parsl.load(config=orthrosAllConfig)
        elif machineName == 'umich':
            numProcs = 36
            os.environ['nNodes'] = str(nNodes)
            from uMichConfig import uMichConfig
            parsl.load(config=uMichConfig)
        elif machineName == 'marquette':
            numProcs = 36
            os.environ['nNodes'] = str(nNodes)
            from marquetteConfig import marquetteConfig
            parsl.load(config=marquetteConfig)
        elif machineName == 'purdue':
            numProcs = 128
            os.environ['nNodes'] = str(nNodes)
            from purdueConfig import purdueConfig
            parsl.load(config=purdueConfig)
        else:
            logger.error(f"Unknown machine name: {machineName}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Create directories
    os.chdir(resultFolder)
    reducedFolder = os.path.dirname(f'{resultFolder}/{reducedName}')
    logDir = f'{resultFolder}/midas_log/'
    os.makedirs(reducedFolder, exist_ok=True)
    os.makedirs(logDir, exist_ok=True)
    
    # Generate HKLs
    logger.info("Making hkls.")
    try:
        run_command(
            cmd=os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/GetHKLListNF") + f' {psFN}',
            working_dir=resultFolder,
            out_file=f'{logDir}/hkls_out.csv',
            err_file=f'{logDir}/hkls_err.csv'
        )
    except Exception as e:
        logger.error(f"Failed to generate HKLs: {e}")
        sys.exit(1)
    logger.info(f"Time taken: {time.time() - t0:.2f} seconds.")
    
    # Generate seed orientations
    logger.info("Making seed orientations.")
    if ffSeedOrientations == 1:
        logger.info("Using far-field seed to generate orientation list.")
        try:
            run_command(
                cmd=os.path.expanduser("~/opt/MIDAS/NF_HEDM/bin/GenSeedOrientationsFF2NFHEDM") + f' {grainsFile} {seedOrientations}',
                working_dir=resultFolder,
                out_file=f'{logDir}/seed_out.csv',
                err_file=f'{logDir}/seed_err.csv'
            )
        except Exception as e:
            logger.error(f"Failed to generate seed orientations: {e}")
            sys.exit(1)
            
    # Update parameter file with number of orientations
    try:
        nrOrientations = len(open(seedOrientations).readlines())
        with open(psFN, 'a') as f_ps:
            f_ps.write(f'NrOrientations {nrOrientations}\n')
    except Exception as e:
        logger.error(f"Failed to update parameter file with orientation count: {e}")
        sys.exit(1)
    logger.info(f"Time taken: {time.time() - t0:.2f} seconds.")
    
    # Generate and filter reconstruction space
    logger.info("Making and filtering reconstruction space.")
    try:
        run_command(
            cmd=os.path.expanduser("~/opt/MIDAS/NF_HEDM/bin/MakeHexGrid") + f' {psFN}',
            working_dir=resultFolder,
            out_file=f'{logDir}/hex_out.csv',
            err_file=f'{logDir}/hex_err.csv'
        )
    except Exception as e:
        logger.error(f"Failed to make hex grid: {e}")
        sys.exit(1)
        
    # Apply tomo filter if specified
    if len(tomoFN) > 1:
        logger.info("Using tomo to filter reconstruction space.")
        try:
            run_command(
                cmd=os.path.expanduser("~/opt/MIDAS/NF_HEDM/bin/filterGridfromTomo") + f' {tomoFN} {tomoPx}',
                working_dir=resultFolder,
                out_file=f'{logDir}/tomo_out.csv',
                err_file=f'{logDir}/tomo_err.csv'
            )
            shutil.move('grid.txt', 'grid_unfilt.txt')
            shutil.move('gridNew.txt', 'grid.txt')
        except Exception as e:
            logger.error(f"Failed to filter grid from tomo: {e}")
            sys.exit(1)
    # Apply grid mask if specified
    elif len(GridMask) > 0:
        logger.info("Applying grid mask.")
        try:
            gridpoints = np.genfromtxt('grid.txt', skip_header=1, delimiter=' ')
            gridpoints = gridpoints[gridpoints[:,2] >= GridMask[0],:]
            gridpoints = gridpoints[gridpoints[:,2] <= GridMask[1],:]
            gridpoints = gridpoints[gridpoints[:,3] >= GridMask[2],:]
            gridpoints = gridpoints[gridpoints[:,3] <= GridMask[3],:]
            nrPoints = gridpoints.shape[0]
            logger.info(f'Filtered number of points: {nrPoints}')
            shutil.move('grid.txt', 'grid_old.txt')
            np.savetxt('grid.txt', gridpoints, fmt='%.6f', delimiter=' ', header=f'{nrPoints}', comments='')
        except Exception as e:
            logger.error(f"Failed to apply grid mask: {e}")
            sys.exit(1)
    logger.info(f"Time taken: {time.time() - t0:.2f} seconds.")
    
    # Make diffraction spots
    logger.info("Making simulated diffraction spots for input seed orientations.")
    try:
        run_command(
            cmd=os.path.expanduser("~/opt/MIDAS/NF_HEDM/bin/MakeDiffrSpots") + f' {psFN}',
            working_dir=resultFolder,
            out_file=f'{logDir}/spots_out.csv',
            err_file=f'{logDir}/spots_err.csv'
        )
    except Exception as e:
        logger.error(f"Failed to make diffraction spots: {e}")
        sys.exit(1)
    logger.info(f"Time taken: {time.time() - t0:.2f} seconds.")
    
    # Image processing
    if doImageProcessing == 1:
        logger.info("Processing images.")
        try:
            # Median calculation
            if machineName == 'local':
                logger.info("Computing median locally")
                p = Pool(nDistances)
                work_data = [i for i in range(1, nDistances + 1)]
                res = p.map(median_local, work_data)
            else:
                logger.info("Computing median remotely")
                resMedian = []
                for distanceNr in range(1, nDistances + 1):
                    resMedian.append(median(psFN, distanceNr, logDir, resultFolder))
                    
            # Image processing
            logger.info("Processing images")
            resImage = []
            for nodeNr in range(nNodes):
                resImage.append(image(psFN, nodeNr, nNodes, numProcs, logDir, resultFolder))
        except Exception as e:
            logger.error(f"Failed during image processing: {e}")
            sys.exit(1)
    logger.info(f"Time taken: {time.time() - t0:.2f} seconds.")
    
    # Map image info
    logger.info("Mapping image info etc.")
    try:
        run_command(
            cmd=os.path.expanduser("~/opt/MIDAS/NF_HEDM/bin/MMapImageInfo") + f' {psFN}',
            working_dir=resultFolder,
            out_file=f'{logDir}/map_out.csv',
            err_file=f'{logDir}/map_err.csv'
        )
        
        # Copy necessary files to shared memory
        shutil.copy2('SpotsInfo.bin', '/dev/shm/SpotsInfo.bin')
        shutil.copy2('DiffractionSpots.bin', '/dev/shm/DiffractionSpots.bin')
        shutil.copy2('Key.bin', '/dev/shm/Key.bin')
        shutil.copy2('OrientMat.bin', '/dev/shm/OrientMat.bin')
    except Exception as e:
        logger.error(f"Failed to map image info: {e}")
        sys.exit(1)
    logger.info(f"Time taken: {time.time() - t0:.2f} seconds.")
    
    # Fit orientations
    logger.info("Fitting orientations")
    try:
        resFit = []
        for nodeNr in range(nNodes):
            resFit.append(fit(psFN, nodeNr, nNodes, numProcs, logDir, resultFolder))
    except Exception as e:
        logger.error(f"Failed during orientation fitting: {e}")
        sys.exit(1)
    logger.info(f"Time taken: {time.time() - t0:.2f} seconds.")
    
    # Parse mic
    logger.info("Parsing mic.")
    try:
        run_command(
            cmd=os.path.expanduser("~/opt/MIDAS/NF_HEDM/bin/ParseMic") + f' {psFN}',
            working_dir=resultFolder,
            out_file=f'{logDir}/parse_out.csv',
            err_file=f'{logDir}/parse_err.csv'
        )
    except Exception as e:
        logger.error(f"Failed to parse mic: {e}")
        sys.exit(1)
    
    # Clean up
    logger.info("Cleaning up.")
    try:
        os.remove('/dev/shm/DiffractionSpots.bin')
        os.remove('/dev/shm/Key.bin')
        os.remove('/dev/shm/OrientMat.bin')
        os.remove('/dev/shm/SpotsInfo.bin')
    except Exception as e:
        logger.error(f"Failed during cleanup: {e}")
    
    logger.info(f"Total time taken: {time.time() - t0:.2f} seconds.")
    
    # Clean up parsl
    parsl.dfk().cleanup()

if __name__ == "__main__":
    main()