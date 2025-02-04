import h5py, sys, argparse ###, numpy
import os,sys
pytpath = sys.executable
import subprocess
import numpy as np
from multiprocessing import Pool

env = dict(os.environ)
midas_path = os.path.expanduser("~/.MIDAS")
libpth = os.environ.get('LD_LIBRARY_PATH')
env['LD_LIBRARY_PATH'] = f'{midas_path}/BLOSC/lib64:{midas_path}/FFTW/lib:{midas_path}/HDF5/lib:{midas_path}/LIBTIFF/lib:{midas_path}/LIBZIP/lib64:{midas_path}/NLOPT/lib:{midas_path}/ZLIB/lib:{libpth}'

def singleJob(fileNr):
    fn = f'{folder}/scan{str(fileNr).zfill(4)}/eiger_0000.h5'
    print(f"Processing dset: {fileNr}")
    dsetpath = f'/entry_0000/ESRF-ID11/eiger/data'
    thisResFolder = f'{resultFolder}/{fileNr}'
    os.makedirs(thisResFolder,exist_ok=True)
    fn2 = f'{thisResFolder}/{OutputFStem}_{str(fileNr).zfill(6)}.h5'
    f = h5py.File(fn2, 'w')
    link = h5py.ExternalLink(fn, dsetpath)
    f['exchange/data'] = link
    f['measurement/process/scan_parameters/startOmeOverride'] = omegaOverrides[fileNr-startScanNr]
    f.close()
    os.chdir(thisResFolder)
    f = open(f'{thisResFolder}/zip_out.txt','w')
    f_err = open(f'{thisResFolder}/zip_err.txt','w')
    zipPath = os.path.expanduser('~/opt/MIDAS/utils/ffGenerateZip.py')
    cmd = f'{pytpath} {zipPath} -resultFolder {thisResFolder} -paramFN {paramFN} -dataFN {fn2} -numFrameChunks {numFrameChunks} -preProcThresh {preProc}'
    print(cmd)
    subprocess.call(cmd,env=env,shell=True,stdout=f,stderr=f_err)
    f.close()
    f_err.close()
    os.chdir(basedir)


class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

parser = MyParser(description='''esrf2zip_pf.py''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-folder', type=str, required=True, help='Folder where data exists')
parser.add_argument('-resultFolder', type=str, required=True, help='Folder where you want to save files')
parser.add_argument('-startScanNr', type=int, required=False, default=1, help='First scanNr. Defaults to 1')
parser.add_argument('-lastScanNr', type=int, required=True, help='Last scanNr, it will always go from startScanNr to lastScanNr+1')
parser.add_argument('-outputFStem', type=str, required=True, help='Output filestem, the resulting files will be OutputFStem_XXXXXX.h5 etc in folder')
parser.add_argument('-paramFN', type=str, required=True, help='Output filestem, the resulting files will be OutputFStem_XXXXXX.h5 etc in folder')
parser.add_argument('-combinedH5FN', type=str, required=True, help='Name of the combined H5 file. The rotation angles are read from this file.')
parser.add_argument('-numFrameChunks', type=int, required=False, default=-1, help='Number of chunks to use when reading the data file if RAM is smaller than expanded data. -1 will disable.')
parser.add_argument('-preProcThresh', type=int, required=False, default=-1, help='If want to save the dark corrected data, then put to whatever threshold wanted above dark. -1 will disable. 0 will just subtract dark. Negative values will be reset to 0.')
parser.add_argument('-nCPUs', type=int, required=False, default=5, help='Number of CPUs to use')
args, unparsed = parser.parse_known_args()
folder = args.folder
resultFolder = args.resultFolder
LastScanNr = args.lastScanNr
startScanNr = args.startScanNr
OutputFStem = args.outputFStem
paramFN = args.paramFN
combinedH5FN = args.combinedH5FN
numFrameChunks = args.numFrameChunks
preProc = args.preProcThresh
numProcs = args.nCPUs
basedir = os.getcwd()

hf = h5py.File(combinedH5FN,'r')
omegaOverrides = np.array((LastScanNr-startScanNr+1))
for i in range(startScanNr,LastScanNr+1):
    dsetName = f'{i}.1/instrument/rot/data'
    omegaOverrides[i-startScanNr] = hf[dsetName][0]
hf.close()

work_data = [fileNr for fileNr in range(startScanNr,LastScanNr+1)]
p = Pool(numProcs)
p.map(singleJob,work_data)
