import h5py, sys, argparse ###, numpy
import os,sys
pytpath = sys.executable
import subprocess
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
    f.close()
    os.chdir(thisResFolder)
    f = open(f'{thisResFolder}/zip_out.txt','w')
    f_err = open(f'{thisResFolder}/zip_err.txt','w')
    cmd = f'{pytpath} {os.path.expanduser('~/opt/MIDAS/utils/ffGenerateZip.py')} -resultFolder {thisResFolder} -paramFN {paramFN} -dataFN {fn2} -numFrameChunks {numFrameChunks} -preProcThresh {preProc}'
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
parser.add_argument('-lastScanNr', type=int, required=True, help='Last scanNr, it will always start from 1')
parser.add_argument('-outputFStem', type=str, required=True, help='Output filestem, the resulting files will be OutputFStem_XXXXXX.h5 etc in folder')
parser.add_argument('-paramFN', type=str, required=True, help='Output filestem, the resulting files will be OutputFStem_XXXXXX.h5 etc in folder')
parser.add_argument('-numFrameChunks', type=int, required=False, default=-1, help='Number of chunks to use when reading the data file if RAM is smaller than expanded data. -1 will disable.')
parser.add_argument('-preProcThresh', type=int, required=False, default=-1, help='If want to save the dark corrected data, then put to whatever threshold wanted above dark. -1 will disable. 0 will just subtract dark. Negative values will be reset to 0.')
parser.add_argument('-nCPUs', type=int, required=False, default=5, help='Number of CPUs to use')
args, unparsed = parser.parse_known_args()
folder = args.folder
resultFolder = args.resultFolder
LastScanNr = args.LastScanNr
OutputFStem = args.OutputFStem
paramFN = args.paramFN
numFrameChunks = args.numFrameChunks
preProc = args.preProcThresh
numProcs = args.nCPUs
basedir = os.getcwd()

work_data = [fileNr for fileNr in range(1,LastScanNr+1)]
p = Pool(numProcs)
p.map(singleJob,work_data)
