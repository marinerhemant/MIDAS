#!/usr/bin/env python

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore",SyntaxWarning)

import subprocess
import argparse
import os,sys
from pathlib import Path
import shutil
import re
import zarr
import fsspec
from multiprocessing import Pool
utilsDir = os.path.expanduser('~/opt/MIDAS/utils/')
sys.path.insert(0,utilsDir)
from midas2zip import Hdf5ToZarr
import warnings
warnings.filterwarnings('ignore')
pytpath = sys.executable

env = dict(os.environ)
midas_path = os.path.expanduser("~/.MIDAS")
libpth = os.environ.get('LD_LIBRARY_PATH','')
env['LD_LIBRARY_PATH'] = f'{midas_path}/BLOSC/lib64:{midas_path}/FFTW/lib:{midas_path}/HDF5/lib:{midas_path}/LIBTIFF/lib:{midas_path}/LIBZIP/lib64:{midas_path}/NLOPT/lib:{midas_path}/ZLIB/lib:{libpth}'

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

def generateZip(resFol,pfn,dfn='',nchunks=-1,preproc=-1,outf='ZipOut.txt',errf='ZipErr.txt'):
    cmd = pytpath+' '+os.path.expanduser('~/opt/MIDAS/utils/ffGenerateZip.py')+' -resultFolder '+ resFol +' -paramFN ' + pfn
    if len(darkFN) != 0:
        cmd += f' -darkFN {darkFN}'
    if dfn!='':
        cmd+= ' -dataFN ' + dfn
    if dloc!='':
        cmd+= ' -dataLoc ' + dloc
    if nchunks!=-1:
        cmd+= ' -numFrameChunks '+str(nchunks)
    if preproc!=-1:
        cmd+= ' -preProcThresh '+str(preproc)
    dfn_base = os.path.basename(dfn)
    outf = f'{resFol}/{logdir}/{dfn_base}_{outf}'
    errf = f'{resFol}/{logdir}/{dfn_base}_{errf}'
    subprocess.call(cmd,shell=True,stdout=open(outf,'w'),stderr=open(errf,'w'))
    lines = open(outf,'r').readlines()
    if lines[-1].startswith('OutputZipName'):
        return lines[-1].split()[1]

def runOneFile(fileNr):
    thisFN = InputFN.replace(startFileNrStr,str(startFileNr+fileNr).zfill(6))
    if fileNr > 0 and convertFiles == 1:
        zipFN = generateZip(resultDir,psFN,dfn=thisFN,nchunks=numFrameChunks,preproc=preProc)
    else:
        if thisFN[-3:] != 'zip':
            thisFN += '.analysis.MIDAS.zip'
        zipFN = resultDir + thisFN
    f = open(f'{resultDir}/{logdir}/{os.path.basename(zipFN)}_integrator_out.csv','w')
    f_err = open(f'{resultDir}/{logdir}/{os.path.basename(zipFN)}_integrator_err.csv','w')
    subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/IntegratorZarr")+f' {zipFN}',shell=True,env=env,stdout=f,stderr=f_err)
    f.close()
    f_err.close()
    finFN = f'{zipFN}.caked.hdf'
    outzip = finFN+'.zarr.zip'
    zipF = Path(outzip)
    if zipF.exists():
        shutil.move(outzip,outzip+'.old')
    with fsspec.open(finFN,mode='rb', anon=False, requester_pays=True,default_fill_cache=False) as f:
        storeZip = zarr.ZipStore(outzip)
        h5chunkszip = Hdf5ToZarr(f, storeZip)
        h5chunkszip.translate()
        storeZip.close()
    return outzip


parser = MyParser(description='''Code to integrate files. Contact: hsharma@anl.gov''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-resultFolder', type=str, required=False, default ='.', help='Folder where you want to save results.')
parser.add_argument('-paramFN', type=str, required=True, help='Parameter file name.')
parser.add_argument('-dataFN', type=str, required=True, default='', help='DataFileName for first file, this should have the full path if not in the current folder.')
parser.add_argument('-darkFN', type=str, required=False, default='', help='DarkFileName, full path.')
parser.add_argument('-dataLoc', type=str, required=False, default='exchange/data', help='Data location.')
parser.add_argument('-numFrameChunks', type=int, required=False, default=-1, help='Number of chunks to use when reading the data file if RAM is smaller than expanded data. -1 will disable.')
parser.add_argument('-preProcThresh', type=int, required=False, default=-1, help='If want to save the dark corrected data, then put to whatever threshold wanted above dark. -1 will disable. 0 will just subtract dark. Negative values will be reset to 0.')
parser.add_argument('-startFileNr', type=int, required=False, default=-1, help='Which fileNr to start from. Default is -1, which means that fileNr in dataFN is read.')
parser.add_argument('-endFileNr', type=int, required=False, default=-1, help='End fileNr. Default is -1, which means a single file is processed.')
parser.add_argument('-convertFiles', type=int, required=False, default=1, help='Whether want to convert files to ZarrZip format or not.')
parser.add_argument('-mapDetector', type=int, required=False, default=1, help='Whether want to generate map of detector or not. If unsure, put to 1. If already have the CORRECT Map.bin and nMap.bin, put it to 0.')
parser.add_argument('-nCPUs', type=int, required=False, default=1, help='If you want to use multiple CPUs.')
args, unparsed = parser.parse_known_args()
resultDir = args.resultFolder
psFN = args.paramFN
InputFN = args.dataFN
darkFN = args.darkFN
numFrameChunks = args.numFrameChunks
preProc = args.preProcThresh
startFileNr = args.startFileNr
endFileNr = args.endFileNr
convertFiles = args.convertFiles
mapDetector = args.mapDetector
nCPUs = args.nCPUs
dloc = args.dataLoc

if len(resultDir) == 0 or resultDir == '.':
    resultDir = os.getcwd()
if resultDir[0] != '/':
    resultDir = os.getcwd()+'/'+resultDir
resultDir += '/'
logdir = 'stdout'
os.makedirs(resultDir,exist_ok=True)
os.makedirs(f'{resultDir}/{logdir}',exist_ok=True)

startFileNrStr = str(startFileNr).zfill(6)
if startFileNr == -1:
    startFileNrStr = re.search(r'\d{% s}' % 6, InputFN)
    print(f'Processing file number: {int(startFileNrStr.group(0))}')
    if not startFileNrStr:
        print("Could not find 6 padded fileNr. Exiting.")
        sys.exit()
    startFileNrStr = startFileNrStr.group(0)
    startFileNr = int(startFileNrStr)
if endFileNr == -1:
    endFileNr = startFileNr
nrFiles = endFileNr - startFileNr + 1
fileNr = 0
thisFN = InputFN.replace(startFileNrStr,str(startFileNr+fileNr).zfill(6))
if convertFiles == 1:
    zipFN = generateZip(resultDir,psFN,dfn=thisFN,nchunks=numFrameChunks,preproc=preProc)
else:
    if thisFN[-3:] != 'zip':
        thisFN += '.analysis.MIDAS.zip'
    zipFN = resultDir + thisFN
    print(f'Processing file: {zipFN}')
if mapDetector == 1:
    f = open(f'{resultDir}/{logdir}/map_out.csv','w')
    f_err = open(f'{resultDir}/{logdir}/map_err.csv','w')
    subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/DetectorMapperZarr")+f' {zipFN}',shell=True,env=env,stdout=f,stderr=f_err)
    f.close()
    f_err.close()

# RUN THIS IN PARALLEL
if nCPUs == 1:
    for fileNr in range(0,nrFiles):
        thisFN = InputFN.replace(startFileNrStr,str(startFileNr+fileNr).zfill(6))
        if fileNr > 0 and convertFiles == 1:
            zipFN = generateZip(resultDir,psFN,dfn=thisFN,nchunks=numFrameChunks,preproc=preProc)
        else:
            if thisFN[-3:] != 'zip':
                thisFN += '.analysis.MIDAS.zip'
            zipFN = resultDir + os.path.basename(thisFN)
            print(f'Processing file: {zipFN}')
        f = open(f'{resultDir}/{logdir}/{os.path.basename(zipFN)}_integrator_out.csv','w')
        f_err = open(f'{resultDir}/{logdir}/{os.path.basename(zipFN)}_integrator_err.csv','w')
        subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/IntegratorZarr")+f' {zipFN}',shell=True,env=env,stdout=f,stderr=f_err)
        f.close()
        f_err.close()
        finFN = f'{zipFN}.caked.hdf'
        outzip = finFN+'.zarr.zip'
        zipF = Path(outzip)
        if zipF.exists():
            shutil.move(outzip,outzip+'.old')
        with fsspec.open(finFN,mode='rb', anon=False, requester_pays=True,default_fill_cache=False) as f:
            storeZip = zarr.ZipStore(outzip)
            h5chunkszip = Hdf5ToZarr(f, storeZip)
            h5chunkszip.translate()
            storeZip.close()
        print(f'Ouput file {outzip} tree structure:')
        print(zarr.open(outzip).tree())
else:
    work_data = [fileNr for fileNr in range(0,nrFiles)]
    print(f"Starting {nCPUs} parallel jobs.")
    p = Pool(nCPUs)
    res = p.map(runOneFile,work_data)
    for outzip in res:
        print(f'Ouput file {outzip} tree structure:')
        print(zarr.open(outzip).tree())
