#!/usr/bin/env python

import subprocess
import numpy as np
import argparse
import warnings
import time
import os,sys,glob
from pathlib import Path
import shutil
import re
import zarr
import fsspec
utilsDir = os.path.expanduser('~/opt/MIDAS/utils/')
sys.path.insert(0,utilsDir)
from numba import jit
from midas2zip import Hdf5ToZarr
import warnings
warnings.filterwarnings('ignore')
pytpath = sys.executable

env = dict(os.environ)
midas_path = os.path.expanduser("~/.MIDAS")
env['LD_LIBRARY_PATH'] = f'{midas_path}/BLOSC/lib64:{midas_path}/FFTW/lib:{midas_path}/HDF5/lib:{midas_path}/LIBTIFF/lib:{midas_path}/LIBZIP/lib64:{midas_path}/NLOPT/lib:{midas_path}/ZLIB/lib'

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

def generateZip(resFol,pfn,dfn='',dloc='',nchunks=-1,preproc=-1,outf='ZipOut.txt',errf='ZipErr.txt'):
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
    outf = f'{resFol}/{logdir}/{outf}'
    errf = f'{resFol}/{logdir}/{errf}'
    subprocess.call(cmd,shell=True,stdout=open(outf,'w'),stderr=open(errf,'w'))
    lines = open(outf,'r').readlines()
    if lines[-1].startswith('OutputZipName'):
        return lines[-1].split()[1]

parser = MyParser(description='''Code to integrate files. Contact: hsharma@anl.gov''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-resultFolder', type=str, required=False, default ='.', help='Folder where you want to save results.')
parser.add_argument('-paramFN', type=str, required=True, help='Parameter file name.')
parser.add_argument('-dataFN', type=str, required=True, default='', help='DataFileName for first file, this should have the full path if not in the current folder.')
parser.add_argument('-darkFN', type=str, required=False, default='', help='DarkFileName, full path.')
parser.add_argument('-numFrameChunks', type=int, required=False, default=-1, help='Number of chunks to use when reading the data file if RAM is smaller than expanded data. -1 will disable.')
parser.add_argument('-preProcThresh', type=int, required=False, default=-1, help='If want to save the dark corrected data, then put to whatever threshold wanted above dark. -1 will disable. 0 will just subtract dark. Negative values will be reset to 0.')
parser.add_argument('-startFileNr', type=int, required=False, default=-1, help='Which fileNr to start from. Default is -1, which means that fileNr in dataFN is read.')
parser.add_argument('-endFileNr', type=int, required=False, default=-1, help='End fileNr. Default is -1, which means a single file is processed.')
parser.add_argument('-convertFiles', type=int, required=False, default=1, help='Whether want to convert files to ZarrZip format or not.')
parser.add_argument('-mapDetector', type=int, required=False, default=1, help='Whether want to generate map of detector or not. If unsure, put to 1. If already have the CORRECT Map.bin and nMap.bin, put it to 0.')
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

if len(resultDir) == 0 or resultDir == '.':
    resultDir = os.getcwd()
if resultDir[0] != '/':
    resultDir = os.getcwd()+'/'+resultDir
resultDir += '/'
logdir = 'stdout'
os.makedirs(resultDir,exist_ok=True)
os.makedirs(f'{resultDir}/{logdir}',exist_ok=True)

if startFileNr == -1:
    startFileNrStr = re.search('\d{% s}' % 6, InputFN)
    if not startFileNrStr:
        print("Could not find 6 padded fileNr. Exiting.")
        sys.exit()
    startFileNrStr = startFileNrStr.group(0)
    startFileNr = int(startFileNrStr)
if endFileNr == -1:
    endFileNr = startFileNr
nrFiles = endFileNr - startFileNr + 1
for fileNr in range(nrFiles):
    thisFN = InputFN.replace(startFileNrStr,str(startFileNr+fileNr).zfill(6))
    if convertFiles == 1:
        zipFN = generateZip(resultDir,psFN,dfn=thisFN,nchunks=numFrameChunks,preproc=preProc)
    else:
        if thisFN[-3:] != 'zip':
            thisFN += '.analysis.MIDAS.zip'
        zipFN = resultDir + thisFN
        print(f'Processing file: {zipFN}')
    if fileNr == 0 and mapDetector == 1:
        f = open(f'{resultDir}/{logdir}/map_out.csv','w')
        f_err = open(f'{resultDir}/{logdir}/map_err.csv','w')
        subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/DetectorMapperZarr")+f' {zipFN}',shell=True,env=env,stdout=f,stderr=f_err)
    # Now we run things
    f = open(f'{resultDir}/{logdir}/{os.path.basename(zipFN)}_integrator_out.csv','w')
    f_err = open(f'{resultDir}/{logdir}/{os.path.basename(zipFN)}_integrator_err.csv','w')
    subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/IntegratorZarr")+f' {zipFN}',shell=True,env=env,stdout=f,stderr=f_err)
    finFN = f'{zipFN}.caked.hdf'
    outzip = finFN+'.zip'
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