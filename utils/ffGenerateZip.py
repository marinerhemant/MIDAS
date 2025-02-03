#!/usr/bin/env python

#### Pameter file would contain the path to the temperature / pressure location in HDF5
### Use mean value of both temperature and pressure.

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore",SyntaxWarning)

import h5py
import numpy as np
import os
import sys
import argparse
from math import ceil
import hdf5plugin
import zarr
from numcodecs import Blosc
from pathlib import Path
import shutil
from numba import jit
import time
import matplotlib.pyplot as plt
import re
from PIL import Image

compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

def geReader(geFN,header=8192,numPxY=2048,numPxZ=2048,bytesPerPx=2):
    sz = os.path.getsize(geFN)
    nFrames = (sz-header) // (bytesPerPx*numPxY*numPxZ)
    print(sz,nFrames,header)
    return np.fromfile(geFN,dtype=np.uint16,offset=header,count=nFrames*numPxY*numPxZ).reshape((nFrames,numPxY,numPxZ))

class MyParser(argparse.ArgumentParser):
	def error(self, message):
		sys.stderr.write('error: %s\n' % message)
		self.print_help()
		sys.exit(2)

def addData (name,node):
    if node.name not in zRoot.keys():
        if isinstance(node,h5py.Dataset):
            print(f"Creating dataset: {node.name}")
            if node.shape==data.shape:
                print("Skipping writing data again.")
                return
            arr = node[()]
            if isinstance(arr,bytes):
                arr = np.bytes_(arr.decode('UTF-8'))
                za = zRoot.create_dataset(node.name,shape=(1,),dtype=arr.dtype,chunks=(1,),compression=compressor)
                za[:] = arr
            else:
                if arr.size == 1:
                    arr = np.array([arr])
                za = zRoot.create_dataset(node.name, shape=arr.shape,dtype=arr.dtype,chunks=arr.shape,compression=compressor)
                za[:] = arr
        else:
            print(f"Creating group: {node.name}")
            zRoot.create_group(node.name)

parser = MyParser(description='''Code to generate ZarrZip dataset from GE or HDF5 files.''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-resultFolder', type=str, required=True, help='Folder where you want to save results')
parser.add_argument('-paramFN', type=str, required=True, help='Parameter file name')
parser.add_argument('-dataFN', type=str, required=False, default='', help='DataFileName')
parser.add_argument('-darkFN', type=str, required=False, default='', help='DarkFileName')
parser.add_argument('-dataLoc', type=str, required=False, default='exchange/data', help='Location of data in the hdf file')
parser.add_argument('-numFrameChunks', type=int, required=False, default=-1, help='Number of chunks to use when reading the data file if RAM is smaller than expanded data. -1 will disable.')
parser.add_argument('-preProcThresh', type=int, required=False, default=-1, help='If want to save the dark corrected data, then put to whatever threshold wanted above dark. -1 will disable. 0 will just subtract dark. Negative values will be reset to 0.')
parser.add_argument('-numFilesPerScan', type=int, required=False, default=1, help='Number of files that constitute a single scan. This will combine multiple ge files into one dataset. 1 will disable.')
parser.add_argument('-LayerNr', type=int, required=False, default=1, help='LayerNr')
parser.add_argument('-correctSD', type=int, required=False, default=0, help='If you want to use an automatically computed threshold, put to 1. It will compute the standard deviation in the image, apply a threshold of 1.1*sigma. ***** WILL APPLY THIS ABOVE PREPROCTHRESH. USE WITH CAUTION *****')
args, unparsed = parser.parse_known_args()
resultDir = args.resultFolder
psFN = args.paramFN
InputFN = args.dataFN
darkFN = args.darkFN
numFrameChunks = args.numFrameChunks
dataLoc = args.dataLoc
preProc = args.preProcThresh
layerNr = args.LayerNr
numFilesPerScan = args.numFilesPerScan
doStd = args.correctSD

if resultDir == '.':
    resultDir = os.getcwd()

darkLoc = 'exchange/dark'
brightLoc = 'exchange/bright'
maskLoc = 'exchange/mask'
panelmaskLoc = 'exchange/panelmask'
lines = open(psFN).readlines()
skipF = 0
NrFilesPerSweep = 1
numPxY = 2048
numPxZ = 2048
HZ = 8192
pad = 6
maskFN = ''
for line in lines:
    if line.startswith('RawFolder '):
        rawFolder = line.split()[1]
    if line.startswith('Dark '):
        darkFN = line.split()[1]
    if line.startswith('DataLocation '):
        dataLoc = line.split()[1]
    if line.startswith('DarkLocation '):
        darkLoc = line.split()[1]
    if line.startswith('BrightLocation '):
        brightLoc = line.split()[1]
    if line.startswith('FileStem '):
        fStem = line.split()[1]
    if line.startswith('StartFileNrFirstLayer '):
        fNr = int(line.split()[1])
    if line.startswith('NrFilesPerSweep '):
        NrFilesPerSweep = int(line.split()[1])
    if line.startswith('NrPixelsY '):
        numPxY = int(line.split()[1])
    if line.startswith('NrPixelsZ '):
        numPxZ = int(line.split()[1])
    if line.startswith('NrPixels '):
        numPxZ = int(line.split()[1])
        numPxY = numPxZ
    if line.startswith('Padding '):
        pad = int(line.split()[1])
    if line.startswith('MaskFN '):
        maskFN = line.split()[1]
    if line.startswith('Ext '):
        ext = line.split()[1]
    if line.startswith('HeadSize '):
        HZ = int(line.split()[1])
    if line.startswith('SkipFrame '):
        skipF = int(line.split()[1])
    searchStr = 'OmegaFirstFile'
    if line.startswith(f'{searchStr} '):
        omegF = float(line.split()[1])
    searchStr = 'OmegaStart'
    if line.startswith(f'{searchStr} '):
        omegF = float(line.split()[1])

if skipF==0 and HZ > 8192:
    skipF = (HZ-8192) // (2*numPxY*numPxZ)
    HZ = 8192
elif skipF > 0 and HZ > 8192:
    HZ = 8192

origInputFN = InputFN
if len(InputFN)==0:
    fNr += (layerNr-1)*NrFilesPerSweep
    fNr = str(fNr)
    InputFN = rawFolder + '/' + fStem + '_' + fNr.zfill(pad) + ext
    outfn = resultDir + '/' + fStem + '_' + fNr.zfill(pad)
else:
    outfn = resultDir+'/'+os.path.basename(InputFN)+'.analysis'
    if len(darkFN) == 0:
        darkFN = InputFN
print(f'Input: {InputFN}')
print(f'Dark: {darkFN}')

@jit(nopython=True)
def applyCorrectionNumba(img,dark,darkpreproc,doStd):
    result = np.empty(img.shape,dtype=np.uint16)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                if (img[i,j,k] < darkpreproc[j,k]): result[i,j,k] = 0
                else: result[i,j,k] = img[i,j,k] - dark[j,k]
    if doStd == 1:
        for i in range(img.shape[0]):
            stdVal = np.std(result[i])
            for j in range(img.shape[1]):
                for k in range(img.shape[2]):
                    if (img[i,j,k] < int(1.1*stdVal)): result[i,j,k] = 0
    return result

print(f'ResultDir: {resultDir}')
print(f'Out: {outfn}.MIDAS.zip')
outfZip = f'{outfn}.MIDAS.zip'
if Path(outfZip).exists():
    shutil.move(outfZip,outfZip+'.old')
zipStore = zarr.ZipStore(outfZip)
zRoot = zarr.group(store=zipStore, overwrite=True)
exc = zRoot.create_group('exchange')
meas = zRoot.create_group('measurement')
pro_meas = meas.create_group('process')
sp_pro_meas = pro_meas.create_group('scan_parameters')
analysis = zRoot.create_group('analysis')
pro_analysis = analysis.create_group('process')
sp_pro_analysis = pro_analysis.create_group('analysis_parameters')
if h5py.is_hdf5(InputFN):
    hf2 = h5py.File(InputFN,'r')
    nFrames,numZ,numY = hf2[dataLoc].shape
    print(hf2[dataLoc].shape,dataLoc)
    if h5py.is_hdf5(darkFN) and darkFN != InputFN:
        print(f"We are going to read dark from a different HDF. Please make sure that the dark file contains information in {dataLoc} dataset.")
        hfDark = h5py.File(darkFN,'r')
        darkData = hfDark[dataLoc][()]
        print(darkData.shape,dataLoc)
        hfDark.close()
    else:
        if darkLoc in hf2:
            darkData = hf2[darkLoc][()]
        else:
            darkData = np.zeros((10,numZ,numY))
    print(darkData.shape)
    dark = exc.create_dataset('dark',shape=darkData.shape,dtype=np.uint16,chunks=(1,darkData.shape[1],darkData.shape[2]),compression=compressor)
    darkMean = np.mean(darkData[skipF:,:,:],axis=0).astype(np.uint16)
    if preProc!=-1:
        darkpreProc = darkMean + preProc
    if maskLoc in hf2:
        maskData = hf2[maskLoc][()].reshape((1,numZ,numY))
        mask = exc.create_dataset('mask',shape=maskData.shape,dtype=np.uint16,chunks=(1,numZ,numY),compressor=compressor)
        mask[:] = maskData
    if panelmaskLoc in hf2:
        panelmaskData = hf2[panelmaskLoc][()].reshape((1,numZ,numY))
        panelmask = exc.create_dataset('panelmask',shape=maskData.shape,dtype=np.uint16,chunks=(1,numZ,numY),compressor=compressor)
        panelmask[:] = panelmaskData
    if brightLoc in hf2:
        brightData = hf2[brightLoc][()]
    else:
        brightData = np.copy(darkData)
    bright = exc.create_dataset('bright',shape=brightData.shape,dtype=np.uint16,chunks=(1,brightData.shape[1],brightData.shape[2]),compression=compressor)
    if numFrameChunks == -1:
        numFrameChunks = nFrames
    numChunks = int(ceil(nFrames/numFrameChunks))
    data = exc.create_dataset('data',shape=(nFrames,numZ,numY),dtype=np.uint16,chunks=(1,numZ,numY),compression=compressor)
    for i in range(numChunks):
        stFrame = i*numFrameChunks
        enFrame = (i+1)*numFrameChunks
        if enFrame > nFrames: enFrame=nFrames
        print(f"StartFrame: {stFrame}, EndFrame: {enFrame-1}, nFrames: {nFrames}")
        dataThis = hf2[dataLoc][stFrame:enFrame,:,:]
        if preProc!=-1:
            dataT = applyCorrectionNumba(dataThis,darkMean,darkpreProc,doStd)
        else:
            dataT = dataThis
        data[stFrame:enFrame,:,:] = dataT
    searchStr = 'startOmeOverride'
    if searchStr in hf2:
        stOmeOver = sp_pro_meas.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        stOmeOver[:] = hf2[searchStr][()][0]
    searchStr = '/measurement/instrument/GSAS2_PVS/Pressure'
    if searchStr in hf2:
        print(f'Pressure values were found, will enter {nFrames} values!')
        pressureDSet = sp_pro_meas.create_dataset(searchStr.split('/')[-1],dtype=np.double,shape=(nFrames,),chunks=(nFrames,),compressor=compressor)
        pressureDSet[:] = hf2[searchStr][()]
    searchStr = '/measurement/instrument/GSAS2_PVS/Temperature'
    if searchStr in hf2:
        pressureDSet = sp_pro_meas.create_dataset(searchStr.split('/')[-1],dtype=np.double,shape=(nFrames,),chunks=(nFrames,),compressor=compressor)
        pressureDSet[:] = hf2[searchStr][()]
    searchStr = '/measurement/instrument/GSAS2_PVS/I'
    if searchStr in hf2:
        pressureDSet = sp_pro_meas.create_dataset(searchStr.split('/')[-1],dtype=np.double,shape=(nFrames,),chunks=(nFrames,),compressor=compressor)
        pressureDSet[:] = hf2[searchStr][()]
    searchStr = '/measurement/instrument/GSAS2_PVS/I0'
    if searchStr in hf2:
        pressureDSet = sp_pro_meas.create_dataset(searchStr.split('/')[-1],dtype=np.double,shape=(nFrames,),chunks=(nFrames,),compressor=compressor)
        pressureDSet[:] = hf2[searchStr][()]
    hf2.close()
elif 'zip' in InputFN[-5:]:
    # This is a zarray dataset. Just copy over the data.
    zf = zarr.open(InputFN,'r')
    data_orig = zf[dataLoc]
    nFrames,numZ,numY = data_orig.shape
    # data = exc.create_dataset('data',shape=(nFrames,numZ,numY),dtype=np.uint16,chunks=(1,numZ,numY),compression=compressor)
    # data[:] = data_orig[:]
    zarr.copy(data_orig, exc, log=sys.stdout, if_exists='skip')
    data = exc['data']
    print(f'Data copied as: {exc.tree()}')
    darkData = np.zeros((10,numPxZ,numPxY))
    brightData = np.copy(darkData)
    dark = exc.create_dataset('dark',shape=darkData.shape,dtype=np.uint16,chunks=(1,darkData.shape[1],darkData.shape[2]),compression=compressor)
    bright = exc.create_dataset('bright',shape=darkData.shape,dtype=np.uint16,chunks=(1,darkData.shape[1],darkData.shape[2]),compression=compressor)
else:
    sz = os.path.getsize(InputFN)
    bytesPerPx = 2
    nFrames = (sz-HZ) // (bytesPerPx*numPxY*numPxZ)
    nFramesAll = nFrames*numFilesPerScan
    if darkFN != '':
        darkData = geReader(darkFN,header=HZ,numPxY=numPxY,numPxZ=numPxZ)
    else:
        darkData = np.zeros((10,numPxZ,numPxY))
    brightData = np.copy(darkData)
    dark = exc.create_dataset('dark',shape=darkData.shape,dtype=np.uint16,chunks=(1,darkData.shape[1],darkData.shape[2]),compression=compressor)
    bright = exc.create_dataset('bright',shape=darkData.shape,dtype=np.uint16,chunks=(1,darkData.shape[1],darkData.shape[2]),compression=compressor)
    darkMean = np.mean(darkData[skipF:,:,:],axis=0).astype(np.uint16)
    if preProc!=-1:
        darkpreProc = darkMean + preProc
    data = exc.create_dataset('data',shape=(nFramesAll,numPxZ,numPxY),dtype=np.uint16,chunks=(1,numPxZ,numPxY),compression=compressor)
    if numFrameChunks == -1:
        numFrameChunks = nFrames
    numChunks = int(ceil(nFrames/numFrameChunks))
    fNr = re.search(r'\d{% s}' % pad, InputFN).group(0)
    fNrOrig = fNr
    fNrLoc = int(fNr)
    for fileNrIter in range(numFilesPerScan):
        fNr = str(fNrLoc)
        if len(origInputFN) == 0:
            InputFN = rawFolder + '/' + fStem + '_' + fNr.zfill(pad) + ext
        else:
            InputFN = origInputFN.replace(fNrOrig,str(fNr).zfill(pad))
        print(InputFN)
        stNr = nFrames*fileNrIter
        for i in range(numChunks):
            stFrame = i*numFrameChunks
            enFrame = (i+1)*numFrameChunks
            if enFrame > nFrames: enFrame=nFrames
            print(f"StartFrame: {stFrame+stNr}, EndFrame: {enFrame+stNr}, nFrames: {nFrames}, nFramesAll: {nFramesAll}")
            delFrames = enFrame - stFrame
            dataThis = np.fromfile(InputFN,dtype=np.uint16,count=delFrames*numPxY*numPxZ,offset=stFrame*numPxY*numPxZ*bytesPerPx+HZ).reshape((delFrames,numPxZ,numPxY))
            if preProc!=-1:
                dataT = applyCorrectionNumba(dataThis,darkMean,darkpreProc,doStd)
            else:
                dataT = dataThis
            data[stFrame+stNr:enFrame+stNr,:,:] = dataT
        fNrLoc += 1
if preProc !=-1:
    darkData *= 0
    brightData *= 0
dark[:] = darkData
bright[:]=brightData
if len(maskFN) > 0:
    maskData = np.array(Image.open(maskFN)).astype(np.uint16)
    maskData = maskData.reshape((1,numZ,numY))
    print(maskData.shape)
    mask = exc.create_dataset('mask',shape=maskData.shape,dtype=np.uint16,chunks=(1,numZ,numY),compressor=compressor)
    mask[:] = maskData


data.attrs['_ARRAY_DIMENSIONS'] = data.shape
dark.attrs['_ARRAY_DIMENSIONS'] = bright.shape
bright.attrs['_ARRAY_DIMENSIONS'] = bright.shape

resultOut = np.bytes_(resultDir)
rf = zRoot.create_dataset('analysis/process/analysis_parameters/ResultFolder',shape=(1,),chunks=(1,),compressor=compressor,dtype=resultOut.dtype)
rf[:]=resultOut

RingThreshArr = np.zeros((1,2))
RingExcludeArr = np.zeros((1,2))
OmegaRanges = np.zeros((1,2))
OmegaRanges[0,0] = -10000
BoxSizes = np.zeros((1,4))
BoxSizes[0,0] = -10000
ImTransOpts = np.zeros((1))
ImTransOpts[0] = -1
skipF = 0
omeStp = 0
OmeFF = 0
for line in lines:
    searchStr = 'GapFile'
    if line.startswith(searchStr):
        gf = np.bytes_(line.split()[1])
        rf = sp_pro_analysis.create_dataset(searchStr,shape=(1,),chunks=(1,),compressor=compressor,dtype=gf.dtype)
        rf[:]=gf
    searchStr = 'BadPxFile'
    if line.startswith(searchStr):
        gf = np.bytes_(line.split()[1])
        rf = sp_pro_analysis.create_dataset(searchStr,shape=(1,),chunks=(1,),compressor=compressor,dtype=gf.dtype)
        rf[:]=gf
    searchStr = 'ImTransOpt'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([int(line.split()[1])]).astype(np.int32)
        if (ImTransOpts[0] == -1):
            ImTransOpts[0] = outArr[0]
        else:
            ImTransOpts = np.vstack((ImTransOpts,outArr))
    searchStr = 'BoxSize'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(x) for x in line.split()[1:5]]).astype(np.double)
        outArr = outArr.reshape((1,4))
        if BoxSizes[0,0] == -10000:
            BoxSizes = outArr
        else:
            BoxSizes = np.vstack((BoxSizes,outArr))
    searchStr = 'OmegaRange'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(x) for x in line.split()[1:3]]).astype(np.double)
        outArr = outArr.reshape((1,2))
        if OmegaRanges[0,0] == -10000:
            OmegaRanges = outArr
        else:
            OmegaRanges = np.vstack((OmegaRanges,outArr))
    searchStr = 'RingThresh'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(x) for x in line.split()[1:3]]).astype(np.double)
        outArr = outArr.reshape((1,2))
        if RingThreshArr[0,0] == 0:
            RingThreshArr = outArr
        else:
            RingThreshArr = np.vstack((RingThreshArr,outArr))
    # searchStr = 'RingsToExclude'
    # if line.startswith(f'{searchStr} '):
    #     outArr = np.array([float(x) for x in line.split()[1:3]]).astype(np.double)
    #     print(outArr)
    #     outArr = outArr.reshape((1,2))
    #     if RingExcludeArr[0,0] == 0:
    #         RingExcludeArr = outArr
    #     else:
    #         RingExcludeArr = np.vstack((RingExcludeArr,outArr))
    searchStr = 'HeadSize'
    if line.startswith(f'{searchStr} '):
        head = int(line.split()[1])
        if head > 8192:
            if skipF==0:
                skipF = (head-8192) // (2*numPxY*numPxZ)
                spsf = sp_pro_analysis.create_dataset('SkipFrame',dtype=np.int32,shape=(1,),chunks=(1,),compressor=compressor)
                spsf[:]=np.array([skipF]).astype(np.int32)
    searchStr = 'Twins'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([int(line.split()[1])]).astype(np.int32)
        spT = sp_pro_analysis.create_dataset(searchStr,dtype=np.int32,shape=(1,),chunks=(1,),compressor=compressor)
        spT[:]=outArr
    searchStr = 'MaxNFrames'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([int(line.split()[1])]).astype(np.int32)
        spMNF = sp_pro_analysis.create_dataset(searchStr,dtype=np.int32,shape=(1,),chunks=(1,),compressor=compressor)
        spMNF[:] = outArr
    searchStr = 'DoFit'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([int(line.split()[1])]).astype(np.int32)
        spDF = sp_pro_analysis.create_dataset(searchStr,dtype=np.int32,shape=(1,),chunks=(1,),compressor=compressor)
        spDF[:] = outArr
    searchStr = 'DiscModel'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([int(line.split()[1])]).astype(np.int32)
        spDM = sp_pro_analysis.create_dataset(searchStr,dtype=np.int32,shape=(1,),chunks=(1,),compressor=compressor)
        spDM[:] = outArr
    searchStr = 'UseMaximaPositions'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([int(line.split()[1])]).astype(np.int32)
        spUMP = sp_pro_analysis.create_dataset(searchStr,dtype=np.int32,shape=(1,),chunks=(1,),compressor=compressor)
        spUMP[:] = outArr
    searchStr = 'MaxNrPx'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([int(line.split()[1])]).astype(np.int32)
        spMaxNP = sp_pro_analysis.create_dataset(searchStr,dtype=np.int32,shape=(1,),chunks=(1,),compressor=compressor)
        spMaxNP[:] = outArr
    searchStr = 'MinNrPx'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([int(line.split()[1])]).astype(np.int32)
        spMinNP = sp_pro_analysis.create_dataset(searchStr,dtype=np.int32,shape=(1,),chunks=(1,),compressor=compressor)
        spMinNP[:] = outArr
    searchStr = 'MaxNPeaks'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([int(line.split()[1])]).astype(np.int32)
        spMaxNPeaks = sp_pro_analysis.create_dataset(searchStr,dtype=np.int32,shape=(1,),chunks=(1,),compressor=compressor)
        spMaxNPeaks[:] = outArr
    searchStr = 'PhaseNr'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([int(line.split()[1])]).astype(np.int32)
        spPhase = sp_pro_analysis.create_dataset(searchStr,dtype=np.int32,shape=(1,),chunks=(1,),compressor=compressor)
        spPhase[:] = outArr
    searchStr = 'NumPhases'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([int(line.split()[1])]).astype(np.int32)
        spNumPh = sp_pro_analysis.create_dataset(searchStr,dtype=np.int32,shape=(1,),chunks=(1,),compressor=compressor)
        spNumPh[:] = outArr
    searchStr = 'MinNrSpots'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([int(line.split()[1])]).astype(np.int32)
        spMinNrSp = sp_pro_analysis.create_dataset(searchStr,dtype=np.int32,shape=(1,),chunks=(1,),compressor=compressor)
        spMinNrSp[:] = outArr
    searchStr = 'UseFriedelPairs'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([int(line.split()[1])]).astype(np.int32)
        spUseFP = sp_pro_analysis.create_dataset(searchStr,dtype=np.int32,shape=(1,),chunks=(1,),compressor=compressor)
        spUseFP[:] = outArr
    searchStr = 'OverAllRingToIndex'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([int(line.split()[1])]).astype(np.int32)
        spRTI = sp_pro_analysis.create_dataset('OverallRingToIndex',dtype=np.int32,shape=(1,),chunks=(1,),compressor=compressor)
        spRTI[:] = outArr
    searchStr = 'SpaceGroup'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([int(line.split()[1])]).astype(np.int32)
        spSG = sp_pro_analysis.create_dataset(searchStr,dtype=np.int32,shape=(1,),chunks=(1,),compressor=compressor)
        spSG[:] = outArr
    searchStr = 'LayerNr'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([int(line.split()[1])]).astype(np.int32)
        spLN = sp_pro_analysis.create_dataset(searchStr,dtype=np.int32,shape=(1,),chunks=(1,),compressor=compressor)
        spLN[:] = outArr
    searchStr = 'DoFullImage'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([int(line.split()[1])]).astype(np.int32)
        spDFI = sp_pro_analysis.create_dataset(searchStr,dtype=np.int32,shape=(1,),chunks=(1,),compressor=compressor)
        spDFI[:] = outArr
    searchStr = 'SkipFrame'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([int(line.split()[1])]).astype(np.int32)
        skipF = int(line.split()[1])
        spSkipF = sp_pro_analysis.create_dataset(searchStr,dtype=np.int32,shape=(1,),chunks=(1,),compressor=compressor)
        spSkipF[:] = outArr
    searchStr = 'OmegaFirstFile'
    if line.startswith(f'{searchStr} '):
        OmeFF = float(line.split()[1])
    searchStr = 'OmegaStart'
    if line.startswith(f'{searchStr} '):
        OmeFF = float(line.split()[1])
    searchStr = 'OmegaStep'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        omeStp = float(line.split()[1])
        spStp = sp_pro_meas.create_dataset('step',dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spStp[:] = outArr
    searchStr = 'BadPxIntensity'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spBPI = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spBPI[:] = outArr
    searchStr = 'GapIntensity'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spBPI = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spBPI[:] = outArr
    searchStr = 'SumImages'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([int(line.split()[1])]).astype(np.int32)
        spBPI = sp_pro_analysis.create_dataset(searchStr,dtype=np.int32,shape=(1,),chunks=(1,),compressor=compressor)
        spBPI[:] = outArr
    searchStr = 'Normalize'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([int(line.split()[1])]).astype(np.int32)
        spBPI = sp_pro_analysis.create_dataset(searchStr,dtype=np.int32,shape=(1,),chunks=(1,),compressor=compressor)
        spBPI[:] = outArr
    searchStr = 'SaveIndividualFrames'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([int(line.split()[1])]).astype(np.int32)
        spBPI = sp_pro_analysis.create_dataset(searchStr,dtype=np.int32,shape=(1,),chunks=(1,),compressor=compressor)
        spBPI[:] = outArr
    searchStr = 'OmegaSumFrames'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([int(line.split()[1])]).astype(np.int32)
        spBPI = sp_pro_analysis.create_dataset(searchStr,dtype=np.int32,shape=(1,),chunks=(1,),compressor=compressor)
        spBPI[:] = outArr
    searchStr = 'FitWeightMean'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([int(line.split()[1])]).astype(np.double)
        spFWM = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spFWM[:] = outArr
    searchStr = 'PixelSplittingRBin'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([int(line.split()[1])]).astype(np.double)
        spPSRB = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spPSRB[:] = outArr
    searchStr = 'tolTilts'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spTolT = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spTolT[:] = outArr
    searchStr = 'tolBC'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spTolBC = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spTolBC[:] = outArr
    searchStr = 'tolLsd'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spTolL = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spTolL[:] = outArr
    searchStr = 'DiscArea'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spDA = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spDA[:] = outArr
    searchStr = 'OverlapLength'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spOLL = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spOLL[:] = outArr
    searchStr = 'ReferenceRingCurrent'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spRR = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spRR[:] = outArr
    searchStr = 'Completeness'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spCompleteness = sp_pro_analysis.create_dataset('MinMatchesToAcceptFrac',dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spCompleteness[:] = outArr
    searchStr = 'zDiffThresh'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spZDiff = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spZDiff[:] = outArr
    searchStr = 'GlobalPosition'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spGP = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spGP[:] = outArr
    searchStr = 'tolPanelFit'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spTPF = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spTPF[:] = outArr
    searchStr = 'tolP'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spTP = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spTP[:] = outArr
    searchStr = 'tolP0'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spTP0 = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spTP0[:] = outArr
    searchStr = 'tolP1'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spTP1 = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spTP1[:] = outArr
    searchStr = 'tolP2'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spTP2 = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spTP2[:] = outArr
    searchStr = 'tolP3'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spTP3 = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spTP3[:] = outArr
    searchStr = 'StepSizePos'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spSSP = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spSSP[:] = outArr
    searchStr = 'tInt'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        sptInt = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        sptInt[:] = outArr
    searchStr = 'tGap'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        sptGap = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        sptGap[:] = outArr
    searchStr = 'StepSizeOrient'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spSSO = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spSSO[:] = outArr
    searchStr = 'MarginRadius'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spMR = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spMR[:] = outArr
    searchStr = 'MarginRadial'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spMRad = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spMRad[:] = outArr
    searchStr = 'MarginEta'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spME = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spME[:] = outArr
    searchStr = 'MarginOme'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spMO = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spMO[:] = outArr
    searchStr = 'MargABG'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spMABG = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spMABG[:] = outArr
    searchStr = 'MargABC'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spMABC = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spMABC[:] = outArr
    searchStr = 'OmeBinSize'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spOBS = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spOBS[:] = outArr
    searchStr = 'EtaBinSize'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spEBS = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spEBS[:] = outArr
    searchStr = 'RBinSize'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spEBS = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spEBS[:] = outArr
    searchStr = 'RMin'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spEBS = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spEBS[:] = outArr
    searchStr = 'RMax'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spEBS = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spEBS[:] = outArr
    searchStr = 'EtaMin'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spEBS = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spEBS[:] = outArr
    searchStr = 'EtaMax'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spEBS = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spEBS[:] = outArr
    searchStr = 'X'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spEBS = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spEBS[:] = outArr
    searchStr = 'Y'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spEBS = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spEBS[:] = outArr
    searchStr = 'Z'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spEBS = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spEBS[:] = outArr
    searchStr = 'U'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spEBS = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spEBS[:] = outArr
    searchStr = 'V'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spEBS = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spEBS[:] = outArr
    searchStr = 'W'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spEBS = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spEBS[:] = outArr
    searchStr = 'SHpL'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spEBS = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spEBS[:] = outArr
    searchStr = 'Polariz'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spMEta = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spMEta[:] = outArr
    searchStr = 'MaxOmeSpotIDsToIndex'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spMaxOSII = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spMaxOSII[:] = outArr
    searchStr = 'MinOmeSpotIDsToIndex'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spMinOSII = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spMinOSII[:] = outArr
    searchStr = 'BeamThickness'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spBT = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spBT[:] = outArr
    searchStr = 'Wedge'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spW = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spW[:] = outArr
    searchStr = 'Rsample'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spRsam = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spRsam[:] = outArr
    searchStr = 'Hbeam'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spHBeam = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spHBeam[:] = outArr
    searchStr = 'Vsample'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spVsam = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spVsam[:] = outArr
    searchStr = 'LatticeConstant'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(x) for x in line.split()[1:7]]).astype(np.double)
        spLatC = sp_pro_analysis.create_dataset('LatticeParameter',dtype=np.double,shape=(6,),chunks=(6,),compressor=compressor)
        spLatC[:] = outArr
    searchStr = 'LatticeParameter'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(x) for x in line.split()[1:7]]).astype(np.double)
        spLatC = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(6,),chunks=(6,),compressor=compressor)
        spLatC[:] = outArr
    searchStr = 'RhoD'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spRHOD = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spRHOD[:] = outArr
    searchStr = 'MaxRingRad'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spMRR = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spMRR[:] = outArr
    searchStr = 'Lsd'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spLSD = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spLSD[:] = outArr
    searchStr = 'Wavelength'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spWL = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spWL[:] = outArr
    searchStr = 'Width'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spWidth = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spWidth[:] = outArr
    searchStr = 'px'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spPx = sp_pro_analysis.create_dataset('PixelSize',dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spPx[:] = outArr
    searchStr = 'UpperBoundThreshold'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spUBT = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spUBT[:] = outArr
    searchStr = 'BC'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spYCen = sp_pro_analysis.create_dataset('YCen',dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spYCen[:] = outArr
        outArr = np.array([float(line.split()[2])]).astype(np.double)
        spZCen = sp_pro_analysis.create_dataset('ZCen',dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spZCen[:] = outArr
    searchStr = 'p3'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spP3 = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spP3[:] = outArr
    searchStr = 'p2'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spP2 = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spP2[:] = outArr
    searchStr = 'p1'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spP1 = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spP1[:] = outArr
    searchStr = 'p0'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spP0 = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spP0[:] = outArr
    searchStr = 'tz'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spTz = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spTz[:] = outArr
    searchStr = 'ty'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spTy = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spTy[:] = outArr
    searchStr = 'tx'
    if line.startswith(f'{searchStr} '):
        outArr = np.array([float(line.split()[1])]).astype(np.double)
        spTx = sp_pro_analysis.create_dataset(searchStr,dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
        spTx[:] = outArr

spRTA  = sp_pro_analysis.create_dataset('RingThresh',dtype=np.double,shape=(RingThreshArr.shape),chunks=(RingThreshArr.shape),compressor=compressor)
spRTA[:] = RingThreshArr
spRTE  = sp_pro_analysis.create_dataset('RingsToExclude',dtype=np.double,shape=(RingExcludeArr.shape),chunks=(RingExcludeArr.shape),compressor=compressor)
spRTE[:] = RingExcludeArr
spOR = sp_pro_analysis.create_dataset('OmegaRanges',dtype=np.double,shape=(OmegaRanges.shape),chunks=(OmegaRanges.shape),compressor=compressor)
spOR[:] = OmegaRanges
spBS = sp_pro_analysis.create_dataset('BoxSizes',dtype=np.double,shape=(BoxSizes.shape),chunks=(BoxSizes.shape),compressor=compressor)
spBS[:] = BoxSizes
spImT = sp_pro_analysis.create_dataset('ImTransOpt',dtype=np.int32,shape=(ImTransOpts.shape),chunks=(ImTransOpts.shape),compressor=compressor)
spImT[:] = ImTransOpts
OmeFF -= skipF*omeStp
spSt = sp_pro_meas.create_dataset('start',dtype=np.double,shape=(1,),chunks=(1,),compressor=compressor)
spSt[:] = np.array([OmeFF])

if h5py.is_hdf5(InputFN):
    with h5py.File(InputFN,'r') as hf2:
        hf2.visititems(addData)

zipStore.close()

print(zarr.open(outfZip,'r').tree())
print(f"OutputZipName: {outfZip}")