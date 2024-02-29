import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def geReader(geFN,header=8192,numPxY=2048,numPxZ=2048,bytesPerPx=2):
    sz = os.path.getsize(geFN)
    nFrames = (sz-header) // (bytesPerPx*numPxY*numPxZ)
    return np.fromfile(geFN,dtype=np.uint16,offset=header,count=(sz-header)).reshape((nFrames,numPxY,numPxZ))

psFN = sys.argv[1]
lines = open(psFN).readlines()
for line in lines:
    if line.startswith('SeedFolder '):
        resultDir = line.split()[1]
    if line.startswith('RawFolder '):
        rawFolder = line.split()[1]
    if line.startswith('Dark '):
        darkFN = line.split()[1]
    if line.startswith('FileStem '):
        fStem = line.split()[1]
    if line.startswith('StartFileNrFirstLayer '):
        fNr = line.split()[1]
    if line.startswith('Padding '):
        pad = int(line.split()[1])
    if line.startswith('Ext '):
        ext = line.split()[1]
    
geFN = rawFolder + '/' + fStem + '_' + fNr.zfill(pad) + ext
outfn = rawFolder + '/' + fStem + '_' + fNr.zfill(pad)
os.makedirs(resultDir,exist_ok=True)
geData = geReader(geFN)
darkData = geReader(darkFN)
print(f'Input: {geFN}')
print(f'Dark: {darkFN}')
print(f'ResultDir: {resultDir}')
print(f'Out: {outfn}')
hf = h5py.File(outfn+'.h5','w')
exc = hf.create_group('exchange')
data = exc.create_dataset('data',shape=geData.shape,dtype=np.uint16,data=geData)
dark = exc.create_dataset('dark',shape=darkData.shape,dtype=np.uint16,data=darkData)
bright = exc.create_dataset('bright',shape=darkData.shape,dtype=np.uint16,data=darkData) # For now
meas = hf.create_group('measurement')
pro_meas = meas.create_group('process')
sp_pro_meas = pro_meas.create_group('scan_parameters')
analysis = hf.create_group('analysis')
pro_analysis = analysis.create_group('process')
sp_pro_analysis = pro_analysis.create_group('analysis_parameters')
sp_pro_analysis.create_dataset('ResultFolder',(1,),data=np.string_(resultDir))
RingThreshArr = np.zeros((1,2))
OmegaRanges = np.zeros((1,2))
OmegaRanges[0,0] = -10000
BoxSizes = np.zeros((1,4))
BoxSizes[0,0] = -10000
ImTransOpts = np.zeros((1))
ImTransOpts[0] = -1
skipF = 0
for line in lines:
    str = 'ImTransOpt'
    if line.startswith(f'{str} '):
        outArr = np.array(int(line.split()[1])).astype(np.int32)
        if (ImTransOpts[0] == -1):
            ImTransOpts[0] = outArr
        else:
            ImTransOpts = np.vstack((ImTransOpts,outArr))
    str = 'BoxSize'
    if line.startswith(f'{str} '):
        outArr = np.array([float(x) for x in line.split()[1:5]]).astype(np.double)
        outArr = outArr.reshape((1,4))
        if BoxSizes[0,0] == -10000:
            BoxSizes = outArr
        else:
            BoxSizes = np.vstack((BoxSizes,outArr))
    str = 'OmegaRange'
    if line.startswith(f'{str} '):
        outArr = np.array([float(x) for x in line.split()[1:3]]).astype(np.double)
        outArr = outArr.reshape((1,2))
        if OmegaRanges[0,0] == -10000:
            OmegaRanges = outArr
        else:
            OmegaRanges = np.vstack((OmegaRanges,outArr))
    str = 'RingThresh'
    if line.startswith(f'{str} '):
        outArr = np.array([float(x) for x in line.split()[1:3]]).astype(np.double)
        outArr = outArr.reshape((1,2))
        if RingThreshArr[0,0] == 0:
            RingThreshArr = outArr
        else:
            RingThreshArr = np.vstack((RingThreshArr,outArr))
    str = 'HeadSize'
    if line.startswith(f'{str} '):
        head = int(line.split()[1])
        if skipF==0:
            skipF = (head-8192) // (2*2048*2048)
            sp_pro_analysis.create_dataset('SkipFrame',dtype=np.int32,shape=(1),data=np.array(skipF))
    str = 'Twins'
    if line.startswith(f'{str} '):
        outArr = np.array(int(line.split()[1])).astype(np.int32)
        sp_pro_analysis.create_dataset(str,dtype=np.int32,shape=(1),data=outArr)
    str = 'MaxNFrames'
    if line.startswith(f'{str} '):
        outArr = np.array(int(line.split()[1])).astype(np.int32)
        sp_pro_analysis.create_dataset(str,dtype=np.int32,shape=(1),data=outArr)
    str = 'DoFit'
    if line.startswith(f'{str} '):
        outArr = np.array(int(line.split()[1])).astype(np.int32)
        sp_pro_analysis.create_dataset(str,dtype=np.int32,shape=(1),data=outArr)
    str = 'DiscModel'
    if line.startswith(f'{str} '):
        outArr = np.array(int(line.split()[1])).astype(np.int32)
        sp_pro_analysis.create_dataset(str,dtype=np.int32,shape=(1),data=outArr)
    str = 'UseMaximaPositions'
    if line.startswith(f'{str} '):
        outArr = np.array(int(line.split()[1])).astype(np.int32)
        sp_pro_analysis.create_dataset(str,dtype=np.int32,shape=(1),data=outArr)
    str = 'MaxNrPx'
    if line.startswith(f'{str} '):
        outArr = np.array(int(line.split()[1])).astype(np.int32)
        sp_pro_analysis.create_dataset(str,dtype=np.int32,shape=(1),data=outArr)
    str = 'MinNrPx'
    if line.startswith(f'{str} '):
        outArr = np.array(int(line.split()[1])).astype(np.int32)
        sp_pro_analysis.create_dataset(str,dtype=np.int32,shape=(1),data=outArr)
    str = 'MaxNPeaks'
    if line.startswith(f'{str} '):
        outArr = np.array(int(line.split()[1])).astype(np.int32)
        sp_pro_analysis.create_dataset(str,dtype=np.int32,shape=(1),data=outArr)
    str = 'PhaseNr'
    if line.startswith(f'{str} '):
        outArr = np.array(int(line.split()[1])).astype(np.int32)
        sp_pro_analysis.create_dataset(str,dtype=np.int32,shape=(1),data=outArr)
    str = 'NumPhases'
    if line.startswith(f'{str} '):
        outArr = np.array(int(line.split()[1])).astype(np.int32)
        sp_pro_analysis.create_dataset(str,dtype=np.int32,shape=(1),data=outArr)
    str = 'MinNrSpots'
    if line.startswith(f'{str} '):
        outArr = np.array(int(line.split()[1])).astype(np.int32)
        sp_pro_analysis.create_dataset(str,dtype=np.int32,shape=(1),data=outArr)
    str = 'UseFriedelPairs'
    if line.startswith(f'{str} '):
        outArr = np.array(int(line.split()[1])).astype(np.int32)
        sp_pro_analysis.create_dataset(str,dtype=np.int32,shape=(1),data=outArr)
    str = 'OverAllRingToIndex'
    if line.startswith(f'{str} '):
        outArr = np.array(int(line.split()[1])).astype(np.int32)
        sp_pro_analysis.create_dataset('OverallRingToIndex',dtype=np.int32,shape=(1),data=outArr)
    str = 'SpaceGroup'
    if line.startswith(f'{str} '):
        outArr = np.array(int(line.split()[1])).astype(np.int32)
        sp_pro_analysis.create_dataset(str,dtype=np.int32,shape=(1),data=outArr)
    str = 'LayerNr'
    if line.startswith(f'{str} '):
        outArr = np.array(int(line.split()[1])).astype(np.int32)
        sp_pro_analysis.create_dataset(str,dtype=np.int32,shape=(1),data=outArr)
    str = 'DoFullImage'
    if line.startswith(f'{str} '):
        outArr = np.array(int(line.split()[1])).astype(np.int32)
        sp_pro_analysis.create_dataset(str,dtype=np.int32,shape=(1),data=outArr)
    str = 'SkipFrame'
    if line.startswith(f'{str} '):
        outArr = np.array(int(line.split()[1])).astype(np.int32)
        skipF = int(line.split()[1])
        sp_pro_analysis.create_dataset(str,dtype=np.int32,shape=(1),data=outArr)
    str = 'OmegaFirstFile'
    if line.startswith(f'{str} '):
        OmeFF = float(line.split()[1])
    str = 'OmegaStep'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        omeStp = float(line.split()[1])
        sp_pro_meas.create_dataset('step',dtype=np.double,shape=(1),data=outArr)
    str = 'BadPxIntensity'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'tolTilts'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'tolBC'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'tolLsd'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'DiscArea'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'OverlapLength'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'ReferenceRingCurrent'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'zDiffThresh'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'GlobalPosition'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'StepSizePos'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'tInt'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'tGap'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'StepSizeOrient'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'MarginRadius'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'MarginRadial'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'MarginEta'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'MarginOme'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'MargABG'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'MargABC'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'OmeBinSize'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'EtaBinSize'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'MinEta'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'MaxOmeSpotIDsToIndex'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'MinOmeSpotIDsToIndex'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'BeamThickness'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'Wedge'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'Rsample'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'Hbeam'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'Vsample'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'LatticeConstant'
    if line.startswith(f'{str} '):
        outArr = np.array([float(x) for x in line.split()[1:7]]).astype(np.double)
        sp_pro_analysis.create_dataset('LatticeParameter',dtype=np.double,data=outArr)
    str = 'LatticeParameter'
    if line.startswith(f'{str} '):
        outArr = np.array([float(x) for x in line.split()[1:7]]).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,data=outArr)
    str = 'RhoD'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'MaxRingRad'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'Lsd'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'Wavelength'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'Width'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'px'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset('PixelSize',dtype=np.double,shape=(1),data=outArr)
    str = 'UpperBoundThreshold'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'BC'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset('YCen',dtype=np.double,shape=(1),data=outArr)
        outArr = np.array(float(line.split()[2])).astype(np.double)
        sp_pro_analysis.create_dataset('ZCen',dtype=np.double,shape=(1),data=outArr)
    str = 'p3'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'p2'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'p1'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'p0'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'tz'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'ty'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)
    str = 'tx'
    if line.startswith(f'{str} '):
        outArr = np.array(float(line.split()[1])).astype(np.double)
        sp_pro_analysis.create_dataset(str,dtype=np.double,shape=(1),data=outArr)

sp_pro_analysis.create_dataset('RingThresh',dtype=np.double,data=RingThreshArr)
sp_pro_analysis.create_dataset('OmegaRanges',dtype=np.double,data=OmegaRanges)
sp_pro_analysis.create_dataset('BoxSizes',dtype=np.double,data=BoxSizes)
sp_pro_analysis.create_dataset('ImTransOpt',dtype=np.int32,data=ImTransOpts)
OmeFF -= skipF*omeStp
sp_pro_meas.create_dataset('start',dtype=np.double,shape=(1),data=np.array(OmeFF))
hf.close()