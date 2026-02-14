from skimage.transform import iradon
import os
from os.path import expanduser, basename
import subprocess
import numpy as np
import sys
try:
    utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils')
    if utils_dir not in sys.path:
        sys.path.append(utils_dir)
    import midas_config
    DT_BIN_DIR = os.path.join(midas_config.MIDAS_ROOT, 'DT', 'bin')
    TOMO_BIN_DIR = midas_config.MIDAS_TOMO_BIN_DIR
except ImportError:
    midas_config = None
    DT_BIN_DIR = expanduser('~/opt/MIDAS/DT/bin')
    TOMO_BIN_DIR = expanduser('~/opt/MIDAS/TOMO/bin')

def findNextPowerOf2(np2):
 np2=np2-1
 while np2&np2-1:
  np2=np2&np2-1
 return np2<<1
  

startOme = 180.25
omeStep = -0.25
fStem = '/local/analysis/mpe_nov22_midas/dt/data/dm_dt_pf_U3O8_600A'
startNr = 43
endNr = 97
pad = 6
nFrames = 1441
numProcs = 32
ext = '.raw'
darkFN = '/local/analysis/mpe_nov22_midas/dt/dark_zeros_001234.raw'
paramFN = 'ps_dt.txt'
BadRotation = 1  # If each alternate scan is in the other direction
filt = '4'
'''                                   0: default\n"
        "                            * 1: Shepp / Logan\n"
        "                            * 2: Hann\n"
        "                            * 3: Hamming\n"
        "                            * 4: Ramp\n"'''
radRange = [100,140]
rads = [118]#,208,412,500,619]
rWidth = 10
rBinSize = 0.25

etaRange = [-180, 180]
etas = [0]
etaWidth = 180
etaBinSize = 3

nFiles = endNr - startNr + 1
paramContents = open(paramFN, 'r').readlines()
updF = open(paramFN+'.upd', 'w')
OutFolder = '.'
for line in paramContents:
    if line.startswith('OutFolder'):
        OutFolder = line.split()[1]
    updF.write(line)
baseFN = basename(fStem)
outfStem = f'{OutFolder}/{baseFN}'
os.makedirs(OutFolder,exist_ok=True)

updF.write(f'RMin {radRange[0]}\n')
updF.write(f'RMax {radRange[1]}\n')
updF.write(f'RBinSize {rBinSize}\n')
updF.write(f'EtaBinSize {etaBinSize}\n')
updF.write(f'EtaMin {etaRange[0]}\n')
updF.write(f'EtaMax {etaRange[1]}\n')
for rad in rads:
    updF.write(f'RadiusToFit {rad} {rWidth}\n')

for eta in etas:
    updF.write(f'EtaToFit {eta} {etaWidth}\n')
updF.close()

#############DO THE LINEOUTS#####

cmd1 = f'{os.path.join(DT_BIN_DIR, "DetectorMapper")} {paramFN}.upd'
subprocess.call(cmd1,shell=True)

cmd = f'{os.path.join(DT_BIN_DIR, "IntegratorPeakFitOMP")} {paramFN}.upd {fStem} {startNr} {endNr} {pad} {ext} {darkFN} {nFrames} {numProcs}'
subprocess.call(cmd,shell=True)


###########METHOD 1##############
#################################

## Read Lineouts:
nEtas = len(etas)
nRads = len(rads)
fn = f'{outfStem}_{str(startNr).zfill(pad)}{ext}.LineOuts.bin'
f = open(fn)
nEls = int(np.fromfile(f,dtype=np.ulonglong,count=(3))[0])
nElsPerRad = int(nEls/nRads)
reconSize=2*findNextPowerOf2(nFiles)
f.close()

sino_arr = np.zeros((nFiles,nFrames,nEtas,nElsPerRad*nRads))
for fnr in range(startNr,endNr+1):
 fn = f'{outfStem}_{str(fnr).zfill(pad)}{ext}.LineOuts.bin'
 f = open(fn)
 nEls = int(np.fromfile(f,dtype=np.ulonglong,count=(3))[0])
 if BadRotation == 1:
  if (fnr-startNr) % 2 == 1:
   
   data = np.fromfile(f,dtype=np.double,count=(nEls*nEtas*nFrames))
   
   data=np.reshape(data,(nFrames,nEtas,nEls))# Intensity values for the lineouts
   data=np.flip(data,0)
   data=np.reshape(data,(nFrames,nEtas,nElsPerRad*nRads))
   
  else:
   data = np.fromfile(f,dtype=np.double,count=(nEls*nEtas*nFrames)).reshape(nFrames,nEtas,nElsPerRad*nRads)
 else:
  data = np.fromfile(f,dtype=np.double,count=(nEls*nEtas*nFrames)).reshape(nFrames,nEtas,nElsPerRad*nRads)
 f.close()
 sino_arr[fnr-startNr] = data

sino_arr_flipped = np.transpose(sino_arr,(2,3,1,0)).astype('float32')

sino_arr_flipped.tofile('sinos.bin')

os.makedirs('tomo',exist_ok=True)
fn = 'tomo_config.txt'
f = open(fn,'w')
f.write('dataFileName sinos.bin\n')
f.write('reconFileName tomo/recon\n')
f.write('areSinos 1\n')
f.write('detXdim '+str(nFiles)+'\n')
f.write('detYdim '+str(nRads*nElsPerRad*nEtas)+'\n')
f.write('filter '+filt+'\n')
f.write('thetaRange '+str(startOme)+' '+str(startOme+(nFrames-1)*omeStep)+' '+str(omeStep)+'\n')
f.write('slicesToProcess -1\n')
f.write('shiftValues 0.000000 0.000000 0.500000\n')
f.write('ringRemovalCoefficient 1.0\n')
f.write('doLog 0\n')
f.write('ExtraPad 1\n')
f.close()

cmdTomo = f'{os.path.join(TOMO_BIN_DIR, "MIDAS_TOMO")} tomo_config.txt {numProcs}'
subprocess.call(cmdTomo,shell=True)

recons = np.empty((nRads*nElsPerRad*nEtas,reconSize,reconSize))
for fNr in range(nRads*nElsPerRad*nEtas):
    fn = 'tomo/recon_'+str(fNr).zfill(5)+'_001_p0000.0_'+str(reconSize)+'_'+str(reconSize)+'_float32.bin'
    f = open(fn)
    data = np.fromfile(f,dtype=np.float32,count=(reconSize*reconSize)).reshape((reconSize,reconSize))
    recons[fNr] = data

recons_reshape = np.transpose(recons.reshape((nRads*nElsPerRad,nEtas,reconSize,reconSize)))
print(recons_reshape.shape)

recons_reshape.astype(np.double).tofile('RawDataPeakFit.bin')
updF = open(paramFN+'.upd', 'a')
updF.write('RawDataPeakFN RawDataPeakFit.bin\n');
updF.write('PeakFitResultFN PeakFitResult.bin\n');
updF.write('nElsPerRad '+str(nElsPerRad)+'\n');
updF.write('ReconSize '+str(reconSize)+'\n');
updF.close()

subprocess.call(f'{os.path.join(DT_BIN_DIR, "PeakFit")} {paramFN}.upd {numProcs}',shell=True)

fitResult = np.fromfile('PeakFitResult.bin',dtype=np.double,count=(reconSize*reconSize*nEtas*nRads*12)).reshape((reconSize,reconSize,nEtas,nRads,12)).transpose()
fitResult[4][1].transpose().tofile('IntensityFit.bin')


##########METHOD 2################
##################################

# Read the sinos and do all recons.
angles = np.linspace(startOme, startOme+(nFrames-1)*omeStep, nFrames)
outputs = ['RMEAN', 'MixFactor', 'SigmaG', 'SigmaL', 'MaxInt', 'BGFit',
           'BGSimple', 'MeanError', 'FitIntegratedIntensity', 'TotalIntensity', 'TotalIntensityBackgroundCorr', 'MaxIntensityObs']

for output in outputs:
    totalsino = 0
    for rad in rads:
        for eta in etas:
            fn = f'{outfStem}_FileNrs_{startNr}_{endNr}_{output}_Rad_{rad}_pm_{rWidth}_Eta_{eta}_pm_{etaWidth}_size_{nFiles}x{nFrames}_float32.bin'
            outfn = f'{fn}_recon_{nFiles}x{nFiles}.bin'
            sino = np.transpose(np.fromfile(fn, dtype=np.float32, count=(nFrames*nFiles)).reshape((nFrames, nFiles)))  # Sinogram values
            if BadRotation == 1:
                for frameNr in range(nFiles):
                    if frameNr % 2 == 1:
                        sino[frameNr] = np.flip(sino[frameNr])
            totalsino = totalsino+sino
            sino.tofile(f'{outfStem}_corrected_sino_flipped_FileNrs_{startNr}_{endNr}_{output}_Rad_{rad}_pm_{rWidth}_Eta_{eta}_pm_{etaWidth}_size_{nFrames}x{nFiles}_float32.bin')
            recon = iradon(sino, theta=angles)
            recon.astype(np.float32).tofile(outfn)
    outputfntotal = f'{outfStem}_FileNrs_{startNr}_{endNr}_{output}_size_{nFiles}x{nFiles}_recon_{nFiles}x{nFiles}.bin'
    if BadRotation == 1:
        totalsino.tofile(
            f'{outfStem}_corrected_totalsino_FileNrs_{startNr}_{endNr}_{output}_size_{nFrames}x{nFiles}_float32.bin')
    else:
        totalsino.tofile(
            f'{outfStem}_totalsino_FileNrs_{startNr}_{endNr}_{output}_size_{nFrames}x{nFiles}_float32.bin')
    recon = iradon(totalsino, theta=angles)
    recon.astype(np.float32).tofile(outputfntotal)
