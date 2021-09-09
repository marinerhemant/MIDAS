import os
from os.path import expanduser, basename
import subprocess

fStem = '/data/tomo1/mpe_mar21_data/mpe_mar21/ge3/TiPt_DAC_s3_dt_PFocus_att000_bot'
startNr = 2917
endNr = 2981
pad = 6
nFrames = 182
numProcs = 33
ext = '.ge3'
darkFN = '/data/tomo1/mpe_mar21_data/mpe_mar21/ge3/dark_before_002916.ge3'
paramFN = 'ps.txt'

radRange = [538,578]
rads = [558]
rWidth = 10
rBinSize = 0.25

etaRange = [-200, 195]
etas = [-180,-90,0,90]
etaWidth = 10
etaBinSize = 0.3

nFiles = endNr - startNr + 1
paramContents = open(paramFN,'r').readlines()
updF = open(paramFN+'.upd','w')
OutFolder = '.'
for line in paramContents:
	if line.startswith('OutFolder'):
		OutFolder = line.split()[1]
	updF.write(line)
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

cmd1 = f'{expanduser("~/opt/MIDAS/DT/bin/DetectorMapper")} {paramFN}.upd'
print(cmd1)
# ~ subprocess.call(cmd1,shell=True)

cmd = f'{expanduser("~/opt/MIDAS/DT/bin/IntegratorPeakFitOMP")} {paramFN}.upd {fStem} {startNr} {endNr} {pad} {ext} {darkFN} {nFrames} {numProcs}'
# ~ subprocess.call(cmd,shell=True)

from skimage.transform import iradon
import numpy as np
import matplotlib.pyplot as plt

## Read the sinos and do all recons.
thetas = np.linspace(0,nFrames-1,nFrames)
baseFN = basename(fStem)
outfStem = f'{OutFolder}/{baseFN}'
outputs = ['RMEAN','MixFactor','SigmaG','SigmaL','MaxInt','BGFit',
	'BGSimple','MeanError','FitIntegratedIntensity','TotalIntensity','TotalIntensityBackgroundCorr','MaxIntensityObs']
for rad in rads:
	for eta in etas:
		for output in outputs:
			fn = f'{outfStem}_FileNrs_{startNr}_{endNr}_{output}_Rad_{rad}_pm_{rWidth}_Eta_{eta}_pm_{etaWidth}_size_{nFiles}x{nFrames}_float32.bin'
			outfn = f'{fn}_recon_{nFiles}x{nFiles}.bin'
			sino = np.transpose(np.fromfile(fn,dtype=np.float32,count=(nFrames*nFiles)).reshape((nFrames,nFiles)))
			recon = iradon(sinogram,theta=thetas)
			recon.astype(np.float32).tofile(outfn)
