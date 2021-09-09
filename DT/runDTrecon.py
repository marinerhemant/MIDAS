import os
from os.path import expanduser, basename
import subprocess
import numpy as np

startOme = 0
omeStep = 1
fStem = 'data/TiPt_DAC_s3_dt_PFocus_att000_bot'
startNr = 2917
endNr = 2981
pad = 6
nFrames = 182
numProcs = 33
ext = '.ge3'
darkFN = 'data/dark_before_002916.ge3'
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
baseFN = basename(fStem)
outfStem = f'{OutFolder}/{baseFN}'

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
subprocess.call(cmd1,shell=True)

cmd = f'{expanduser("~/opt/MIDAS/DT/bin/IntegratorPeakFitOMP")} {paramFN}.upd {fStem} {startNr} {endNr} {pad} {ext} {darkFN} {nFrames} {numProcs}'
subprocess.call(cmd,shell=True)

## Read caked output
# Map:
fn = f'{outfStem}_{str(startNr).zfill(pad)}{ext}.REtaAreaMap.csv'
f = open(fn)
line1 = f.readline().split()
etaBins = int(line1[1])
rBins = int(line1[3])
f.readline()
data = np.genfromtxt(f) # Cols: Radius[px] 2Theta[degrees] Eta[degrees] BinArea[pixels]
f.close()
sizeFile = nFrames*etaBins*rBins
for fnr in range(startNr,endNr+1):
	fn = f'{outfStem}_{str(fnr).zfill(pad)}{ext}_integrated.bin'
	data = np.fromfile(fn,dtype=np.double,count=(sizeFile)).reshape(nFrames,etaBins*rBins)
	print(data.shape)

## Read Lineouts:
nEtas = len(etas)
for fnr in range(startNr,endNr+1):
	fn = f'{outfStem}_{str(fnr).zfill(pad)}{ext}.LineOuts.bin'
	f = open(fn)
	nEls = int(np.fromfile(f,dtype=np.ulonglong,count=(3))[0])
	data = np.fromfile(f,dtype=np.double,count=(nFrames*nEls*nEtas)).reshape((nFrames,nEls*nEtas))
	f.close()
	print(data.shape)

## Read the sinos and do all recons.
from skimage.transform import iradon

thetas = np.linspace(startOme,startOme+(nFrames-1)*omeStep,nFrames)
outputs = ['RMEAN','MixFactor','SigmaG','SigmaL','MaxInt','BGFit',
	'BGSimple','MeanError','FitIntegratedIntensity','TotalIntensity','TotalIntensityBackgroundCorr','MaxIntensityObs']
for rad in rads:
	for eta in etas:
		for output in outputs:
			fn = f'{outfStem}_FileNrs_{startNr}_{endNr}_{output}_Rad_{rad}_pm_{rWidth}_Eta_{eta}_pm_{etaWidth}_size_{nFiles}x{nFrames}_float32.bin'
			outfn = f'{fn}_recon_{nFiles}x{nFiles}.bin'
			print(f'ReconFN: {outfn}')
			# Read the sino
			sino = np.transpose(np.fromfile(fn,dtype=np.float32,count=(nFrames*nFiles)).reshape((nFrames,nFiles)))
			recon = iradon(sino,theta=thetas)
			recon.astype(np.float32).tofile(outfn)
