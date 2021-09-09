import os
from os.path import expanduser
import subprocess

fStem = '/data/tomo1/mpe_mar21_data/mpe_mar21/ge3/TiPt_DAC_s3_dt_PFocus_att000_bot'
startNr = 2917
endNr = 2981
pad = 6
nFrames = 182
numProcs = 32
ext = '.ge3'
darkFN = '/data/tomo1/mpe_mar21_data/mpe_mar21/ge3/dark_before_002916.ge3'
paramFN = 'ps.txt'

radRange = [200, 1200]
rads = [558, 664, 687]
rWidth = 10
rBinSize = 0.25

etaRange = [-220, 220]
etas = [-180,-90,0,90,180]
etaWidth = 10
etaBinSize = 0.3

paramContents = open(paramFN,'r').readlines()
updF = open(paramFN+'.upd','w')
for line in paramContents:
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

cmd1 = f'{expanduser("~/opt/MIDAS/DT/bin/DetectorMapper")} {paramFN}.upd'
subprocess.call(cmd1,shell=True)

cmd = f'{expanduser("~/opt/MIDAS/DT/bin/IntegratorPeakFitOMP")} {paramFN}.upd {fStem} {startNr} {endNr} {pad} {ext} {darkFN} {nFrames} {numProcs}'
subprocess.call(cmd,shell=True)
