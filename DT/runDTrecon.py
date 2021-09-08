import os
from os.path import expanduser
import subprocess

fStem = '/data/tomo1/mpe_mar21_data/mpe_mar21/ge3/TiPt_DAC_s3_dt_PFocus_att000'
startNr = 2917
endNr = 2981
pad = 6
ext = '.ge3'
darkFN = '/data/tomo1/mpe_mar21_data/mpe_mar21/ge3/dark_before_002916.ge3'
paramFN = 'ps.txt'

subprocess.call(expanduser('~/opt/MIDAS/DT/bin/DetectorMapper')+' '+paramFN,shell=True)

for frameNr in range(startNr,endNr+1):
	thisFN = f'{fStem}_{str(frameNr).zfill(pad)}{ext}'
	cmd = f'{expanduser("~/opt/MIDAS/DT/bin/IntegratorPeakFit")} {paramFN} {thisFN} {darkFN}'
	subprocess.call(cmd,shell=True)
