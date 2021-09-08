import os
from os.path import expanduser
import subprocess

fStem = '/data/tomo1/mpe_mar21_data/mpe_mar21/ge3/TiPt_DAC_s3_dt_PFocus_att000_bot'
startNr = 2917
# ~ endNr = 2981
endNr = 2918
pad = 6
nFrames = 1
numProcs = 1
ext = '.ge3'
darkFN = '/data/tomo1/mpe_mar21_data/mpe_mar21/ge3/dark_before_002916.ge3'
paramFN = 'ps.txt'

# ~ subprocess.call(expanduser('~/opt/MIDAS/DT/bin/DetectorMapper')+' '+paramFN,shell=True)

cmd = f'{expanduser("~/opt/MIDAS/DT/bin/IntegratorPeakFitOMP")} {paramFN} {fStem} {startNr} {endNr} {pad} {ext} {darkFN} {nFrames} {numProcs}'
subprocess.call(cmd,shell=True)
