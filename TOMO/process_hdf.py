import numpy as np
import h5py
import argparse
import sys, os
import subprocess
import cv2

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

parser = MyParser(description='''Code to do tomo recon using HDF input.''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-dataFN', type=str, required=True, help='Data file name.')
parser.add_argument('-nCPUs', type=int, required=True, help='Number of CPU cores to use for recon.')
args, unparsed = parser.parse_known_args()
dataFN = args.dataFN
nCPUs = args.nCPUs

hf = h5py.File(dataFN,'r')

rot = 0

dxL = int(hf['analysis/process/analysis_parameters/CropXL'][0])
dxR = int(hf['analysis/process/analysis_parameters/CropXR'][0])
dzL = int(hf['analysis/process/analysis_parameters/CropZL'][0])
dzR = int(hf['analysis/process/analysis_parameters/CropZR'][0])
shift = hf['analysis/process/analysis_parameters/shift'][0]
#### shift or position of rotation axis.
if 'analysis/process/analysis_parameters/RotationAngle' in hf:
    rot = hf['analysis/process/analysis_parameters/RotationAngle'][0]

dark = hf['exchange/dark'][dzL:-dzR,dxL:-dxR].astype(np.float32)
bright = hf['exchange/bright'][:,dzL:-dzR,dxL:-dxR].astype(np.float32)
data = hf['exchange/data']
nFrames = data.shape[0]
nX = dark.shape[0]
nY = dark.shape[1]
M = cv2.getRotationMatrix2D((nX/2, nY/2), rot, scale=1)
if not os.path.exists(f'{dataFN}.raw'):
    print('Raw file was not found. Will generate raw file.')
    data = data[:,dzL:-dzR,dxL:-dxR].astype(np.uint16)
    if rot != 0:
        data = np.copy(data)
        dark = np.copy(dark)
        bright = np.copy(bright)
        for frameNr in nFrames:
            data[frameNr] = cv2.warpAffine(data[frameNr],M,(dark.shape[0],dark.shape[1]))
        dark = cv2.warpAffine(dark,M,(nX,nY))
        bright[0] = cv2.warpAffine(bright[0],M,(nX,nY))
        bright[1] = cv2.warpAffine(bright[1],M,(nX,nY))
    outf = open(f'{dataFN}.raw','w')
    dark.tofile(outf)
    bright.tofile(outf)
    data.tofile(outf)
    outf.close()
else:
    print("The raw file exists. Skipping generation of raw file.")

st_ome = hf['measurement/process/scan_parameters/start'][0]
step_ome = hf['measurement/process/scan_parameters/step'][0]
stop_ome = st_ome + step_ome*(nFrames-1)
angles = np.linspace(st_ome,num=nFrames,stop=stop_ome)

np.savetxt('mt_angles.txt',angles.T,fmt='%.6f')

# os.makedirs('rec',exist_ok=True)

f_out = open('mt_par.txt','w')
f_out.write('areSinos 0\n')
f_out.write('saveReconSeparate 0\n')
f_out.write(f'dataFileName {dataFN}.raw\n')
f_out.write(f'reconFileName recon_{dataFN}\n')
f_out.write(f'detXdim {dark.shape[1]}\n')
f_out.write(f'detYdim {dark.shape[0]}\n')
f_out.write('thetaFileName mt_angles.txt\n')
f_out.write('filter 2\n')
f_out.write(f'shiftValues {shift} {shift} 1.000\n')
f_out.write('ringRemovalCoefficient 1.0\n')
f_out.write('slicesToProcess -1\n')
f_out.close()

subprocess.call(os.path.expanduser('~/opt/MIDAS/TOMO/bin/MIDAS_TOMO')+f' mt_par.txt {nCPUs}',shell=True,cwd=os.getcwd())