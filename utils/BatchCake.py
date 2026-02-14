import numpy as np
from PIL import Image
import h5py
from scipy.optimize import curve_fit
import subprocess
import os, sys
import argparse
from tqdm import tqdm
try:
    utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    if utils_dir not in sys.path:
        sys.path.append(utils_dir)
    import midas_config
except ImportError:
    midas_config = None

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

parser = MyParser(description='''Code to integrate and then fit peak positions''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-paramFN', type=str, required=True, help='Parameter file name.')
args, unparsed = parser.parse_known_args()
params = args.paramFN

paramContents = open(params).readlines()
startNr = 1
endNr = 1
folder = '.'
darkFN = ''
fStem = ''
for line in paramContents:
    if line.startswith('StartNr'):
        startNr = int(line.split()[1])
    elif line.startswith('EndNr'):
        endNr = int(line.split()[1])
    elif line.startswith('Folder'):
        folder = line.split()[1]
    elif line.startswith('Dark'):
        darkFN = line.split()[1]
    elif line.startswith('FileStem'):
        fStem = line.split()[1]

if len(fStem) == 0 or len(darkFN) == 0:
    print('Must provide FileStem and Dark in the parameter file.')
    sys.exit()

if folder == '.':
    folder = os.getcwd()

conv_folder = f'{folder}/converted_files/'
os.makedirs(conv_folder,exist_ok=True)
os.makedirs(f'{conv_folder}/output',exist_ok=True)

f = open(params,'w')
OutFolderWritten = 0
for line in paramContents:
    if line.startswith('OutFolder'):
        f.write(f'OutFolder {conv_folder}/output\n')
        OutFolderWritten = 1
    else:
        f.write(line)
if OutFolderWritten == 0:
    f.write(f'OutFolder {conv_folder}/output\n')
f.close()

dark = np.array(Image.open(f'{folder}/{darkFN}'))
dark.astype(np.uint16).tofile(f'{conv_folder}/{darkFN}.mar')

print("Generating a map file to do quick caking afterwards.")
if midas_config and midas_config.MIDAS_BIN_DIR:
    detector_mapper = os.path.join(midas_config.MIDAS_BIN_DIR, 'DetectorMapper')
else:
    detector_mapper = os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/DetectorMapper')
subprocess.call(detector_mapper + f' {params}',shell=True)
print('Map file generated. Now doing caking.')

for fileNr in tqdm(range(startNr,endNr+1)):
    im = np.array(Image.open(f'{folder}/{fStem}{fileNr}.tif'))
    im.astype(np.uint16).tofile(f'{conv_folder}/{fStem}{fileNr}.mar')
    if midas_config and midas_config.MIDAS_BIN_DIR:
        integrator = os.path.join(midas_config.MIDAS_BIN_DIR, 'Integrator')
    else:
        integrator = os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/Integrator')
    subprocess.call(integrator + f' {params} {conv_folder}/{fStem}{fileNr}.mar {conv_folder}/{darkFN}.mar',shell=True,cwd=folder)
