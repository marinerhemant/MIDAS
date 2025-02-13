#!/usr/bin/env python

import parsl
import subprocess
import sys, os
import argparse
import signal
import shutil
import numpy as np
from multiprocessing import Pool
utilsDir = os.path.expanduser('~/opt/MIDAS/utils/')
sys.path.insert(0,utilsDir)
v7Dir = os.path.expanduser('~/opt/MIDAS/NF_HEDM/v7/')
binDir = os.path.expanduser('~/opt/MIDAS/NF_HEDM/bin/')
sys.path.insert(0,v7Dir)
from parsl.app.app import python_app
pytpath = sys.executable

def median(psFN,distanceNr,logDir,resultFolder):
    f = open(f'{logDir}/median{distanceNr}_out.csv','w')
    f_err = open(f'{logDir}/median{distanceNr}_err.csv','w')
    cmd = os.path.expanduser("~/opt/MIDAS/NF_HEDM/bin/MedianImageLibTiff")+f' {psFN} {distanceNr}'
    f_err.write(cmd)
    subprocess.call(cmd,shell=True,stdout=f,stderr=f_err,cwd=resultFolder)
    f.close()
    f_err.close()

def median_local(distanceNr):
    f = open(f'{logDir}/median{distanceNr}_out.csv','w')
    f_err = open(f'{logDir}/median{distanceNr}_err.csv','w')
    cmd = os.path.expanduser("~/opt/MIDAS/NF_HEDM/bin/MedianImageLibTiff")+f' {psFN} {distanceNr}'
    f_err.write(cmd)
    subprocess.call(cmd,shell=True,stdout=f,stderr=f_err,cwd=resultFolder)
    f.close()
    f_err.close()

def image(psFN,nodeNr,nNodes,numProcs,logDir,resultFolder):
    f = open(f'{logDir}/image{nodeNr}_out.csv','w')
    f_err = open(f'{logDir}/image{nodeNr}_err.csv','w')
    cmd = os.path.expanduser("~/opt/MIDAS/NF_HEDM/bin/ImageProcessingLibTiffOMP")+f' {psFN} {nodeNr} {nNodes} {numProcs}'
    f_err.write(cmd)
    subprocess.call(cmd,shell=True,stdout=f,stderr=f_err,cwd=resultFolder)
    f.close()
    f_err.close()

def fit(psFN,nodeNr,nNodes,numProcs,logDir,resultFolder):
    f = open(f'{logDir}/fit{nodeNr}_out.csv','w')
    f_err = open(f'{logDir}/fit{nodeNr}_err.csv','w')
    cmd = os.path.expanduser("~/opt/MIDAS/NF_HEDM/bin/FitOrientationOMP")+f' {psFN} {nodeNr} {nNodes} {numProcs}'
    f_err.write(cmd)
    subprocess.call(cmd,shell=True,stdout=f,stderr=f_err,cwd=resultFolder)
    f.close()
    f_err.close()

default_handler = None

def handler(num, frame):    
    subprocess.call("rm -rf /dev/shm/*.bin",shell=True)
    print("Ctrl-C was pressed, cleaning up.")
    return default_handler(num, frame) 

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

default_handler = signal.getsignal(signal.SIGINT)
signal.signal(signal.SIGINT, handler)
parser = MyParser(description='''Near-field HEDM analysis using MIDAS. V7.0.0, contact hsharma@anl.gov
                  The machine MUST have write access to the DataDirectory.
                  The data is constructed from the parameter file as follows: DataDirectory/OrigFileName_XXXX.tif
                  nNodes or nCPUs must exceed number of distances.
                  ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-paramFN', type=str, required=False, default='', help='Parameter file name. Provide either paramFN and/or dataFN.')
parser.add_argument('-nCPUs', type=int, required=False, default=10, help='Number of CPU cores to use if running locally.')
parser.add_argument('-machineName', type=str, required=False, default='local', help='Machine name for execution, local, orthrosnew, orthrosall, umich, marquette, purdue.')
parser.add_argument('-nNodes', type=int, required=False, default=1, help='Number of nodes for execution, omit if want to automatically select.')
parser.add_argument('-startLayerNr', type=int, required=False, default=1, help='NOT IMPLEMENTED YET!!! Start LayerNr to process.')
parser.add_argument('-endLayerNr', type=int, required=False, default=1, help=f'NOT IMPLEMENTED YET!!! End LayerNr to process. If Start and End'+
                    ' LayerNrs are equal, it will only process 1 layer, else will process multiple layers.')
parser.add_argument('-ffSeedOrientations', type=int, required=False, default=0, 
                    help=f'If want to use seedOrientations from far-field results, provide 1 else 0. If 1, paramFN must have a parameter GrainsFile '+
                        'pointing to the location of the Grains.csv file. NEXT PART NOT IMPLEMENTED YET!!!!!'+
                        ' If put to 1, you can add the following parameters: FinalGridSize, FinalEdgeLength,'+
                        ' FullSeedFile, MinConfidenceLowerBound and MinConfidence to rerun analysis with all possible orientations.')
parser.add_argument('-doImageProcessing', type=int, required=False, default=1, 
                    help='If want do ImageProcessing, put to 1, else 0. This is only for single layer processing.')
if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    print("MUST PROVIDE paramFN")
    sys.exit(1)
args, unparsed = parser.parse_known_args()
psFN = args.paramFN
numProcs = args.nCPUs
machineName = args.machineName
nNodes = args.nNodes
ffSeedOrientations = args.ffSeedOrientations
doImageProcessing = args.doImageProcessing

#### What are the steps here: initialSetup, then (optional) median, image processing, then one point, multiple points or full recon.
lines = open(psFN).readlines()
tomoFN = ''
GridMask = []
for line in lines:
    if line.startswith('DataDirectory '):
        resultFolder = line.split(' ')[1].rstrip()
    elif line.startswith('OrigFileName '):
        origFileName = line.split(' ')[1].rstrip()
    elif line.startswith('RawStartNr '):
        firstNr = int(line.split(' ')[1])
    elif line.startswith('nDistances '):
        nDistances = int(line.split(' ')[1])
    elif line.startswith('ReducedFileName '):
        reducedName = line.split(' ')[1].rstrip()
    elif line.startswith('GrainsFile '):
        grainsFile = line.split(' ')[1].rstrip()
    elif line.startswith('SeedOrientations '):
        seedOrientations = line.split(' ')[1].rstrip()
    elif line.startswith('TomoImage '):
        tomoFN = line.split(' ')[1].rstrip()
    elif line.startswith('TomoPixelSize '):
        tomoPx = float(line.split(' ')[1])
    elif line.startswith('GridMask '):
        GridMask = [float(line.split(' ')[i+1]) for i in range(4)]

os.environ['MIDAS_SCRIPT_DIR'] = resultFolder
if machineName == 'local':
    nNodes = 1
    from localConfig import *
    parsl.load(config=localConfig)
elif machineName == 'orthrosnew':
    numProcs = 32
    nNodes = 11
    from orthrosAllConfig import *
    parsl.load(config=orthrosNewConfig)
elif machineName == 'orthrosall':
    numProcs = 64
    nNodes = 5
    from orthrosAllConfig import *
    parsl.load(config=orthrosAllConfig)
elif machineName == 'umich':
    numProcs = 36
    os.environ['nNodes'] = str(nNodes)
    from uMichConfig import *
    parsl.load(config=uMichConfig)
elif machineName == 'marquette':
    numProcs = 36
    os.environ['nNodes'] = str(nNodes)
    from marquetteConfig import *
    parsl.load(config=marquetteConfig)
elif machineName == 'purdue':
    numProcs = 128
    os.environ['nNodes'] = str(nNodes)
    from purdueConfig import *
    parsl.load(config=purdueConfig)


os.chdir(resultFolder)
reducedFolder = os.path.dirname(f'{resultFolder}/{reducedName}')
logDir = f'{resultFolder}/midas_log/'
os.makedirs(reducedFolder,exist_ok=True)
os.makedirs(logDir,exist_ok=True)

#### HKLS ####
f = open(f'{logDir}/hkls_out.csv','w')
f_err = open(f'{logDir}/hkls_err.csv','w')
cmd = os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/GetHKLList")+f' {psFN}'
f_err.write(cmd)
subprocess.call(cmd,shell=True,stdout=f,stderr=f_err,cwd=resultFolder)
f.close()
f_err.close()

#### SEED ####
if ffSeedOrientations == 1:
    f = open(f'{logDir}/seed_out.csv','w')
    f_err = open(f'{logDir}/seed_err.csv','w')
    cmd = os.path.expanduser("~/opt/MIDAS/NF_HEDM/bin/GenSeedOrientationsFF2NFHEDM")+f' {grainsFile} {seedOrientations}'
    f_err.write(cmd)
    subprocess.call(cmd,shell=True,stdout=f,stderr=f_err,cwd=resultFolder)
    f.close()
    f_err.close()

nrOrientations = len(open(seedOrientations).readlines())
f_ps = open(psFN,'a')
f_ps.write(f'NrOrientations {nrOrientations}\n')
f_ps.close()

#### HEXGRID ####
f = open(f'{logDir}/hex_out.csv','w')
f_err = open(f'{logDir}/hex_err.csv','w')
cmd = os.path.expanduser("~/opt/MIDAS/NF_HEDM/bin/MakeHexGrid")+f' {psFN}'
f_err.write(cmd)
subprocess.call(cmd,shell=True,stdout=f,stderr=f_err,cwd=resultFolder)
f.close()
f_err.close()
if len(tomoFN) > 1:
    f = open(f'{logDir}/tomo_out.csv','w')
    f_err = open(f'{logDir}/tomo_err.csv','w')
    cmd = os.path.expanduser("~/opt/MIDAS/NF_HEDM/bin/filterGridfromTomo")+f' {tomoFN} {tomoPx}'
    f_err.write(cmd)
    subprocess.call(cmd,shell=True,stdout=f,stderr=f_err,cwd=resultFolder)
    f.close()
    f_err.close()
    shutil.move('grid.txt','grid_unfilt.txt')
    shutil.move('gridNew.txt','grid.txt')
elif len(GridMask) > 0:
    print(GridMask)
    gridpoints = np.genfromtxt('grid.txt',skip_header=1,delimiter=' ')
    print(gridpoints)
    gridpoints = gridpoints[gridpoints[:,2]>=GridMask[0],:]
    gridpoints = gridpoints[gridpoints[:,2]<=GridMask[1],:]
    gridpoints = gridpoints[gridpoints[:,3]<=GridMask[2],:]
    gridpoints = gridpoints[gridpoints[:,3]>=GridMask[3],:]
    nrPoints = gridpoints.shape[0]
    print(nrPoints)
    print(gridpoints)
    shutil.move('grid.txt','grid_old.txt')
    np.savetxt('grid.txt',gridpoints,fmt='%.6f',delimiter=' ',header=f'{nrPoints}',comments='')

#### MakeDiffrSpots
f = open(f'{logDir}/spots_out.csv','w')
f_err = open(f'{logDir}/spots_err.csv','w')
cmd = os.path.expanduser("~/opt/MIDAS/NF_HEDM/bin/MakeDiffrSpots")+f' {psFN}'
f_err.write(cmd)
subprocess.call(cmd,shell=True,stdout=f,stderr=f_err,cwd=resultFolder)
f.close()
f_err.close()

#### ImageProcessing
if doImageProcessing == 1:
    #### We can now do median, then peaks
    if machineName == 'local':
        p = Pool(nDistances)
        work_data = [i for i in range(1,nDistances+1)]
        p.map(median_local,work_data)
    else:
        resMedian = []
        for distanceNr in range(1,nDistances+1):
            resMedian.append(median(psFN,distanceNr,logDir,resultFolder))
    resImage = []
    for nodeNr in range(nNodes):
        resImage.append(image(psFN,nodeNr,nNodes,numProcs,logDir,resultFolder))

#### MMAP
f = open(f'{logDir}/map_out.csv','w')
f_err = open(f'{logDir}/map_err.csv','w')
cmd = os.path.expanduser("~/opt/MIDAS/NF_HEDM/bin/MMapImageInfo")+f' {psFN}'
f_err.write(cmd)
subprocess.call(cmd,shell=True,stdout=f,stderr=f_err,cwd=resultFolder)
f.close()
f_err.close()

#### FitOrientation
resFit = []
for nodeNr in range(nNodes):
    resFit.append(fit(psFN,nodeNr,nNodes,numProcs,logDir,resultFolder))


#### ParseMic
f = open(f'{logDir}/parse_out.csv','w')
f_err = open(f'{logDir}/parse_err.csv','w')
cmd = os.path.expanduser("~/opt/MIDAS/NF_HEDM/bin/ParseMic")+f' {psFN}'
f_err.write(cmd)
subprocess.call(cmd,shell=True,stdout=f,stderr=f_err,cwd=resultFolder)
f.close()
f_err.close()
