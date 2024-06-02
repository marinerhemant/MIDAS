#!/usr/bin/env python

import os,sys
import numpy as np
import subprocess
import matplotlib.pyplot as plt
utilsDir = os.path.expanduser('~/opt/MIDAS/utils/')
sys.path.insert(0,utilsDir)
from multiprocessing import Pool
import argparse

class MyParser(argparse.ArgumentParser):
	def error(self, message):
		sys.stderr.write('error: %s\n' % message)
		self.print_help()
		sys.exit(2)

IDs = []

def runFitGrainOne(WD):
    ID = WD[0]
    Folder = WD[1]
    psFN = WD[2]
    subprocess.call(os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/FitGrain ')+f' {Folder} {psFN} {ID}',shell=True,stdout=open(f'{Folder}/fitGrain/fitGrainOut{ID}.txt','w'))
    lines = open(f'{Folder}/fitGrain/fitGrainOut{ID}.txt','r').readlines()
    for line in lines:
        if line.startswith('Input tx:'):
            return float(line.rstrip().split()[-1])
    return np.nan

def main():
    global IDs
    parser = MyParser(description='''Fit Grains to find tx, contact hsharma@anl.gov''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-resultFolder', type=str, required=True, help='Folder where the results live')
    parser.add_argument('-psFN', type=str, required=True, help='ps file that will be used. This is temporary. Will be replaced by zarr soon.')
    parser.add_argument('-fractionGrains', type=int, required=False, default=20, help='1/fractionGrains will be used optimized. Typically ~20.')
    parser.add_argument('-numProcs', type=int, required=False, default=8, help='Number of processors to use.')
    parser.add_argument('-propertyToSortGrains', type=str, required=False, default='DiffOme', help='Which property to choose to sort grains, choose either DiffOme or DiffPos.')
    args, unparsed = parser.parse_known_args()
    Folder = args.resultFolder
    psFN = args.psFN
    frac = args.fractionGrains
    numProcs = args.numProcs
    prop = args.propertyToSortGrains
    if prop == 'DiffOme':
        colVal = 20
    elif prop == 'DiffPos':
        colVal = 19
    else: # Default DiffOme
        colVal = 20
    os.makedirs(f'{Folder}/fitGrain/',exist_ok=True)
    GrainsFN = Folder + '/Grains.csv'
    os.chdir(Folder)
    subprocess.call(os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/GetHKLList')+f' {psFN}',shell=True,stdout=open('hklsout.csv','w'))

    grs = np.genfromtxt(GrainsFN,skip_header=9)
    grsSort = grs[grs[:,colVal].argsort()]
    nGrs = grsSort.shape[0]
    nSpotsToRefine = nGrs // frac # We will take 1/nth of grains and refine them
    IDs = list(tuple(grsSort[:nSpotsToRefine,0]))
    WD = []
    txVals = []
    for ID in IDs:
        WD.append([int(ID),Folder,psFN])
    with Pool(processes=numProcs) as pool:
        for result in pool.map(runFitGrainOne,WD):
            txVals.append(result)
    txVals = np.array(txVals)
    plt.plot(txVals)
    print('Mean tx:',np.mean(txVals))
    plt.show()

if __name__=="__main__":
    main()
