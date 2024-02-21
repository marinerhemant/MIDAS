import os,sys
import numpy as np
import subprocess
import matplotlib.pyplot as plt
utilsDir = os.path.expanduser('~/opt/MIDAS/utils/')
sys.path.insert(0,utilsDir)
from multiprocessing import Pool

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
    Folder = '/Users/hsharma/Desktop/analysis/erdeniz_jul22/'
    os.makedirs(f'{Folder}/fitGrain/',exist_ok=True)
    psFN = 'ps.txt' # Relative inside the folder
    GrainsFN = Folder + '/Grains.csv'
    os.chdir(Folder)
    subprocess.call(os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/GetHKLList')+f' {psFN}',shell=True,stdout=open('hklsout.csv','w'))

    grs = np.genfromtxt(GrainsFN,skip_header=9)
    grsSort = grs[grs[:,20].argsort()]
    nGrs = grsSort.shape[0]
    nSpotsToRefine = nGrs // 200 # We will take 1/10th of grains and refine them
    IDs = list(tuple(grsSort[:nSpotsToRefine,0]))
    WD = []
    txVals = []
    for ID in IDs:
        WD.append([int(ID),Folder,psFN])
    with Pool() as pool:
        for result in pool.map(runFitGrainOne,WD):
            txVals.append(result)
    txVals = np.array(txVals)
    plt.plot(txVals)
    print('Mean tx:',np.mean(txVals))
    plt.show()

if __name__=="__main__":
    main()
