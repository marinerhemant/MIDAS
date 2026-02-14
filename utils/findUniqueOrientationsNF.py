import os,sys
try:
    import midas_config
    utilsDir = midas_config.MIDAS_UTILS_DIR
except ImportError:
    utilsDir = os.path.expanduser('~/opt/MIDAS/utils/')
sys.path.insert(0,utilsDir)
from calcMiso import *

fn = sys.argv[1]
sg = int(sys.argv[2])
grFN = sys.argv[3]
minAng = np.double(sys.argv[4])
mic = np.genfromtxt(fn,skip_header=4)
eulers = mic[7:10]
oms = eul2omMat(eulers)
grF = open(grFN,'a')

doneArr = np.zeros(oms.shape[0])
for i in range(oms.shape[0]):
    if doneArr[i] == 1:
        continue
    om1 = oms[i]
    grF.write(f'{np.random.randint(100000)} {om1[0]} {om1[1]} {om1[2]} {om1[3]} {om1[4]} {om1[5]} {om1[6]} {om1[7]} {om1[8]}\n')
    for j in range(i+1,oms.shape[0]):
        if doneArr[j] == 1: continue
        om2 = oms[j]
        ang,axis = GetMisOrientationAngleOM(om1,om2,sg)
        ang*= rad2deg
        if ang < minAng:
            doneArr[j] = 1