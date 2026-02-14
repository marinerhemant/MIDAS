import sys,os
try:
    import midas_config
    utilsDir = midas_config.MIDAS_UTILS_DIR
except ImportError:
    utilsDir = os.path.expanduser('~/opt/MIDAS/utils/')
sys.path.insert(0,utilsDir)
import numpy as np
from numba import jit
import calcMiso

rad2deg = 57.2957795130823
sgNum = 225
minAngle = 1
micFile = 'MicrostructureTxtLayer0_H2.mic'
grainsFN = 'Grains.csv'
minConfidence = 0.5
maxNrGrains = 5000 # You can increase this if number of expected grains is higher.

def calcGrains(mic,grains):
	nrows,ncols = mic.shape
	nGrains = 0
	for rownr in range(nrows):
		eulThis = mic[rownr,7:10]
		if nGrains == 0:
			grains[nGrains,0:3] = eulThis
			nGrains += 1
		else:
			newGrain = 1
			for grNr in range(nGrains):
				if newGrain == 0:
					break
				eulN = grains[grNr,:]
				ang = rad2deg*calcMiso.GetMisOrientationAngle(eulThis,eulN,sgNum)
				if ang < minAngle:
					newGrain = 0
			if newGrain == 1:
				grains[nGrains,0:3] = eulThis
				nGrains += 1
		if nGrains > maxNrGrains:
			print("MaxNrGrains Reached")
			return -1
	print(nGrains)
	return nGrains

mic = np.genfromtxt(micFile,skip_header=4)
mic = mic[mic[:,10] > minConfidence,:]
grains = np.zeros((maxNrGrains,3))
nGrains = calcGrains(mic,grains)
if nGrains == -1:
	print("Increase the max limit for NrOfGrains and try again.")
grains = grains[:nGrains,:]
grOrients = np.zeros((nGrains,10))
for nr,gr in enumerate(grains):
	om = calcMiso.Euler2OrientMat(gr)
	grOrients[nr,0] = nr+1
	grOrients[nr,1:10] = om
grainsFile = open(grainsFN,"w")
grainsFile.write("%NumGrains "+str(nGrains)+"\n")
grainsFile.write("%BeamCenter 0\n")
grainsFile.write("%BeamThickness 400.000000\n")
grainsFile.write("%GlobalPosition 0.000000\n")
grainsFile.write("%NumPhases 1\n")
grainsFile.write("%PhaseInfo\n")
grainsFile.write("%\tSpaceGroup:"+str(sgNum)+"\n")
grainsFile.write("%\tLattice Parameter: 0.0 0.0 0.0 0.0 0.0 0.0\n")
grainsFile.write("%GrainID\tO11\tO12\tO13\tO21\tO22\tO23\tO31\tO32\tO33\tX\tY\tZ\ta\tb\tc\talpha\tbeta\tgamma\tDiffPos\tDiffOme\tDiffAngle\tGrainRadius\tConfidence\teFab11\teFab12\teFab13\teFab21\teFab22\teFab23\teFab31\teFab32\teFab33\teKen11\teKen12\teKen13\teKen21\teKen22\teKen23\teKen31\teKen32\teKen33\tRMSErrorStrain\tPhaseNr\n")
for gr in grOrients:
	grainsFile.write(str(int(gr[0]))+' '+str(gr[1])+' '+str(gr[2])+' '+str(gr[3])+' '+str(gr[4])+' '+str(gr[5])+' '+str(gr[6])+' '+str(gr[7])+' '+str(gr[8])+' '+str(gr[9])+'\n')
