import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi
import calcMiso
micdata = np.genfromtxt('MicrostructureText_Layer4.mic',skip_header=4)
grains = np.genfromtxt('GrainsLayer4.csv',skip_header=9)
nGrains,z = grains.shape
out=np.zeros((nGrains,8))
for grainNr in range(nGrains):
	grain = grains[grainNr,:]
	grCOM = grain[10:12]
	filtPts = micdata[micdata[:,0] == grainNr]
	nrPts,z = filtPts.shape
	avgPos = [np.mean(filtPts[:,3]),np.mean(filtPts[:,4])]
	out[grainNr,:] = np.array([grCOM[0],grCOM[1],avgPos[0],avgPos[1],sqrt((grCOM[0]-avgPos[0])**2+(grCOM[1]-avgPos[1])**2),grain[22],sqrt(nrPts*sqrt(3)/pi),grain[22]-sqrt(nrPts*sqrt(3)/pi)])

print(np.mean(out[:,4]))
print(out)
plt.scatter(out[:,4],out[:,6])
plt.show()

fig,ax = plt.subplots(1,1)
ax.scatter(micdata[:,3],micdata[:,4],c=micdata[:,0],cmap=plt.get_cmap('gray'))
ax.scatter(grains[:,10],grains[:,11],s=grains[:,22],c=range(1,nGrains+1),cmap=plt.get_cmap('gray'))
ax.set_aspect('equal')
plt.show()
