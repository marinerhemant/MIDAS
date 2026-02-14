import numpy as np
import calcMiso
import os,sys
try:
    import midas_config
    utilsDir = midas_config.MIDAS_UTILS_DIR
except ImportError:
    utilsDir = os.path.expanduser('~/opt/MIDAS/utils/')
sys.path.insert(0,utilsDir)
import matplotlib.pyplot as plt
from scipy import ndimage
from math import sqrt, pi

def posDiff(a,b):
	return sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def getPosDiff(grs,outputMat):
	diffX = []
	diffY = []
	posXGrs = []
	posYGrs = []
	grSzs = []
	posE = []
	posI = []
	notMatched = []
	grout = np.zeros(grs.shape)
	ct = 0
	for grains in outputMat:
		euler1 = grains[4:]
		totSize = 0
		totPosX = 0
		totPosY = 0
		minA = 1
		minDist = 200
		for gr in grs:
			euler2 = gr[-3:]
			ang = rad2deg*calcMiso.GetMisOrientationAngle(euler1,euler2,sgNum)
			if ang < minA:
				minDist = posDiff([grains[2],grains[3]],[gr[10],gr[11]])
				thisSize = gr[22]
				totPosX = totPosX*totSize + gr[10]*thisSize
				totPosY = totPosY*totSize + gr[11]*thisSize
				totSize += thisSize
				totPosX /= totSize
				totPosY /= totSize
				bestPosE = gr[19]
				bestIA = gr[21]
				bestgr = gr
		if totPosX != 0:
			diffX.append(grains[2] - totPosX)
			diffY.append(grains[3] - totPosY)
			posXGrs.append(totPosX)
			posYGrs.append(totPosY)
			grSzs.append(sqrt(grains[1]/pi))
			posE.append(bestPosE)
			posI.append(bestIA)
			grout[ct,:] = bestgr
			ct+=1
		else:
			notMatched.append(grains[0])
	grout = grout[:ct,:]
	return((diffX,diffY,posXGrs,posYGrs,grSzs,posE,posI,notMatched,grout))

def calcUniqueGrains(grs):
	minAng = 0.5
	retArr = np.zeros(grs.shape)
	mm = np.zeros(grs.shape[0])
	ctr = 0
	for idx,grain in enumerate(grs):
		if (mm[idx]) != 0:
			continue
		eul1 = grain[-3:]
		for idx2 in range(idx+1,grs.shape[0]):
			eul2 = grs[idx2,-3:]
			ang = rad2deg*calcMiso.GetMisOrientationAngle(eul1,eul2,sgNum)
			if (ang < minAng):
				mm[idx2] = 1
		retArr[ctr,:] = grain
		ctr += 1
	retArr = retArr[:ctr,:]
	return retArr

def calcUnmatchedGrains(grs1,grs2):
	retarr = np.zeros(grs1.shape)
	ct = 0
	for grain1 in grs1:
		minAng = 100
		eul1 = grain1[-3:]
		for grain2 in grs2:
			eul2 = grain2[-3:]
			ang = rad2deg*calcMiso.GetMisOrientationAngle(eul1,eul2,sgNum)
			if (ang < minAng):
				minAng = ang
				bestGr = grain2
		if minAng > 2:
			retarr[ct,:] = grain1
			ct+=1
	retarr = retarr[:ct,:]
	return retarr

rad2deg = 57.2957795130823
fillVal = -15
sgNum = 194
minAngle = 0.5
cutconfidencevar = 0.9
micfile = 'MicrostructureText_Layer6.mic.map'
grainFN = 'Grains66CONew.csv'
grains2 = 'Grains66_r0_orig.csv'
grains3 = 'Grains66_r1_orig.csv'
if (micfile[-3:] != 'map'):
	print('Need to supply map file.')
f = open(micfile,'r')
if (micfile[-3:] == 'map'):
	micfiletype = 2
	sizeX = int(np.fromfile(f,dtype=np.double,count=1)[0])
	sizeY = int(np.fromfile(f,dtype=np.double,count=1)[0])
	refX = int(np.fromfile(f,dtype=np.double,count=1)[0])
	refY = int(np.fromfile(f,dtype=np.double,count=1)[0])
	micfiledata = np.fromfile(f,dtype=np.double)
	if (micfiledata.size/7) != (sizeX*sizeY):
		print("Size of the map file is not correct. Please check that the file was written properly.")
exten = [refX,refX+sizeX,refY+sizeY,refY]
micfiledatacut = np.copy(micfiledata)
badcoords = micfiledatacut[:sizeX*sizeY]
badcoords = badcoords < cutconfidencevar
eul1 = micfiledata[sizeX*sizeY:sizeX*sizeY*2]
eul2 = micfiledata[sizeX*sizeY*2:sizeX*sizeY*3]
eul3 = micfiledata[sizeX*sizeY*3:sizeX*sizeY*4]
grainID = micfiledata[sizeX*sizeY*4:sizeX*sizeY*5]
confidence = micfiledata[:sizeX*sizeY]
eul1[badcoords] = fillVal
eul2[badcoords] = fillVal
eul3[badcoords] = fillVal
grainID[badcoords] = fillVal
eul1 = eul1.reshape((sizeX,sizeY))
eul2 = eul2.reshape((sizeX,sizeY))
eul3 = eul3.reshape((sizeX,sizeY))
grainID = grainID.reshape((sizeX,sizeY))
confidence = confidence.reshape((sizeX,sizeY))
# ~ eul1[0:-486-exten[0]+1,:] = fillVal
# ~ eul1[503-exten[0]:,:] = fillVal
# ~ eul1[:,0:-492-exten[3]+1] = fillVal
# ~ eul1[:,495-exten[3]:] = fillVal
# ~ eul2[0:-486-exten[0]+1,:] = fillVal
# ~ eul2[503-exten[0]:,:] = fillVal
# ~ eul2[:,0:-492-exten[3]+1] = fillVal
# ~ eul2[:,495-exten[3]:] = fillVal
# ~ eul3[0:-486-exten[0]+1,:] = fillVal
# ~ eul3[503-exten[0]:,:] = fillVal
# ~ eul3[:,0:-492-exten[3]+1] = fillVal
# ~ eul3[:,495-exten[3]:] = fillVal
# ~ grainID[0:-486-exten[0]+1,:] = fillVal
# ~ grainID[503-exten[0]:,:] = fillVal
# ~ grainID[:,0:-492-exten[3]+1] = fillVal
# ~ grainID[:,495-exten[3]:] = fillVal
# ~ confidence[0:-486-exten[0]+1,:] = fillVal
# ~ confidence[503-exten[0]:,:] = fillVal
# ~ confidence[:,0:-492-exten[3]+1] = fillVal
# ~ confidence[:,495-exten[3]:] = fillVal
grainNrs = np.unique(grainID)
orientationList = []
for grainNr in grainNrs:
	if grainNr == fillVal:
		continue
	y,x = ndimage.measurements.center_of_mass(grainID==grainNr)
	orientationList.append([int(grainNr),eul1[grainID==grainNr][0],eul2[grainID==grainNr][0],eul3[grainID==grainNr][0],np.sum(grainID == grainNr),x+refX,y+refY])
nrOrients = grainNrs.size
nrOrients -= 1
matchedMat = np.zeros(nrOrients)
outputMat = np.zeros((nrOrients,7))
ctr = 0
for idx,orient in enumerate(orientationList):
	if matchedMat[idx] != 0:
		continue
	outputMat[ctr,0] = orientationList[idx][0]
	outputMat[ctr,1] = orientationList[idx][4]
	outputMat[ctr,2] = orientationList[idx][5]
	outputMat[ctr,3] = orientationList[idx][6]
	outputMat[ctr,4] = orientationList[idx][1]
	outputMat[ctr,5] = orientationList[idx][2]
	outputMat[ctr,6] = orientationList[idx][3]
	for idx2 in range(idx+1,nrOrients):
		if matchedMat[idx2] != 0:
			continue
		euler1 = orient[1:4]
		euler2 = orientationList[idx2][1:4]
		angle = rad2deg*calcMiso.GetMisOrientationAngle(euler1,euler2,sgNum)
		if (angle < minAngle):
			outputMat[ctr,2] = (outputMat[ctr,2]*outputMat[ctr,1] + orientationList[idx2][4]*orientationList[idx2][5])
			outputMat[ctr,3] = (outputMat[ctr,3]*outputMat[ctr,1] + orientationList[idx2][4]*orientationList[idx2][6])
			outputMat[ctr,1] += orientationList[idx2][4]
			outputMat[ctr,3] /= outputMat[ctr,1]
			outputMat[ctr,2] /= outputMat[ctr,1]
			matchedMat[idx2] = idx+1
			matchedMat[idx] = idx2
	ctr += 1

outputMat = outputMat[:ctr,:]

grs = np.genfromtxt(grainFN,skip_header=9)
# ~ grs[:,44] /= rad2deg
# ~ grs[:,45] /= rad2deg
# ~ grs[:,46] /= rad2deg
dx,dy,px,py,g1,posE1,posI1,nm1,gp1 = getPosDiff(grs,outputMat)
ab1 = [sqrt(a*a+b*b) for a,b in zip(dx,dy)]
# ~ print(np.mean(np.abs(dx)),np.mean(np.abs(dy)),np.mean(ab1),np.median(ab1))
gr2 = np.genfromtxt(grains2,skip_header=9)
dx2,dy2,px2,py2,g2,posE2,posI2,nm2,gp2 = getPosDiff(gr2,outputMat)
ab2 = [sqrt(a*a+b*b) for a,b in zip(dx2,dy2)]
# ~ print(np.mean(np.abs(dx2)),np.mean(np.abs(dy2)),np.mean(ab2),np.median(ab2))
gr3 = np.genfromtxt(grains3,skip_header=9)
dx3,dy3,px3,py3,g3,posE3,posI3,nm3,gp3 = getPosDiff(gr3,outputMat)
ab3 = [sqrt(a*a+b*b) for a,b in zip(dx3,dy3)]
# ~ print(np.mean(np.abs(dx3)),np.mean(np.abs(dy3)),np.mean(ab3),np.median(ab3))
# ~ print(len(gp1),len(gp2),len(gp3))
print(np.mean(posI1),np.mean(posE1))
print(np.mean(posI3),np.mean(posE3))
unique1 = calcUniqueGrains(grs).shape[0]
unique2 = calcUniqueGrains(gr2).shape[0]
unique3 = calcUniqueGrains(gr3).shape[0]
# ~ print(unique1,unique2,unique3)

pltdata = confidence
fig,axs = plt.subplots(2,3)
axs[0,0].imshow(np.ma.masked_where(pltdata==fillVal,pltdata),extent=exten,cmap=plt.get_cmap('bone'))
axs[0,0].scatter(outputMat[:,2],outputMat[:,3],marker='s',c='black',label='NF')
axs[0,0].scatter(px,py,marker='o',c='green',label='CO')
axs[0,0].scatter(px2,py2,marker='8',c='red',label='R0')
axs[0,0].scatter(px3,py3,marker='^',c='blue',label='R1')
axs[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axs[0,0].set_title('Grain positions')
p1 = axs[0,1].scatter(g1,ab1,s=np.array(posI1)*500,c=posE1)
axs[0,1].set_title('CO mean: L2:{0:.3f}'.format(np.mean(ab1)))#', PosError:{1:.3f}, IA:{2:.3f}'.format(np.mean(ab1),np.mean(posE1),np.mean(posI1)))
axs[0,1].set(xlabel ='Grain Size',ylabel='L2-norm error',xlim=[0,170],ylim=[0,90])
fig.colorbar(p1,ax=axs[0,1])
p3 = axs[0,2].scatter(g3,ab3,s=np.array(posI3)*500,c=posE3)
axs[0,2].set_title('R1 mean: L2:{0:.3f}'.format(np.mean(ab3)))#, PosError:{1:.3f}, IA:{2:.3f}'.format(np.mean(ab3),np.mean(posE3),np.mean(posI3)))
axs[0,2].set(xlabel ='Grain Size',ylabel='L2-norm error',xlim=[0,170],ylim=[0,90])
fig.colorbar(p3,ax=axs[0,2])
# ~ p2 = axs[1,0].scatter(g2,ab2,s=np.array(posI2)*500,c=posE2)
# ~ axs[1,0].set_title('R0 mean: L2:{0:.3f}'.format(np.mean(ab2)))#, PosError:{1:.3f}, IA:{2:.3f}'.format(np.mean(ab2),np.mean(posE2),np.mean(posI2)))
# ~ axs[1,0].set(xlabel ='Grain Size',ylabel='L2-norm error',xlim=[0,170],ylim=[0,90])
# ~ fig.colorbar(p2,ax=axs[1,0])

### Plot the missing grains
r1 = calcUnmatchedGrains(gp1,gp2)
dx,dy,px,py,g1,posE1,posI1,nm1,gp1 = getPosDiff(r1,outputMat)
axs[1,0].imshow(np.ma.masked_where(pltdata==fillVal,pltdata),extent=exten,cmap=plt.get_cmap('bone'))
axs[1,0].scatter(r1[:,10],r1[:,11])
# ~ print(r1)

plt.show()
