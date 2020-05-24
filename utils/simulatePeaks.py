import numpy as np
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from math import sin, cos, exp, sqrt, atan, asin, acos
rad2deg = 57.2957795130823
deg2rad = 0.0174532925199

NrPixels = 2048
SpaceGroup = 225
LatticeParameter = [3.6, 3.6, 3.6, 90, 90, 90]
Wavelength = 0.22291
Lsd = 1000000
MaxRingRad = 204800
RingsToUse = [1,2,3]
nFrames = 200
SpotsFrame = 10 # guide for number of spots per frame, not necessary that it will be this much
px = 200
ycen = 1024
zcen = 1024

fPS = open('PS.txt','w')
fPS.write('SpaceGroup '+str(SpaceGroup)+'\n')
line = 'LatticeParameter ' + ' '.join(map(str,LatticeParameter))
fPS.write(line+'\n')
fPS.write('Wavelength '+str(Wavelength)+'\n')
fPS.write('Lsd '+str(Lsd)+'\n')
fPS.write('MaxRingRad '+str(MaxRingRad)+'\n')
fPS.close()

home = os.path.expanduser("~")
os.system(home+'/opt/MIDAS/FF_HEDM/bin/GetHKLList PS.txt')
fHKL = open('hkls.csv')
ringRadii = [ x for x in RingsToUse ]
for line in fHKL.readlines()[1:]:
	for itr, ringNr in enumerate(RingsToUse):
		if int(line.split()[4]) == int(ringNr):
			ringRadii[itr] = float(line.split()[-1])

fPeakInfo = open('PeakInfo.csv','w')
fPeakInfo.write('FrameNr,SpotNr,yPeak,zPeak,A,BG,Mu,yWidth,zWidth,Eta,RingRadius\n')
for frameNr in range(nFrames):
	print(frameNr)
	spotsThisFrame = random.randint(SpotsFrame//len(RingsToUse),2*(SpotsFrame//len(RingsToUse)))
	frame = np.zeros((NrPixels,NrPixels))
	outfn = 'Frame_' + str(frameNr) + '.tif'
	for idx, ringNr in enumerate(RingsToUse):
		spotsThisRing = random.randint(spotsThisFrame//2,spotsThisFrame)
		etaVar = (350/spotsThisRing - 10)/2
		etaMids = np.arange(-175+etaVar,175,(350-etaVar)/spotsThisRing)
		etas = [etaThis + np.random.uniform(-etaVar,etaVar) for etaThis in etaMids]
		yWidths = np.random.uniform(0,3,size=spotsThisRing)
		zWidths = np.random.uniform(0,3,size=spotsThisRing)
		As = np.random.uniform(50,1500,size=spotsThisRing)
		BGs = np.random.uniform(0,10,size=spotsThisRing)
		Mus = np.random.uniform(0,1,size=spotsThisRing)
		thisRad = ringRadii[idx]
		for spotNr in range(spotsThisRing):
			eta = etas[spotNr]
			yPeak = ycen + (sin(eta*deg2rad)*thisRad)/px
			zPeak = zcen + (cos(eta*deg2rad)*thisRad)/px
			yWidth = yWidths[spotNr]
			zWidth = zWidths[spotNr]
			tR = thisRad/px
			etaWidth = atan(yWidth/tR)*rad2deg
			rWidth =zWidth
			yExtent = yWidth*50
			zExtent = zWidth*50
			A = As[spotNr]
			BG = 0
			Mu = Mus[spotNr]
			for y in range(int(yPeak-yExtent),int(yPeak+yExtent+1)):
				if y < 0 or y >= NrPixels:
					continue
				for z in range(int(zPeak-zExtent),int(zPeak+zExtent+1)):
					if z < 0 or z >= NrPixels:
						continue
					radHere = sqrt((y-ycen)**2+(z-zcen)**2)
					etaThis = acos((z - zcen)/radHere)*rad2deg if eta > 0 else -acos((z - zcen)/radHere)*rad2deg
					etaDiff = (etaThis-eta)/etaWidth
					radDiff = (radHere-thisRad/px)/rWidth
					SigY = (etaDiff**2)
					SigZ = (radDiff**2)
					L = 1/(1 + SigY + SigZ)
					SigY = -0.5*(etaDiff**2)
					SigZ = -0.5*(radDiff**2)
					G = exp(SigY+SigZ)
					Int = int(BG + A*((Mu*L)+(1-Mu)*G))
					frame[y,z] += Int
					# ~ print([y,z,Int])
			fPeakInfo.write(str(frameNr)+','+str(spotNr)+','+str(yPeak)+','+str(zPeak)+','+str(A)+','+str(BG)+','+str(Mu)+','+str(yWidth)+','+str(zWidth)+','+str(eta)+','+str(thisRad)+'\n')
	im = Image.fromarray(frame)
	im.save(outfn,compression=None)

fPeakInfo.close()
