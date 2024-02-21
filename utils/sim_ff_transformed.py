import numpy as np
import matplotlib.pyplot as plt
from math import ceil

def gaussian_2d(x, y, mu_x=0, mu_y=0, sigma_x=1, sigma_y=1,intensity=1):
	"""
	Compute the value of a 2D Gaussian function with means mu_x, mu_y and standard deviations sigma_x, sigma_y.
	"""
	factor = 1 / (2 * np.pi * sigma_x * sigma_y)
	exponent = np.exp(-((x - mu_x)**2 / (2 * sigma_x**2) + (y - mu_y)**2 / (2 * sigma_y**2)))
	return intensity * factor * exponent

nPxRad = 1500
nPxEta = 9424
img = np.zeros((nPxRad,nPxEta)).astype(np.uint16)
nRings = 30
sigmaEtaMax = 4
sigmaRMax = 2
nSigmas = 20
maxIntensity = 10000
minRad = 200
minEta = 10
rWidth = 0 # How wide should the peak centers building this peak be?
maxNPeaks = 50
maxEtaWidth = 200
maxPeaksRing = 100

rads = minRad + np.random.random(nRings)*(nPxRad-(minRad+100))
peakPositions = []
for ringNr in range(nRings):
	radCen = int(rads[ringNr])
	nPeaksRing = np.random.randint(0,maxPeaksRing)
	etaCens = minEta + np.random.random(nPeaksRing)*(nPxEta-(minEta+10))
	for peakRing in range(nPeaksRing):
		etaCen = int(etaCens[peakRing])
		# Add stretching based on radCen: for radCen=nPxRad, stretching should be 0, otherwise, we would have a factor for multiplying with etaWidth
		etaWidth = np.random.randint(20,maxEtaWidth) # How wide should the peak centers building this peak be?
		numPeaks = np.random.randint(2,maxNPeaks)
		for peakNr in range(numPeaks):
			peakCenRad = (radCen + np.random.random(1)*rWidth).item()
			peakCenEta = (etaCen + np.random.random(1)*etaWidth).item()
			rWidthPeak = 1 + np.random.random(1).item()*(sigmaRMax-1)
			etaWidthPeak = 2 + np.random.random(1).item()*(sigmaEtaMax-1)
			x = np.linspace(-int(nSigmas*ceil(rWidthPeak)),int(nSigmas*ceil(rWidthPeak)),endpoint=True,num=(2*int(nSigmas*ceil(rWidthPeak))+1))
			y = np.linspace(-int(nSigmas*ceil(etaWidthPeak)),int(nSigmas*ceil(etaWidthPeak)),endpoint=True,num=(2*int(nSigmas*ceil(etaWidthPeak))+1))
			X,Y = np.meshgrid(x,y)
			Z = gaussian_2d(X,Y,sigma_x=rWidthPeak,sigma_y=etaWidthPeak,intensity=np.random.randint(maxIntensity))
			xStart = int(peakCenRad)-int(nSigmas*ceil(rWidthPeak))
			yStart = int(peakCenEta)-int(nSigmas*ceil(etaWidthPeak))
			if xStart< 0: continue
			if yStart< 0: continue
			if xStart+x.shape[0]>nPxRad: continue
			if yStart+y.shape[0]>nPxEta: continue
			img[xStart:xStart+x.shape[0],yStart:yStart+y.shape[0]] += np.transpose(Z).astype(np.uint16)
			peakPositions.append([peakCenRad,peakCenEta])

peakPositions = np.array(peakPositions)
plt.imshow(np.log(img))
# plt.savefig('outout.png')
# plt.scatter(peakPositions[:,1],peakPositions[:,0])
plt.show()
