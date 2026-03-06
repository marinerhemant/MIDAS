import numpy as np
from math import sqrt

folder = '/data/tomo1/pilatus_midas/analysis/'
fstems = ['ruby_pilatus_61keV.csv','ruby_pilatus_71keV.csv','ruby_pilatus_81keV.csv','ruby_pilatus_91keV.csv']
nFrames = 18000
startOme = 180
omeStep = -0.02
overlapLen = 1.4

def distt(a,b,c,d):
	return sqrt((a-c)**2+(b-d)**2)

for fstem in fstems:
	fname = folder + fstem
	outfn = fname+'.merged.csv'
	outf = open(outfn,'w')
	outf.write('PeakNr\tyCen\tzCen\tOmega\tIntensity\tStartFrame\tEndFrame\tNrPixels\n')
	peaksData = np.genfromtxt(fname,skip_header=1)
	peaksList = []
	ctrx = 1
	for frameNr in range(0,nFrames):
		thisOme = startOme + omeStep*frameNr
		thisPeaks = peaksData[peaksData[:,0] == frameNr]
		if thisPeaks.shape[0] > 0:
			# calculate and update peak info here
			if len(peaksList) == 0:
				for thisPeak in thisPeaks:
					peaksList.append([thisPeak[2],thisPeak[3],thisOme,thisPeak[4],frameNr,frameNr,int(thisPeak[1])])
			else:
				for thisPeak in thisPeaks:
					pF = 0
					for ctr,peaks in enumerate(peaksList):
						if (distt(thisPeak[2],thisPeak[3],peaks[0],peaks[1]) < overlapLen):
							totInt = peaks[3] + thisPeak[4]
							peaksList[ctr][0] = (thisPeak[2]*thisPeak[4] + peaks[0]*peaks[3])/totInt # y
							peaksList[ctr][1] = (thisPeak[3]*thisPeak[4] + peaks[1]*peaks[3])/totInt # z
							peaksList[ctr][2] = (thisOme*thisPeak[4] + peaks[2]*peaks[3])/totInt # ome
							peaksList[ctr][3] = totInt
							peaksList[ctr][5] = frameNr
							peaksList[ctr][6] += int(thisPeak[1])
							pF = 1
					if pF == 0:
						peaksList.append([thisPeak[2],thisPeak[3],thisOme,thisPeak[4],frameNr,frameNr,int(thisPeak[1])])
		else:
			# write out the peak info here
			for peaks in peaksList:
				outf.write(str(ctrx)+'\t'+str(peaks[0])+'\t'+str(peaks[1])+'\t'+str(peaks[2])+'\t'+str(peaks[3])+'\t'+str(peaks[4])+'\t'+str(peaks[5])+'\t'+str(peaks[6])+'\n') # y, z, ome, intensity, startFrame, endFrame, totNrPx
				ctrx += 1
			peaksList = []
