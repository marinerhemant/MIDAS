from PIL import Image
import numpy as np
import os

padding=6
layerNr = 1
fHead = 8192
NrPixels = 2048
folder = '/data/tomo1/shastri_mar20_midas/ff/ss_ff_line_beam_Layer1_Analysis_Time_2020_05_18_19_46_54/'
Rings = [1,2,3,4,5]
nFrames = 1440
fStem = 'ss_ff_line_beam'
fn = '/data/tomo1/shastri_mar20_data/shastri_mar20/ge5/ss_ff_line_beam_000019.ge5'
darkfn = '/data/tomo1/shastri_mar20_data/shastri_mar20/ge5/dark_before_000018.ge5'
window = 12 # Size of window to extract
peakInfoFN = folder + 'PeakInfo.csv'
piF = open(peakInfoFN,'w')
piF.write("FileName\txleft\txright\tybottom\tytop\txcen\tycen")

f = open(fn,'rb')
darkf = open(darkfn,'rb')
darkf.seek(fHead,os.SEEK_SET)
dark = np.fromfile(darkf,dtype=np.uint16,count=(NrPixels*NrPixels))
dark = np.reshape(dark,(NrPixels,NrPixels))
dark = dark.astype(float)

for fNr in range(1,nFrames+1):
	BytesToSkip = fHead + (fNr-1)*NrPixels*NrPixels*2
	f.seek(BytesToSkip,os.SEEK_SET)
	thisFrame = np.fromfile(f,dtype=np.uint16,count=(NrPixels*NrPixels))
	thisFrame = np.reshape(thisFrame,(NrPixels,NrPixels))
	thisFrame = thisFrame.astype(float)
	thisFrame = thisFrame - dark
	for ring in Rings:
		thisFN = folder+'Ring'+str(ring)+'/Temp/'+fStem+'_'+str(layerNr)+'_'+str(fNr).zfill(padding)+'_'+str(ring)+'_PS.csv'
		peaksData = np.genfromtxt(thisFN,skip_header=1)
		nPeaks = peaksData.size // 10
		for peakNr in range(nPeaks):
			if nPeaks > 1:
				peakInfo = peaksData[peakNr]
			else:
				peakInfo = peaksData
			xPos = int(peakInfo[4])
			yPos = int(peakInfo[3])
			thisPeakInfo = thisFrame[xPos-int(window/2):xPos+int(window/2+1),yPos-int(window/2):yPos+int(window/2+1)]
			thisExt = '_'+str(peakNr)+'.tif'
			outfn = thisFN.replace('.csv',thisExt)
			im = Image.fromarray(thisPeakInfo)
			im.save(outfn)
			piF.write(outfn+' '+str(xPos-int(window/2))+' '+str(xPos+int(window/2+1))+' '+str(yPos-int(window/2))+' '+str(yPos+int(window/2+1))+' '+str(peakInfo[4])+' '+str(peakInfo[3])+'\n')
