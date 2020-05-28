import numpy as np
import os
import os.path as path
from PIL import Image

fName = '/data/tomo1/shastri_mar20_data/shastri_mar20/ge5/ss_ff_line_beam_000019.ge5'
dName = '/data/tomo1/shastri_mar20_data/shastri_mar20/ge5/dark_before_000018.ge5'
fHead = 8192
NrPixels = 2048
nFrames = 1440
thresh = 100

if path.exists(dName):
	darkf = open(dName,'rb')
	darkf.seek(fHead,os.SEEK_SET)
	dark = np.fromfile(darkf,dtype=np.uint16,count=(NrPixels*NrPixels))
	dark = np.reshape(dark,(NrPixels,NrPixels))
	dark = dark.astype(float)
else:
	dark = np.zeros((NrPixels,NrPixels)).astype(float)

f = open(fName,'rb')
f.seek(fHead,os.SEEK_SET)
for frameNr in range(nFrames):
	print(frameNr)
	outFN =path.dirname(fName) + '/tiffs/' + path.basename(fName).replace('.ge5','') + '_FrameNr_'+ str(frameNr) + '.tif'
	thisFrame = np.fromfile(f,dtype=np.uint16,count=(NrPixels*NrPixels))
	thisFrame = np.reshape(thisFrame,(NrPixels,NrPixels))
	thisFrame = thisFrame.astype(float)
	thisFrame = thisFrame - dark
	thisFrame -= thresh
	thisFrame[thisFrame < 0] = 0
	im = Image.fromarray(thisFrame)
	im.save(outFN,compression=None)
