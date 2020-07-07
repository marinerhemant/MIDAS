import numpy as np
import os
import os.path as path
from PIL import Image

# ~ fName = '/data/tomo1/shastri_mar20_data/shastri_mar20/ge5/ss_ff_line_beam_000019.ge5'
# ~ dName = '/data/tomo1/shastri_mar20_data/shastri_mar20/ge5/dark_before_000018.ge5'
# ~ fName = '/data/tomo1/kenesei_preuss_nov18/data/ge5/Au_ff_000022.ge5'
# ~ dName = '/data/tomo1/kenesei_preuss_nov18/data/ge5/dark_before_000021.ge5'
fName = '/data/tomo1/mli_aug18_data/ge3/ss_sam_ff3_000394.ge3'
dName = '/data/tomo1/mli_aug18_data/ge3/dark_before_000393.ge3'
# ~ fName = '/data/tomo1/mli_aug18_data/ge3/ss_sam_ff3_000406.ge3'
# ~ dName = '/data/tomo1/mli_aug18_data/ge3/dark_before_000405.ge3'
# ~ fHead = 8192
fHead = 8396800
NrPixels = 2048
nFrames = 1440
thresh = 80

dark = np.zeros(NrPixels*NrPixels)
if path.exists(dName):
	darkf = open(dName,'rb')
	nFrames = int((os.path.getsize(dName) - 8192) / (2*NrPixels*NrPixels))
	darkf.seek(8192,os.SEEK_SET)
	for nr in range(nFrames):
		dark += np.fromfile(darkf,dtype=np.uint16,count=(NrPixels*NrPixels))
	dark /= nFrames
	dark = np.reshape(dark,(NrPixels,NrPixels))
	dark = dark.astype(float)
else:
	dark = np.zeros((NrPixels,NrPixels)).astype(float)

f = open(fName,'rb')
f.seek(fHead,os.SEEK_SET)
for frameNr in range(nFrames):
	print(frameNr)
	outFN =path.dirname(fName) + '/tiffs/' + path.basename(fName).replace('.ge3','') + '_FrameNr_'+ str(frameNr) + '.tif'
	thisFrame = np.fromfile(f,dtype=np.uint16,count=(NrPixels*NrPixels))
	thisFrame = np.reshape(thisFrame,(NrPixels,NrPixels))
	thisFrame = thisFrame.astype(float)
	thisFrame = thisFrame - dark
	thisFrame[thisFrame < thresh] = 0
	im = Image.fromarray(thisFrame)
	im.save(outFN,compression=None)
