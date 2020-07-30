import numpy as np
import os
import os.path as path
from PIL import Image

# ~ fName = '/data/tomo1/shastri_mar20_data/shastri_mar20/ge5/ss_ff_line_beam_000019.ge5'
# ~ dName = '/data/tomo1/shastri_mar20_data/shastri_mar20/ge5/dark_before_000018.ge5'
# ~ fName = '/data/tomo1/AI-HEDM/Training_Dataset/NoTwins_Microstructure_4/FF_Sim_Cycle4_Orig_MoreRings_000001.ge3'
# ~ dName = '/data/tomo1/AI-HEDM/Training_Dataset/NoTwins_Microstructure_4/dark_before_000018.ge5'
# ~ fName = '/data/tomo1/kenesei_preuss_nov18/data/ge5/Au_ff_000022.ge5'
# ~ dName = '/data/tomo1/kenesei_preuss_nov18/data/ge5/dark_before_000021.ge5'
# ~ fName = '/data/tomo1/mli_aug18_data/ge3/ss_sam_ff3_000394.ge3'
# ~ dName = '/data/tomo1/mli_aug18_data/ge3/dark_before_000393.ge3'
# ~ fName = '/data/tomo1/mli_aug18_data/ge3/ss_sam_ff3_000406.ge3'
# ~ dName = '/data/tomo1/mli_aug18_data/ge3/dark_before_000405.ge3'
fStem = '/data/tomo1/PUP_AFRL_Dec14_data/GE/Ti7_23_Crack1_'
dName = '/data/tomo1/PUP_AFRL_Dec14_data/GE/dark_0pt3s_00026.ge3'
outfolder = 'tiffs'
firstFN = 531
padding = 5
nrFiles = 6
ext = '.ge3'
fHead = 8192
# ~ fHead = 8396800
NrPixels = 2048
nFrames = 1440
thresh = 80

dark = np.zeros(NrPixels*NrPixels)
if path.exists(dName):
	darkf = open(dName,'rb')
	nFramesDark = int((os.path.getsize(dName) - 8192) / (2*NrPixels*NrPixels))
	darkf.seek(8192,os.SEEK_SET)
	for nr in range(nFramesDark):
		dark += np.fromfile(darkf,dtype=np.uint16,count=(NrPixels*NrPixels))
	dark /= nFramesDark
	dark = np.reshape(dark,(NrPixels,NrPixels))
	dark = dark.astype(float)
else:
	dark = np.zeros((NrPixels,NrPixels)).astype(float)

for fnr in range(nrFiles):
	thisFN = firstFN + fnr
	startFrameNr = (nFrames//nrFiles)*fnr
	endFrameNr = (nFrames//nrFiles)*(fnr+1)
	fName = fStem + '_' + str(thisFN).zfill(padding) + ext
	f = open(fName,'rb')
	f.seek(fHead,os.SEEK_SET)
	for frameNr in range(startFrameNr,endFrameNr):
		print([frameNr,nFrames])
		outFN =path.dirname(fName) + '/tiffs/' + path.basename(fName).replace('.ge3','') + '_FrameNr_'+ str(frameNr) + '.tif'
		thisFrame = np.fromfile(f,dtype=np.uint16,count=(NrPixels*NrPixels))
		thisFrame = np.reshape(thisFrame,(NrPixels,NrPixels))
		thisFrame = thisFrame.astype(float)
		thisFrame = thisFrame - dark
		thisFrame[thisFrame < thresh] = 0
		im = Image.fromarray(thisFrame)
		im.save(outFN,compression=None)
