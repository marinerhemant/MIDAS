import numpy as np
import os
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import h5py

nFrames = 21
NrPixelsY = 2048
NrPixelsZ = 2048
nTotalPixels = NrPixelsY * NrPixelsZ
fHead = 8192 + 2*2048*2048
thresh = 100
fn = 'shade_LSHR_voi_ff_000302.ge3'
darkfn = 'dark_after_000295.ge3'
window = 7

hf = h5py.File('patches.h5','w')

f = open(fn,'rb')
darkf = open(darkfn,'rb')
darkf.seek(fHead+nTotalPixels*2,os.SEEK_SET)
dark = np.fromfile(darkf,dtype=np.uint16,count=nTotalPixels)
dark = np.reshape(dark,(NrPixelsZ,NrPixelsY))
dark = dark.astype(float)
darkf.close()

for fNr in range(1,nFrames):
	BytesToSkip = fHead + fNr*nTotalPixels*2
	f.seek(BytesToSkip,os.SEEK_SET)
	thisFrame = np.fromfile(f,dtype=np.uint16,count=nTotalPixels)
	thisFrame = np.reshape(thisFrame,(NrPixelsZ,NrPixelsY))
	thisFrame = thisFrame.astype(float)
	thisFrame = thisFrame - dark
	thisFrame[thisFrame < thresh] = 0
	thisFrame = thisFrame.astype(int)
	thisFrame2 = np.copy(thisFrame)
	thisFrame2[thisFrame2>0] = 1
	labels = label(thisFrame2)
	regions = regionprops(labels)
	patches = []
	xy_positions = []
	for prop_nr,props in enumerate(regions):
		if props.area < 4 or props.area > 150:
			continue
		y0,x0   = props.centroid
		start_x = int(x0)-window
		end_x   = int(x0)+window+1
		start_y = int(y0)-window
		end_y   = int(y0)+window+1
		if start_x < 0 or end_x > NrPixelsY - 1 or start_y < 0 or end_y > NrPixelsZ - 1:
			continue
		sub_img = np.copy(thisFrame)
		sub_img[labels != prop_nr+1] = 0
		sub_img = sub_img[start_y:end_y,start_x:end_x]
		patches.append(sub_img)
		xy_positions.append([start_y,start_x])
	patches = np.array(patches)
	xy_positions = np.array(xy_positions)
	hf.create_dataset('frame_nr'+str(fNr),data=patches)
	print(patches.shape)

hf.close()
