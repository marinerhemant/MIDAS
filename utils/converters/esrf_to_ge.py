import numpy as np
import h5py,hdf5plugin
import matplotlib.pyplot as plt
import os

def write_ge(data,fn):
	if os.path.exists(fn):
		os.remove(fn)
	f = open(fn,'a')
	arr = np.zeros(8192).astype(np.uint8)
	arr.tofile(f)
	data.astype(np.uint16).tofile(f)

fn = 'silicon_attrz-10.h5'
hf = h5py.File(fn,'r')
for key in hf.keys():
	data = np.array(hf[key]['instrument']['eiger']['data'])
	high_dim = np.max(data.shape[1:])
	low_dim = np.min(data.shape[1:])
	high_dim_pos = np.argmax(data.shape[1:])
	if high_dim_pos == 0:
		dim1_pad = 0
		dim2_pad = high_dim-low_dim
	else:
		dim2_pad = 0
		dim1_pad = high_dim-low_dim
	pad_tuple = [(0,0),(0,dim1_pad),(0,dim2_pad)]
	data_reshape = np.pad(data,pad_tuple,mode='constant',constant_values=0)
	del data
	med_img = np.median(data_reshape[:50,:,:],axis=0)
	outfn = fn.split('.')[0]+'_dataset_'+str(key)+'_'+str(1).zfill(6)+'.eiger.ge5'
	medfn = 'median_'+fn.split('.')[0]+'_dataset_'+str(key)+'_'+str(1).zfill(6)+'.eiger.ge5'
	write_ge(data_reshape,outfn)
	write_ge(med_img,medfn)
	
