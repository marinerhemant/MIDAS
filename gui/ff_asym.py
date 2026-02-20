#!/usr/bin/env python

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as Tk
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import tempfile
import tkinter.filedialog as tkFileDialog
import math
from math import sin, cos, sqrt
from numpy import linalg as LA
import subprocess
from multiprocessing.dummy import Pool
import h5py
import bz2
import shutil
import threading
try:
    import tifffile
except ImportError:
    tifffile = None

import sys

# Try to import midas_config from utils
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    utils_dir = os.path.join(os.path.dirname(current_dir), 'utils')
    if utils_dir not in sys.path:
        sys.path.append(utils_dir)
    import midas_config
except ImportError as e:
    print(f"Warning: Could not import midas_config: {e}")
    midas_config = None

# Helpers
deg2rad = 0.0174532925199433
rad2deg = 57.2957795130823

import itertools
import glob
_color_cycle = itertools.cycle(['r','g','b','c','m','y'])

# Auto-detect files from current directory name
_auto_fileStem = None
_auto_folder = None
_auto_padding = None
_auto_firstFileNr = None
_auto_ext = None
_auto_darkStem = None
_auto_darkNum = None
_auto_nFramesPerFile = None

def _try_auto_detect():
    """Try to auto-detect data and dark files from the current working directory.
    
    Convention: if cwd is /path/to/ff_Holder3_50um, look for files starting with
    ff_Holder3_50um in the current directory. The folder name is the prefix for
    data files. Dark files start with dark_before or dark_after.
    """
    global _auto_fileStem, _auto_folder, _auto_padding, _auto_firstFileNr
    global _auto_ext, _auto_darkStem, _auto_darkNum, _auto_nFramesPerFile
    
    cwd = os.getcwd()
    dir_basename = os.path.basename(cwd)
    
    if not dir_basename:
        return False
    
    # Find files that start with the directory name
    all_files = sorted(os.listdir(cwd))
    
    # Find data files: files starting with dir_basename
    data_files = []
    for f in all_files:
        name_no_ext = os.path.splitext(f)[0]
        # File stem must start with dir_basename and have _NNNNNN pattern
        if name_no_ext.startswith(dir_basename + '_'):
            # Extract the trailing number
            parts = name_no_ext.split('_')
            if parts[-1].isdigit():
                data_files.append(f)
    
    if not data_files:
        print(f"Auto-detect: no files starting with '{dir_basename}_' found in {cwd}")
        return False
    
    # Use the first (smallest number) data file
    first_file = data_files[0]
    basename_full = os.path.basename(first_file)
    dot_idx = basename_full.find('.')
    if dot_idx == -1:
        return False
    name_part = basename_full[:dot_idx]
    ext_part = basename_full[dot_idx+1:]  # e.g. 'tif', 'ge5'
    
    parts = name_part.split('_')
    if not parts[-1].isdigit():
        return False
    
    _auto_firstFileNr = int(parts[-1])
    _auto_padding = len(parts[-1])
    _auto_fileStem = '_'.join(parts[:-1])
    _auto_folder = cwd + '/'
    _auto_ext = ext_part
    
    # Calculate nFramesPerFile from file size for binary formats
    _auto_nFramesPerFile = 1  # default for tiff
    
    print(f"Auto-detect: stem='{_auto_fileStem}', firstNr={_auto_firstFileNr}, "
          f"padding={_auto_padding}, ext='{_auto_ext}'")
    
    # Find dark files: prefer dark_before, fallback to dark_after
    dark_before = []
    dark_after = []
    for f in all_files:
        name_no_ext = os.path.splitext(f)[0]
        parts = name_no_ext.split('_')
        if len(parts) >= 2 and parts[-1].isdigit():
            prefix = '_'.join(parts[:-1])
            if prefix == 'dark_before':
                dark_before.append(f)
            elif prefix == 'dark_after':
                dark_after.append(f)
    
    dark_candidates = dark_before if dark_before else dark_after
    if dark_candidates:
        dark_file = dark_candidates[0]  # first (smallest number)
        dark_basename = os.path.splitext(os.path.basename(dark_file))[0]
        dark_parts = dark_basename.split('_')
        if dark_parts[-1].isdigit():
            _auto_darkNum = int(dark_parts[-1])
            _auto_darkStem = '_'.join(dark_parts[:-1])
            source = 'dark_before' if dark_before else 'dark_after'
            print(f"Auto-detect: dark='{_auto_darkStem}_{str(_auto_darkNum).zfill(len(dark_parts[-1]))}', source={source}")
    else:
        print("Auto-detect: no dark files found (dark_before_*.* or dark_after_*.*)")
    
    return True

_auto_detected = False  # Will be set True by background thread

def _apply_auto_detect(root_widget):
    """Callback to apply auto-detected values to Tk variables. Called from main thread via root.after()."""
    global fileStem, folder, padding, darkStem, darkNum, dark, firstFileNumber, nFramesPerFile
    global nDetectors, startDetNr, endDetNr, nFilesPerLayer, _auto_detected
    if _auto_fileStem:
        _auto_detected = True
        fileStem = _auto_fileStem
        folder = _auto_folder
        padding = _auto_padding
        firstFileNumber = _auto_firstFileNr
        nFramesPerFile = _auto_nFramesPerFile if _auto_nFramesPerFile else 1
        firstFileNrVar.set(str(firstFileNumber))
        nFramesPerFileVar.set(str(nFramesPerFile))
        fnextvar.set(_auto_ext if _auto_ext else 'tif')
        if _auto_ext and _auto_ext.startswith('ge') and len(_auto_ext) == 3 and _auto_ext[-1].isdigit():
            detnumbvar.set(_auto_ext[-1])
        else:
            detnumbvar.set('-1')
        if _auto_darkStem:
            darkStem = _auto_darkStem
            darkNum = _auto_darkNum
            var.set(1)
        root_widget.wm_title(root_widget.wm_title().replace(' [scanning...]', '') + ' [files detected]')
        print("Auto-detection complete. GUI updated.")
    else:
        root_widget.wm_title(root_widget.wm_title().replace(' [scanning...]', ''))
        print("Could not auto-detect files. Please select files manually.")

def _start_auto_detect_thread(root_widget):
    """Run auto-detection in a background thread to avoid blocking the GUI."""
    t = threading.Thread(target=_try_auto_detect, daemon=True)
    t.start()
    def _poll():
        if t.is_alive():
            root_widget.after(200, _poll)  # Check again in 200ms
        else:
            _apply_auto_detect(root_widget)
    root_widget.after(200, _poll)

def get_ring_colors(n):
	return [next(_color_cycle) for _ in range(n)]

def _quit():
	root.quit()
	root.destroy()

def CalcEtaAngle(XYZ):
	alpha = rad2deg*np.arccos(np.divide(XYZ[2,:],LA.norm(XYZ[1:,:],axis = 0)))
	alpha[XYZ[1,:]>0] = -alpha[XYZ[1,:]>0]
	return alpha

def CalcEtaAngleRad(y,z):
	Rad = sqrt(y*y+z*z)
	alpha = rad2deg*math.acos(z/Rad)
	if y > 0:
		alpha = -alpha
	return [alpha,Rad]

def YZ4mREta(R,Eta):
	return [-R*sin(Eta*deg2rad),R*cos(Eta*deg2rad)]

def getfn(fstem,fnum,geNum):
	if sepfolderVar.get():
		fldr = folder + '/ge' + str(geNum) + '/'
	else:
		fldr = folder
	if geNum != -1:
		return fldr + fstem + '_' + str(fnum).zfill(padding) + '.ge' + str(geNum)
	else:
		return fldr + fstem + '_' + str(fnum).zfill(padding) + '.' + fnextvar.get()


def get_bz2_data(fn):
	# Decompress to temp file, preserving original extension for format detection
	# e.g., "image.tif.bz2" -> strip ".bz2" -> get ".tif" suffix
	inner_name = fn[:-4] if fn.lower().endswith('.bz2') else fn
	suffix = os.path.splitext(inner_name)[1]  # e.g., '.tif'
	with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
		with bz2.BZ2File(fn, 'rb') as source:
			shutil.copyfileobj(source, tmp)
		temp_name = tmp.name
	return temp_name

lastMaskState = None

def readMask():
	global badPixelMask, lastMaskState
	fn = maskFNVar.get()
	if not fn or not os.path.exists(fn):
		badPixelMask = None
		lastMaskState = None
		return
	
	try:
		currentState = (fn, transpose.get(), hflip.get(), vflip.get(), NrPixelsY, NrPixelsZ)
		if badPixelMask is not None and lastMaskState == currentState:
			return

		# Assume int8 binary mask as per user request
		mask = np.fromfile(fn, dtype=np.int8, count=NrPixelsY*NrPixelsZ)
		badPixelMask = mask.reshape((NrPixelsY, NrPixelsZ))
		
		# Handle flips to match image data
		# Note: Image data logic: transpose -> flip_h -> flip_v (or h/v/transpose?)
		# getImage logic: 
		# 1. transpose
		# 2. flip_h and flip_v reverse
		
		if transpose.get() == 1:
			badPixelMask = np.transpose(badPixelMask)
		flip_h = hflip.get() == 1
		flip_v = vflip.get() == 1
		if flip_h and flip_v:
			badPixelMask = badPixelMask[::-1, ::-1].copy()
		elif flip_h:
			badPixelMask = badPixelMask[::-1, :].copy()
		elif flip_v:
			badPixelMask = badPixelMask[:, ::-1].copy()
			
		lastMaskState = currentState
	except Exception as e:
		print(f"Error reading mask: {e}")
		badPixelMask = None
		lastMaskState = None

def getImage(fn, bytesToSkip, frame_idx=0, is_dark=False):
	print("Reading file: " + fn)
	global Header, BytesPerPixel, NrPixelsY, NrPixelsZ
	Header = HeaderVar.get()
	BytesPerPixel = BytesVar.get()
	
	# Compression handling
	if fn.endswith('.bz2'):
		# Decompress and read (recursive call with decompressed file)
		temp_fn = get_bz2_data(fn)
		try:
			data = getImage(temp_fn, bytesToSkip, frame_idx, is_dark)
		finally:
			if os.path.exists(temp_fn):
				os.remove(temp_fn)
		return data
		
	# Format handling
	ext = os.path.splitext(fn)[1].lower()
	
	if ext in ['.h5', '.hdf', '.hdf5', '.nxs']:
		# HDF5
		with h5py.File(fn, 'r') as f:
			if is_dark:
				dset_path = hdf5DarkPathVar.get()
			else:
				dset_path = hdf5PathVar.get()
				
			if dset_path in f:
				# Assume dataset is 3D (frames, y, x) or 2D (y, x)
				dset = f[dset_path]
				if dset.ndim == 2:
					data = dset[:]
				elif dset.ndim == 3:
					data = dset[frame_idx, :, :]
				else:
					print(f"Error: HDF5 dataset {dset_path} has {dset.ndim} dimensions, expected 2 or 3.")
					return np.zeros((NrPixelsY, NrPixelsZ))
			else:
				print(f"Error: HDF5 dataset {dset_path} not found in {fn}")
				return np.zeros((NrPixelsY, NrPixelsZ))

	elif ext in ['.tif', '.tiff']:
		# TIFF
		if tifffile:
			data = tifffile.imread(fn, key=frame_idx)
		else:
			print("Error: tifffile module not found. Please install it.")
			return np.zeros((NrPixelsY, NrPixelsZ))
			
	else:
		# Binary (Default)
		f = open(fn,'rb')
		f.seek(bytesToSkip,os.SEEK_SET)
		if BytesPerPixel == 2:
			data = np.fromfile(f,dtype=np.uint16,count=(NrPixelsY*NrPixelsZ))
		elif BytesPerPixel == 4:
			data = np.fromfile(f,dtype=np.int32,count=(NrPixelsY*NrPixelsZ))
		f.close()
		data = np.reshape(data,(NrPixelsY,NrPixelsZ))
	
	# Auto-update pixel dimensions from self-describing formats (HDF5, TIFF)
	if ext in ['.h5','.hdf','.hdf5','.nxs','.tif','.tiff']:
		if data.shape[0] != NrPixelsY or data.shape[1] != NrPixelsZ:
			NrPixelsY = data.shape[0]
			NrPixelsZ = data.shape[1]
			NrPixelsYVar.set(str(NrPixelsY))
			NrPixelsZVar.set(str(NrPixelsZ))

	data = data.astype(float)
	if transpose.get() == 1:
		data = np.transpose(data)
	flip_h = hflip.get() == 1
	flip_v = vflip.get() == 1
	if flip_h and flip_v:
		data = data[::-1, ::-1].copy()
	elif flip_h:
		data = data[::-1, :].copy()
	elif flip_v:
		data = data[:, ::-1].copy()
	if applyMaskVar.get() and maskFNVar.get():
		readMask()
		if badPixelMask is not None:
			if badPixelMask.shape == data.shape:
				data[badPixelMask == 1] = 0
				
	return data

def getImageMax(fn):
	print("Calculating Max for file: " + fn)
	global Header, BytesPerPixel
	Header = HeaderVar.get()
	BytesPerPixel = BytesVar.get()
	nFramesToDo = nFramesMaxVar.get()
	startFrameNr = maxStartFrameNrVar.get()
	
	t1 = time.time()
	dataMax = None
	ext = os.path.splitext(fn)[1].lower()
	
	# For single-page TIFF files, iterate over sequential numbered files
	# For multi-frame files (binary, HDF5, multi-page TIFF), iterate frames within file
	is_per_file = False
	if ext in ['.tif', '.tiff'] and tifffile:
		try:
			with tifffile.TiffFile(fn) as tif:
				if len(tif.pages) == 1:
					is_per_file = True
		except Exception:
			pass

	if is_per_file:
		# Extract base name pattern: stem_NNNNNN.ext -> (stem_, 6, .ext)
		import re
		basename = os.path.basename(fn)
		dirname = os.path.dirname(fn)
		m = re.match(r'^(.*?)(\d+)(\.\w+)$', basename)
		if m:
			prefix, numstr, suffix = m.group(1), m.group(2), m.group(3)
			num_start = int(numstr)
			pad = len(numstr)
			for i in range(nFramesToDo):
				seq_fn = os.path.join(dirname, prefix + str(num_start + startFrameNr + i).zfill(pad) + suffix)
				if not os.path.exists(seq_fn):
					print(f"Warning: file {seq_fn} not found, stopping at {i} frames")
					break
				img = getImage(seq_fn, 0, frame_idx=0)
				if dataMax is None:
					dataMax = img
				else:
					np.maximum(dataMax, img, out=dataMax)
		else:
			# Fallback: single file, single frame
			dataMax = getImage(fn, Header, frame_idx=0)
	else:
		# HDF5 batch fast-path: read all frames at once
		if ext in ['.h5', '.hdf', '.hdf5', '.nxs']:
			try:
				with h5py.File(fn, 'r') as f:
					dset_path = hdf5PathVar.get()
					if dset_path in f and f[dset_path].ndim == 3:
						end_idx = min(startFrameNr + nFramesToDo, f[dset_path].shape[0])
						print(f"  HDF5 batch max: frames {startFrameNr}..{end_idx-1}")
						slab = f[dset_path][startFrameNr:end_idx, :, :]
						dataMax = np.max(slab, axis=0).astype(float)
						# Apply flips/transpose/mask once
						if transpose.get() == 1:
							dataMax = np.transpose(dataMax)
						flip_h = hflip.get() == 1
						flip_v = vflip.get() == 1
						if flip_h and flip_v:
							dataMax = dataMax[::-1, ::-1].copy()
						elif flip_h:
							dataMax = dataMax[::-1, :].copy()
						elif flip_v:
							dataMax = dataMax[:, ::-1].copy()
						if applyMaskVar.get() and maskFNVar.get():
							readMask()
							if badPixelMask is not None and badPixelMask.shape == dataMax.shape:
								dataMax[badPixelMask == 1] = 0
						t2 = time.time()
						print("Time taken to calculate max: " + str(t2-t1))
						return dataMax
			except Exception as e:
				print(f"HDF5 batch max failed ({e}), falling back to per-frame")
		for i in range(nFramesToDo):
			frame_idx = startFrameNr + i
			bytesToSkip = Header + frame_idx*(BytesPerPixel*NrPixelsY*NrPixelsZ)
			img = getImage(fn, bytesToSkip, frame_idx=frame_idx)
			if dataMax is None:
				dataMax = img
			else:
				np.maximum(dataMax, img, out=dataMax)
			
	t2 = time.time()
	print("Time taken to calculate max: " + str(t2-t1))
	return dataMax

def getImageSum(fn):
	print("Calculating Sum for file: " + fn)
	global Header, BytesPerPixel
	Header = HeaderVar.get()
	BytesPerPixel = BytesVar.get()
	nFramesToDo = nFramesMaxVar.get()
	startFrameNr = maxStartFrameNrVar.get()
	
	t1 = time.time()
	dataSum = None
	ext = os.path.splitext(fn)[1].lower()
	
	is_per_file = False
	if ext in ['.tif', '.tiff'] and tifffile:
		try:
			with tifffile.TiffFile(fn) as tif:
				if len(tif.pages) == 1:
					is_per_file = True
		except Exception:
			pass

	if is_per_file:
		import re
		basename = os.path.basename(fn)
		dirname = os.path.dirname(fn)
		m = re.match(r'^(.*?)(\d+)(\.\w+)$', basename)
		if m:
			prefix, numstr, suffix = m.group(1), m.group(2), m.group(3)
			num_start = int(numstr)
			pad = len(numstr)
			for i in range(nFramesToDo):
				seq_fn = os.path.join(dirname, prefix + str(num_start + startFrameNr + i).zfill(pad) + suffix)
				if not os.path.exists(seq_fn):
					print(f"Warning: file {seq_fn} not found, stopping at {i} frames")
					break
				img = getImage(seq_fn, 0, frame_idx=0)
				if dataSum is None:
					dataSum = img
				else:
					dataSum += img
		else:
			dataSum = getImage(fn, Header, frame_idx=0)
	else:
		# HDF5 batch fast-path: read all frames at once
		if ext in ['.h5', '.hdf', '.hdf5', '.nxs']:
			try:
				with h5py.File(fn, 'r') as f:
					dset_path = hdf5PathVar.get()
					if dset_path in f and f[dset_path].ndim == 3:
						end_idx = min(startFrameNr + nFramesToDo, f[dset_path].shape[0])
						print(f"  HDF5 batch sum: frames {startFrameNr}..{end_idx-1}")
						slab = f[dset_path][startFrameNr:end_idx, :, :]
						dataSum = np.sum(slab, axis=0).astype(float)
						if transpose.get() == 1:
							dataSum = np.transpose(dataSum)
						flip_h = hflip.get() == 1
						flip_v = vflip.get() == 1
						if flip_h and flip_v:
							dataSum = dataSum[::-1, ::-1].copy()
						elif flip_h:
							dataSum = dataSum[::-1, :].copy()
						elif flip_v:
							dataSum = dataSum[:, ::-1].copy()
						if applyMaskVar.get() and maskFNVar.get():
							readMask()
							if badPixelMask is not None and badPixelMask.shape == dataSum.shape:
								dataSum[badPixelMask == 1] = 0
						t2 = time.time()
						print("Time taken to calculate sum: " + str(t2-t1))
						return dataSum
			except Exception as e:
				print(f"HDF5 batch sum failed ({e}), falling back to per-frame")
		for i in range(nFramesToDo):
			frame_idx = startFrameNr + i
			bytesToSkip = Header + frame_idx*(BytesPerPixel*NrPixelsY*NrPixelsZ)
			img = getImage(fn, bytesToSkip, frame_idx=frame_idx)
			if dataSum is None:
				dataSum = img
			else:
				dataSum += img
			
	t2 = time.time()
	print("Time taken to calculate sum: " + str(t2-t1))
	return dataSum


def getDataB(geNum,bytesToSkip, frame_idx=0):
	fn = getfn(fileStem,fileNumber,geNum)
	global getMax
	getMax = getMaxVar.get()
	getSum = getSumVar.get()
	
	if getMax:
		data = getImageMax(fn)
	elif getSum:
		data = getImageSum(fn)
	else:
		data = getImage(fn,bytesToSkip, frame_idx=frame_idx)
		
	doDark = var.get()
	if doDark == 1:
		darkfn = getfn(darkStem,darkNum,geNum)
		if nDetectors > 1:
			if dark[geNum-startDetNr] is None:
				dark[geNum-startDetNr] = getImage(darkfn,Header, is_dark=True)
			thisdark = dark[geNum-startDetNr]
		else:
			# Cache dark for single-detector
			if darkfn not in _dark_cache:
				_dark_cache[darkfn] = getImage(darkfn,Header, is_dark=True)
			thisdark = _dark_cache[darkfn]
		corrected = np.subtract(data,thisdark)
	else:
		corrected = data
	return corrected

def transforms(idx):
	txr = tx[idx]*deg2rad
	tyr = ty[idx]*deg2rad
	tzr = tz[idx]*deg2rad
	Rx = np.array([[1,0,0],[0,cos(txr),-sin(txr)],[0,sin(txr),cos(txr)]])
	Ry = np.array([[cos(tyr),0,sin(tyr)],[0,1,0],[-sin(tyr),0,cos(tyr)]])
	Rz = np.array([[cos(tzr),-sin(tzr),0],[sin(tzr),cos(tzr),0],[0,0,1]])
	return np.dot(Rx,np.dot(Ry,Rz))

def bcoord():
	numrows, numcols = bdata.shape
	def format_coord(x, y):
		col = int(x+0.5)
		row = int(y+0.5)
		bcx = float(bclocalvar1.get())
		bcy = float(bclocalvar2.get())
		[eta, rr] = CalcEtaAngleRad(-x+bcx,y-bcy)
		if col>=0 and col<numcols and row>=0 and row<numrows:
			z = bdata[row,col]
			return 'x=%1.4f, y=%1.4f, Intensity=%1.4f, RingRad(pixels)=%1.4f, Eta(degrees)=%1.4f'%(x,y,z,rr,eta)
		else:
			return 'x=%1.4f, y=%1.4f, RingRad(pixels)=%1.4f, Eta(degrees)=%1.4f'%(x,y,rr,eta)
	b.format_coord = format_coord



def plotRingsOffset():
	global lines2
	global lsdlocal, bclocal
	global DisplRingInfo, refreshPlot
	lsdlocal = float(lsdlocalvar.get())
	bclocal[0] = float(bclocalvar1.get())
	bclocal[1] = float(bclocalvar2.get())
	Etas = np.linspace(-180,180,num=360)
	lines2 = []
	ring_colors = get_ring_colors(len(ringRads))
	txtDisplay = 'Selected Rings (Increasing radius): '
	if bdata is not None:
		lims = [b.get_xlim(), b.get_ylim()]
	for idx, ringrad in enumerate(ringRads):
		Y = []
		Z = []
		txtDisplay += 'HKL:['
		for i in range(3):
			txtDisplay += str(hkls[idx][i]) + ','
		txtDisplay += '],RingNr:' + str(RingsToShow[idx]) + ',Rad[px]:' + str(int(ringrad/px)) + 'Color:' + ring_colors[idx] + ', '
		for eta in Etas:
			ringrad2 = ringrad * (lsdlocal / lsdorig)
			tmp = YZ4mREta(ringrad2,eta)
			Y.append(tmp[0]/px + bclocal[0])
			Z.append(tmp[1]/px + bclocal[1])
		if bdata is not None:
			lines2.append(b.plot(Y,Z,color=ring_colors[idx]))
	if bdata is not None and refreshPlot != 1:
		b.set_xlim([lims[0][0],lims[0][1]])
		b.set_ylim([lims[1][0],lims[1][1]])
	txtDisplay = txtDisplay[:-2]
	maxl = 270
	if len(txtDisplay) >maxl:
		tmpdisplay = ''
		nseps = int(len(txtDisplay)/maxl + 1)
		for i in range(nseps):
			tmpdisplay += txtDisplay[i*maxl:(i+1)*maxl] + '\n'
		txtDisplay = tmpdisplay[:-1]
	DisplRingInfo = Tk.Label(master=root,text=txtDisplay,justify=Tk.LEFT)
	DisplRingInfo.pack(side=Tk.TOP, fill=Tk.X)
	if bdata is not None and refreshPlot != 1:
		bcoord()
	if refreshPlot != 1:
		canvas.draw_idle()

# Removed plotRings (Big Detector) logic

def doRings():
	global lines2
	global DisplRingInfo
	plotYesNo = plotRingsVar.get()
	if lines2 is not None:
		for line2 in lines2:
			line2.pop(0).remove()
		lines2 = None
	if DisplRingInfo is not None:
		DisplRingInfo.grid_forget()
		DisplRingInfo = None
	if plotYesNo == 1:
		if ringRads is None:
			ringSelection()
		else:
			# Only plot on Single Detector
			plotRingsOffset()
	else:
		canvas.draw_idle()

def clickRings():
	global refreshPlot
	refreshPlot = 0
	doRings()

def incr_plotupdater():
	global frameNr
	global framenrvar
	frameNr = int(framenrvar.get())
	frameNr += 1
	framenrvar.set(str(frameNr))
	if getMaxVar.get():
		return
	loadbplot()

def decr_plotupdater():
	global frameNr
	global framenrvar
	frameNr = int(framenrvar.get())
	frameNr -= 1
	framenrvar.set(str(frameNr))
	if getMaxVar.get():
		return
	loadbplot()

def readParams():
	global paramFN
	paramFN = paramfilevar.get()
	global folder, fileStem, padding, startDetNr, endDetNr, bigFN
	global wedge, lsd, px, bcs, tx, wl, bigdetsize, nFramesPerFile
	global firstFileNumber, darkStem, darkNum, omegaStep, nFilesPerLayer
	global omegaStart, NrPixelsY, NrPixelsZ, threshold, RingsToShow, nDetectors
	global RhoDs, LatC, sg, maxRad, border, ringslines, lsdline, hkls
	global ty, tz, p0, p1, p2, fileNumber, dark, ringRads, ringNrs, lsdlocal
	global bclocal, lsdlocalvar, WidthTTh, tolTilts, tolBC, tolLsd, tolP
	paramContents = open(paramFN,'r').readlines()
	lsd = []
	bcs = []
	tx = []
	ty = []
	tz = []
	p0 = []
	p1 = []
	p2 = []
	RingsToShow = []
	threshold = 0
	RhoDs = []
	ringslines = []
	hkls = []
	lsdline = None
	for line in paramContents:
		if line == '\n':
			continue
		if line[0] == '#':
			continue
		if 'RingThresh' == line.split()[0]:
			ringslines.append(line)
			RingsToShow.append(int(line.split()[1]))
			threshold = max(threshold,float(line.split()[2]))
		if 'tolTilts' == line.split()[0]:
			tolTilts = line.split()[1]
		if 'tolBC' == line.split()[0]:
			tolBC = line.split()[1]
		if 'tolLsd' == line.split()[0]:
			tolLsd = line.split()[1]
		if 'tolP' == line.split()[0]:
			tolP = line.split()[1]
		if 'RawFolder' == line.split()[0]:
			folder = line.split()[1]
		if 'FileStem' == line.split()[0]:
			fileStem = line.split()[1]
		if 'Padding' == line.split()[0]:
			padding = int(line.split()[1])
		if 'StartDetNr' == line.split()[0]:
			startDetNr = int(line.split()[1])
		if 'EndDetNr' == line.split()[0]:
			endDetNr = int(line.split()[1])
		if 'Wedge' == line.split()[0]:
			wedge = float(line.split()[1])
		if 'px' == line.split()[0]:
			px = float(line.split()[1])
		if 'Wavelength' == line.split()[0]:
			wl = float(line.split()[1])
		if 'BigDetSize' == line.split()[0]:
			bigdetsize = int(line.split()[1])
		if 'nFramesPerFile' == line.split()[0]:
			nFramesPerFile = int(line.split()[1])
		if 'FirstFileNumber' == line.split()[0]:
			firstFileNumber = int(line.split()[1])
			fileNumber = firstFileNumber
		if 'StartFileNrFirstLayer' == line.split()[0]:
			firstFileNumber = int(line.split()[1])
			fileNumber = firstFileNumber
		if 'DarkStem' == line.split()[0]:
			darkStem = line.split()[1]
		if 'LatticeParameter' == line.split()[0]:
			LatC = line
		if 'LatticeConstant' == line.split()[0]:
			LatC = line
		if 'Lsd' == line.split()[0]:
			lsdline = line
		if 'MaxRingRad' == line.split()[0]:
			maxRad = line
		if 'BorderToExclude' == line.split()[0]:
			border = line
		if 'DarkNum' == line.split()[0]:
			darkNum = int(line.split()[1])
		if 'SpaceGroup' == line.split()[0]:
			sg = int(line.split()[1])
		if 'OmegaStep' == line.split()[0]:
			omegaStep = float(line.split()[1])
		if 'OmegaFirstFile' == line.split()[0]:
			omegaStart = float(line.split()[1])
		if 'NrFilesPerSweep' == line.split()[0]:
			nFilesPerLayer = int(line.split()[1])
		if 'NrPixelsY' == line.split()[0]:
			NrPixelsY = int(line.split()[1])
		if 'NrPixelsZ' == line.split()[0]:
			NrPixelsZ = int(line.split()[1])
		if 'NumDetectors' == line.split()[0]:
			nDetectors = int(line.split()[1])
		if 'Lsd' == line.split()[0]:
			lsd.append(float(line.split()[1]))
		if 'tx' == line.split()[0]:
			tx.append(float(line.split()[1]))
		if 'Width' == line.split()[0]:
			WidthTTh = line.split()[1]
		if 'DetParams' == line.split()[0]:
			lsd.append(float(line.split()[1]))
			bcs.append([float(line.split()[2]),float(line.split()[3])])
			tx.append(float(line.split()[4]))
			ty.append(float(line.split()[5]))
			tz.append(float(line.split()[6]))
			p0.append(float(line.split()[7]))
			p1.append(float(line.split()[8]))
			p2.append(float(line.split()[9]))
			RhoDs.append(float(line.split()[10]))
	if folder[0] == '~':
		folder = os.path.expanduser(folder)
	bigFN = 'BigDetectorMaskEdgeSize' + str(bigdetsize) + 'x' + str(bigdetsize) + 'Unsigned16Bit.bin'
	if midas_config and midas_config.MIDAS_BIN_DIR:
		hklGenPath = os.path.join(midas_config.MIDAS_BIN_DIR, 'GetHKLList')
	else:
		hklGenPath = os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/GetHKLList')
	
	subprocess.run([hklGenPath, paramFN], check=True)
	hklfn = 'hkls.csv'
	hklfile = open(hklfn,'r')
	hklfile.readline()
	hklinfo = hklfile.readlines()
	hklfile.close()
	ringRads = []
	ringNrs = []
	lsdlocal = lsd[0]
	lsdlocalvar.set(str(lsdlocal))
	bclocal[0] = bcs[0][0]
	bclocal[1] = bcs[0][1]
	bclocalvar1.set(str(bclocal[0]))
	bclocalvar2.set(str(bclocal[1]))
	for ringNr in RingsToShow:
		for line in hklinfo:
			if int(line.split()[4]) == ringNr:
				ringRads.append(float(line.split()[-1].split('\n')[0]))
				ringNrs.append(ringNr)
				hkls.append([int(line.split()[0]),int(line.split()[1]),int(line.split()[2])])
				break
	# initialization of dark
	dark = []
	for i in range(nDetectors):
		dark.append(None)

def writeCalibrateParams(pfname,detNum,ringsToExclude):
	f = open(pfname,'w')
	f.write('Folder '+ folder+'\n')
	f.write('FileStem ' + fileStem+'\n')
	f.write('Dark ' + getfn(darkStem,darkNum,detNum)+'\n')
	f.write('Padding '+str(padding)+'\n')
	f.write('Ext .ge'+str(detNum)+'\n')
	f.write('ImTransOpt 0\n')
	f.write('BC '+str(bcs[detNum-startDetNr][0])+' '+str(bcs[detNum-startDetNr][1])+'\n')
	f.write('px '+str(px)+'\n')
	f.write('Width '+WidthTTh+'\n')
	f.write('LatticeParameter 5.411651 5.411651 5.411651 90 90 90\nSpaceGroup 225\n')
	f.write('NrPixelsY '+str(NrPixelsY)+'\n')
	f.write('NrPixelsZ '+str(NrPixelsZ)+'\n')
	f.write('Wavelength '+str(wl)+'\n')
	f.write('Lsd '+str(lsd[detNum-startDetNr])+'\n')
	f.write('RhoD '+ str(RhoDs[detNum-startDetNr])+'\n')
	f.write('StartNr ' + str(firstFileNumber)+'\n')
	f.write('EndNr ' + str(firstFileNumber+nFilesPerLayer-1)+'\n')
	f.write('tolTilts '+tolTilts+'\ntolBC '+tolBC+'\ntolLsd '+tolLsd+'\ntolP '+tolP+'\n')
	f.write('p0 '+str(p0[detNum-startDetNr])+'\np1 '+str(p1[detNum-startDetNr])+'\np2 '+str(p2[detNum-startDetNr])+'\nEtaBinSize 5\n')
	f.write('ty '+str(ty[detNum-startDetNr])+'\ntz '+str(tz[detNum-startDetNr])+'\nWedge 0\n')
	f.write('tx '+str(tx[detNum-startDetNr])+'\n')
	if len(ringsToExclude) > 0:
		for ring in ringsToExclude:
			f.write('RingsToExclude '+str(ring)+'\n')

def writeParams():
	pfname = os.getcwd() + '/GeneratedParameters.txt'
	f = open(pfname,'w')
	f.write('NumDetectors '+str(nDetectors)+'\n')
	f.write('RawFolder '+ folder+'\n')
	f.write('FileStem ' + fileStem+'\n')
	f.write('Padding '+str(padding)+'\n')
	f.write('StartDetNr '+str(startDetNr)+'\n')
	f.write('EndDetNr '+str(endDetNr)+'\n')
	f.write('Wedge '+str(wedge)+'\n')
	sep = ' '
	for i in range(nDetectors):
		strout = 'DetParams '+ str(lsd[i]) + sep + str(bcs[i][0]) + sep + str(bcs[i][1]) + sep + str(tx[i]) + sep + str(ty[i]) + sep + str(tz[i]) + sep + str(p0[i]) + sep + str(p1[i]) + sep + str(p2[i]) + sep + str(RhoDs[i]) + '\n'
		f.write(strout)
	f.write(LatC)
	if lsdline is not None:
		f.write(lsdline)
	f.write('SpaceGroup '+str(sg)+'\n')
	f.write('Wavelength '+str(wl)+'\n')
	f.write(maxRad)
	f.write('NrPixelsY '+str(NrPixelsY)+'\n')
	f.write('NrPixelsZ '+str(NrPixelsZ)+'\n')
	f.write('BigDetSize '+str(bigdetsize)+'\n')
	f.write(border)
	f.write('px '+str(px)+'\n')
	f.write('nFramesPerFile '+str(nFramesPerFile)+'\n')
	f.write('FirstFileNumber '+str(firstFileNumber)+'\n')
	f.write('DarkStem '+darkStem+'\n')
	f.write('DarkNum '+str(darkNum)+'\n')
	f.write('OmegaStep '+str(omegaStep)+'\n')
	f.write('OmegaFirstFile '+str(omegaStart)+'\n')
	f.write('NrFilesPerSweep '+str(nFilesPerLayer)+'\n')
	for line in ringslines:
		f.write(line)
	topWrite = Tk.Toplevel()
	Tk.Label(topWrite,text='File written to '+pfname).grid(row=1)
	Tk.Button(master=topWrite,text="Close",command=topWrite.destroy).grid(row=2)

def redoCalibration():
	global topCalibrate
	topCalibrate.destroy()
	askRingsToExclude()

def parseOutputs(outputs):
	global topCalibrate
	global ty, tz, p0, p1, p2, meanStrain, stdStrain
	ty = []
	tz = []
	p0 = []
	p1 = []
	p2 = []
	meanStrain = []
	stdStrain = []
	for i in range(nDetectors):
		lsdtemp = 0
		ybctemp = 0
		zbctemp = 0
		tytemp = 0
		tztemp = 0
		p0temp = 0
		p1temp = 0
		p2temp = 0
		meanstrtemp = 0
		stdstrtemp = 0
		output = outputs[i]
		fileWrite = open('DetectorCalibrationOutputDetNr'+str(i)+'.txt','w')
		for line in output:
			fileWrite.write(line+'\n')
			if 'LsdFit' in line:
				lsdtemp += float(line.split('\t')[-1])/nFilesPerLayer
			if 'YBCFit' in line:
				ybctemp += float(line.split('\t')[-1])/nFilesPerLayer
			if 'ZBCFit' in line:
				zbctemp += float(line.split('\t')[-1])/nFilesPerLayer
			if 'tyFit' in line:
				tytemp += float(line.split('\t')[-1])/nFilesPerLayer
			if 'tzFit' in line:
				tztemp += float(line.split('\t')[-1])/nFilesPerLayer
			if 'P0Fit' in line:
				p0temp += float(line.split('\t')[-1])/nFilesPerLayer
			if 'P1Fit' in line:
				p1temp += float(line.split('\t')[-1])/nFilesPerLayer
			if 'P2Fit' in line:
				p2temp += float(line.split('\t')[-1])/nFilesPerLayer
			if 'MeanStrain' in line:
				meanstrtemp += float(line.split('\t')[-1])/nFilesPerLayer
			if 'StdStrain' in line:
				stdstrtemp += float(line.split('\t')[-1])/nFilesPerLayer
		fileWrite.close()
		lsd[i] = lsdtemp
		bcs[i][0] = ybctemp
		bcs[i][1] = zbctemp
		ty.append(tytemp)
		tz.append(tztemp)
		p0.append(p0temp)
		p1.append(p1temp)
		p2.append(p2temp)
		meanStrain.append(meanstrtemp)
		stdStrain.append(stdstrtemp)
	# Display new values on screen, ask whether to run again?
	Tk.Label(topCalibrate,text="The refined values are:").grid(row=1)
	for i in range(nDetectors):
		strOut="For detector %d, Lsd: %lf, YBC: %lf, ZBC: %lf, ty: %lf, tz: %lf, Ps: %lf %lf %lf, MeanStrain: %lf, StdStrain: %lf"%(startDetNr+i,
			lsd[i],bcs[i][0],bcs[i][1],ty[i],tz[i],p0[i],p1[i],p2[i],meanStrain[i],stdStrain[i])
		Tk.Label(topCalibrate,text=strOut).grid(row=2+i)
	Tk.Label(topCalibrate,text="Do you want to run calibration again with these parameters?").grid(row=nDetectors+2)
	Tk.Button(master=topCalibrate,text='Yes',command=redoCalibration).grid(row=nDetectors+3)
	Tk.Button(master=topCalibrate,text='No',command=topCalibrate.destroy).grid(row=nDetectors+4)

def calibrateDetector():
	global ringsexcludevar
	global eringsexclude, buttonConfirmRingsExclude, ringsexcludelabel
	eringsexclude.grid_forget()
	buttonConfirmRingsExclude.grid_forget()
	ringsexcludelabel.grid_forget()
	ringsexcludestr = ringsexcludevar.get()
	if midas_config and midas_config.MIDAS_BIN_DIR:
		calibratecmd = os.path.join(midas_config.MIDAS_BIN_DIR, 'Calibrant')
	else:
		calibratecmd = os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/Calibrant')
	
	if ringsexcludestr == '0':
		ringsToExclude = []
	else:
		ringsToExclude = [int(rings) for rings in ringsexcludestr.split(',')]
	
	pfnames = []
	for i in range(startDetNr,endDetNr+1):
		pfnames.append('CalibrationDetNr' + str(i) + '.txt')
		writeCalibrateParams(pfnames[-1],i,ringsToExclude)
	
	# Safe Popen usage with list arguments
	cmds = [[calibratecmd, pfname] for pfname in pfnames]
	processes = [subprocess.Popen(cmd,
				stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True) for cmd in cmds]
	
	def get_lines(process):
		return process.communicate()[0].decode('utf-8').splitlines()
	
	outputs = Pool(len(processes)).map(get_lines,processes)
	parseOutputs(outputs)

def askRingsToExclude():
	global topCalibrate
	global ringsexcludevar
	global eringsexclude, buttonConfirmRingsExclude, ringsexcludelabel
	topCalibrate = Tk.Toplevel()
	topCalibrate.title("Select rings to exclude from Calibration")
	ringsexcludelabel = Tk.Label(master=topCalibrate,text="Please enter the rings you would like to exclude, each separated by a comma, no spaces please")
	ringsexcludelabel.grid(row=1)
	ringsexcludevar = Tk.StringVar()
	ringsexcludevar.set(str(0))
	eringsexclude = Tk.Entry(topCalibrate,textvariable=ringsexcludevar)
	eringsexclude.grid(row=2)
	buttonConfirmRingsExclude = Tk.Button(master=topCalibrate,text="Confirm",command=calibrateDetector)
	buttonConfirmRingsExclude.grid(row=3)

def paramfileselect():
	global paramFN
	global paramfilevar
	paramFN = tkFileDialog.askopenfilename()
	paramfilevar.set(paramFN)



def loadbplot():
	global bclocal
	global lsdlocal
	global lsdorig
	global initplot2
	global origdetnum
	global bclocalvar1, bclocalvar2
	global ax
	global fileNumber, refreshPlot
	global bdata
	global lines2, NrPixelsY, NrPixelsZ
	global firstFileNumber, nFramesPerFile
	global Header, BytesPerPixel
	Header = HeaderVar.get()
	BytesPerPixel = BytesVar.get()
	if not initplot2:
		lims = [b.get_xlim(), b.get_ylim()]
	frameNr = int(framenrvar.get())
	threshold = float(thresholdvar.get())
	upperthreshold = float(maxthresholdvar.get())
	NrPixelsY = int(NrPixelsYVar.get())
	NrPixelsZ = int(NrPixelsZVar.get())
	firstFileNumber = int(firstFileNrVar.get())
	nFramesPerFile = int(nFramesPerFileVar.get())
	fileNumber = int(firstFileNumber + frameNr/nFramesPerFile)
	framesToSkip = int(frameNr % nFramesPerFile)
	bytesToSkip = Header + framesToSkip*(BytesPerPixel*NrPixelsY*NrPixelsZ)
	detnr = int(detnumbvar.get())
	if detnr != -1:
		if (detnr != origdetnum or initplot2) and bcs is not None:
			origdetnum = detnr
			bclocal[0] = bcs[detnr-startDetNr][0]
			bclocal[1] = bcs[detnr-startDetNr][1]
			bclocalvar1.set(str(bclocal[0]))
			bclocalvar2.set(str(bclocal[1]))
		else:
			bclocal[0] = float(bclocalvar1.get())
			bclocal[1] = float(bclocalvar2.get())
	else:
		bclocal[0] = float(bclocalvar1.get())
		bclocal[1] = float(bclocalvar2.get())
	bdata = getDataB(detnr,bytesToSkip, frame_idx=framesToSkip)
	if nDetectors > 1:
		lsdorig = lsd[detnr-startDetNr]
	lsdlocal = float(lsdlocalvar.get())
	refreshPlot = 1
	doRings()
	global _b_artist, _b_logmode
	use_log = dolog.get() != 0
	if use_log:
		if threshold == 0:
			threshold = 1
		if upperthreshold == 0:
			upperthreshold = 1
		mask3 = np.copy(bdata)
		mask3 [ mask3 == 0 ] = 1
		display_data = np.log(mask3)
		clim = (np.log(threshold),np.log(upperthreshold))
	else:
		display_data = bdata
		clim = (threshold,upperthreshold)
	can_reuse = (not initplot2 and _b_artist is not None and
	             _b_logmode == use_log and
	             _b_artist.get_array().shape == display_data.shape)
	if can_reuse:
		_b_artist.set_data(display_data)
		_b_artist.set_clim(*clim)
		b.set_xlim([lims[0][0],lims[0][1]])
		b.set_ylim([lims[1][0],lims[1][1]])
	else:
		b.clear()
		cmap = plt.get_cmap('bone')
		if colorMinVar.get():
			cmap.set_under('blue')
		if colorMaxVar.get():
			cmap.set_over('red')
			
		_b_artist = b.imshow(display_data,cmap=cmap,interpolation='nearest',clim=clim)
		_b_logmode = use_log
		if initplot2:
			initplot2 = 0
			b.invert_yaxis()
		else:
			b.set_xlim([lims[0][0],lims[0][1]])
			b.set_ylim([lims[1][0],lims[1][1]])
	bcoord()
	b.title.set_text("Single Detector Display")
	canvas.draw_idle()

def acceptRings():
	global RingsToShow
	global ringRads
	global topSelectRings
	global hkls
	global plotRingsVar
	items = ListBox1.curselection()
	ringRads = [RingRad[int(item)] for item in items]
	RingsToShow = [int(item)+1 for item in items]
	hkls = [hkl[int(item)] for item in items]
	topSelectRings.destroy()
	plotRingsVar.set(1)
	doRings()



def selectRings():
	global topSelectRings
	global hklLines, hkl, ds, Ttheta, RingRad, ListBox1
	
	hklinfo = []
	header = ""
	
	# Use TemporaryDirectory to contain output files safely
	with tempfile.TemporaryDirectory() as temp_dir:
		if midas_config and midas_config.MIDAS_BIN_DIR:
			hklGenPath = os.path.join(midas_config.MIDAS_BIN_DIR, 'GetHKLList')
		else:
			hklGenPath = os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/GetHKLList')
		
		pfname = os.path.join(temp_dir, 'ps_midas_ff.txt')
		with open(pfname, 'w') as f:
			f.write('Wavelength ' + str(wl) + '\n')
			f.write('SpaceGroup ' + str(sg) + '\n')
			f.write('Lsd ' + str(tempLsd) + '\n')
			f.write('MaxRingRad ' + str(tempMaxRingRad) + '\n')
			f.write('LatticeConstant ')
			for i in range(6):
				f.write(str(LatticeConstant[i]) + ' ')
			f.write('\n')
		
		# Run GetHKLList in temp_dir
		subprocess.run([hklGenPath, pfname], check=True, cwd=temp_dir)
		
		hklfn = os.path.join(temp_dir, 'hkls.csv')
		if os.path.exists(hklfn):
			with open(hklfn, 'r') as hklfile:
				header = hklfile.readline()
				header = header.replace(' ','      ')
				hklinfo = hklfile.readlines()
		else:
			print("Error: hkls.csv not found.")
			return
	maxRingNr = 101
	hkl = []
	ds = []
	Ttheta = []
	RingRad = []
	hklLines = []
	for ringNr in range(1,maxRingNr):
		for line in hklinfo:
			if int(line.split()[4]) == ringNr:
				hkl.append([int(line.split()[0]),int(line.split()[1]),int(line.split()[2])])
				ds.append(float(line.split()[3]))
				Ttheta.append(float(line.split()[9]))
				RingRad.append(float(line.split()[10].split('\n')[0]))
				hklLines.append(line.split('\n')[0])
				break
	topSelectRings = Tk.Toplevel()
	topSelectRings.title('Select Rings')
	nrhkls = len(hklLines)
	Tk.Label(master=topSelectRings,text=header.split('\n')[0]).grid(row=0,column=0,sticky=Tk.W,columnspan=2)
	ListBox1 = Tk.Listbox(topSelectRings,width=80,height=15,selectmode=Tk.EXTENDED)
	ListBox1.grid(row=1,column=0)
	yscroll=Tk.Scrollbar(topSelectRings)
	yscroll.grid(row=1,column=1,sticky=Tk.N+Tk.S)
	for line in hklLines:
		ListBox1.insert(Tk.END,line)
	ListBox1.config(yscrollcommand=yscroll.set)
	yscroll.config(command=ListBox1.yview)
	Tk.Button(master=topSelectRings,text='Done',command=acceptRings).grid(row=2,column=0,columnspan=2)

def acceptSgWlLatC():
	global wl, sg, LatticeConstant, tempLsd, tempMaxRingRad, px, bigdetsize
	global topRingMaterialSelection, lsdlocal, lsdorig
	wl = float(wlVar.get())
	if wl > 1:
		wl = 12.398/wl
	sg = int(sgVar.get())
	px = float(pxVar.get())
	tempLsd = float(tempLsdVar.get())
	for i in range(4):
		if lsd[i] == 0:
			lsd[i] = tempLsd
	lsdlocal = tempLsd
	lsdorig = tempLsd
	lsdlocalvar.set(str(tempLsd))
	tempMaxRingRad = float(tempMaxRingRadVar.get())
	for i in range(6):
		LatticeConstant[i] = float(LatticeConstantVar[i].get())
	topRingMaterialSelection.destroy()
	selectRings()

def ringSelection():
	global wlVar, sgVar, LatticeConstantVar, tempLsdVar, tempMaxRingRadVar, pxVar
	global topRingMaterialSelection, refreshPlot
	wlVar = Tk.StringVar()
	sgVar = Tk.StringVar()
	pxVar = Tk.StringVar()
	tempLsdVar = Tk.StringVar()
	tempMaxRingRadVar = Tk.StringVar()
	sgVar.set(str(sg))
	wlVar.set(str(wl))
	pxVar.set(str(px))
	tempLsdVar.set(lsdlocalvar.get())
	tempMaxRingRadVar.set(str(tempMaxRingRad))
	LatticeConstantVar = [Tk.StringVar(),Tk.StringVar(),Tk.StringVar(),Tk.StringVar(),Tk.StringVar(),Tk.StringVar()]
	for i in range(6):
		LatticeConstantVar[i].set(str(LatticeConstant[i]))
	topRingMaterialSelection = Tk.Toplevel()
	topRingMaterialSelection.title('Select the SpaceGroup, Wavelength(or Energy), Lattice Constant')
	Tk.Label(master=topRingMaterialSelection,text='Please enter the SpaceGroup, Wavelength(or Energy), Lattice Constant, Sample To Detector Distance(Lsd)').grid(row=1,column=1,columnspan=7)
	Tk.Label(master=topRingMaterialSelection,text='SpaceGroup').grid(row=2,column=1,sticky=Tk.W)
	Tk.Entry(master=topRingMaterialSelection,textvariable=sgVar,width=4).grid(row=2,column=2,sticky=Tk.W)
	Tk.Label(master=topRingMaterialSelection,text='Wavelength (A) or Energy (KeV)').grid(row=3,column=1,sticky=Tk.W)
	Tk.Entry(master=topRingMaterialSelection,textvariable=wlVar,width=8).grid(row=3,column=2,sticky=Tk.W)
	Tk.Label(master=topRingMaterialSelection,text='LatticeConstant (A)').grid(row=4,column=1,sticky=Tk.W)
	for i in range(6):
		Tk.Entry(master=topRingMaterialSelection,textvariable=LatticeConstantVar[i],width=8).grid(row=4,column=i+2,sticky=Tk.W)
	Tk.Label(master=topRingMaterialSelection,text='Lsd (um)').grid(row=5,column=1,sticky=Tk.W)
	Tk.Entry(master=topRingMaterialSelection,textvariable=tempLsdVar,width=8).grid(row=5,column=2,sticky=Tk.W)
	Tk.Label(master=topRingMaterialSelection,text='MaxRingRad (um)').grid(row=6,column=1,sticky=Tk.W)
	Tk.Entry(master=topRingMaterialSelection,textvariable=tempMaxRingRadVar,width=8).grid(row=6,column=2,sticky=Tk.W)
	Tk.Label(master=topRingMaterialSelection,text='Pixel Size (um)').grid(row=7,column=1,sticky=Tk.W)
	Tk.Entry(master=topRingMaterialSelection,textvariable=pxVar,width=8).grid(row=7,column=2,sticky=Tk.W)
	Tk.Button(master=topRingMaterialSelection,text='Continue',command=acceptSgWlLatC).grid(row=8,column=1,columnspan=7)
	refreshPlot = 0

def selectFile():
	return tkFileDialog.askopenfilename()

def selectHDF5Path(is_dark=False):
	# Show cached datasets from the already-opened HDF5 file
	global hdf5_cached_datasets
	if not hdf5_cached_datasets:
		print("No HDF5 datasets cached. Open a file with FirstFile first.")
		return
	
	topH5 = Tk.Toplevel()
	topH5.title("Select HDF5 Dataset Path")
	
	listbox = Tk.Listbox(topH5, width=60, height=20, font=("Helvetica", 14))
	listbox.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=True)
	
	scrollbar = Tk.Scrollbar(topH5, orient="vertical")
	scrollbar.pack(side="right", fill="y")
	
	listbox.config(yscrollcommand=scrollbar.set)
	scrollbar.config(command=listbox.yview)
	
	for p in hdf5_cached_datasets:
		listbox.insert(Tk.END, p)
		
	def on_select():
		selection = listbox.curselection()
		if selection:
			idx = selection[0]
			path = listbox.get(idx)
			if is_dark:
				hdf5DarkPathVar.set(path)
			else:
				hdf5PathVar.set(path)
			topH5.destroy()
			
	Tk.Button(topH5, text="Select", command=on_select, font=("Helvetica", 14)).pack()

def firstFileSelector():
	global fileStem, folder, padding,firstFileNumber,nFramesPerFile
	global nDetectors, detnumbvar,nFramesMaxVar,nFramesPerFileVar
	global NrPixelsY,NrPixelsZ
	global Header, BytesPerPixel
	global fnextvar
	Header = HeaderVar.get()
	BytesPerPixel = BytesVar.get()
	NrPixelsY = int(NrPixelsYVar.get())
	NrPixelsZ = int(NrPixelsZVar.get())
	firstfilefullpath = selectFile()
	if not firstfilefullpath: return
	
	# Handling compression for detection
	check_fn = firstfilefullpath
	if firstfilefullpath.endswith('.bz2'):
		check_fn = firstfilefullpath[:-4]
		
	# Check format
	ext = os.path.splitext(check_fn)[1].lower()
	
	basename_full = os.path.basename(check_fn)
	dot_idx = basename_full.find('.')
	if dot_idx != -1:
		fullfilename = basename_full[:dot_idx]
		full_ext = basename_full[dot_idx+1:]  # e.g., 'vrx.h5' or 'h5'
	else:
		fullfilename = basename_full
		full_ext = ''
	
	# If HDF5, structure might be different
	if ext in ['.h5', '.hdf', '.hdf5', '.nxs']:
		fileStem = fullfilename
		firstFileNumber = 0 # Default for single HDF5 file? Or numbered series?
		# If series: stem_001.h5
		parts = fullfilename.split('_')
		if parts[-1].isdigit():
			firstFileNumber = int(parts[-1])
			fileStem = '_'.join(parts[:-1])
			padding = len(parts[-1])
		else:
			padding = 0
		
		# Propose dataset path if not set
		if not hdf5PathVar.get():
			hdf5PathVar.set('/exchange/data')
			
		nDetectors = 1
		detnumbvar.set('-1')
		fnextvar.set(full_ext)
		folder = os.path.dirname(firstfilefullpath) + '/'
		firstFileNrVar.set(str(firstFileNumber))
		framenrvar.set('0')
		
		# Get dimensions and cache all datasets from HDF5
		global hdf5_cached_datasets
		try:
			with h5py.File(firstfilefullpath, 'r') as f:
				# Cache all dataset paths
				hdf5_cached_datasets = []
				def visit_func(name, node):
					if isinstance(node, h5py.Dataset):
						hdf5_cached_datasets.append('/' + name)
				f.visititems(visit_func)
				print(f"Found {len(hdf5_cached_datasets)} datasets: {hdf5_cached_datasets}")
				
				# Auto-detect data and dark paths
				data_path = None
				dark_path = None
				for p in hdf5_cached_datasets:
					pl = p.lower()
					if 'dark' in pl and dark_path is None:
						dark_path = p
					elif ('data' in pl or 'image' in pl) and data_path is None:
						data_path = p
				
				# If only one dataset, use it for data
				if data_path is None and len(hdf5_cached_datasets) == 1:
					data_path = hdf5_cached_datasets[0]
				
				if data_path:
					hdf5PathVar.set(data_path)
				if dark_path:
					hdf5DarkPathVar.set(dark_path)
				
				# Get dimensions from selected data path
				dset_path = data_path or hdf5PathVar.get()
				nFramesPerFile = 1  # default
				if dset_path in f:
					shape = f[dset_path].shape
					if len(shape) == 3:
						nFramesPerFile = shape[0]
						NrPixelsY = shape[1]
						NrPixelsZ = shape[2]
					elif len(shape) == 2:
						nFramesPerFile = 1
						NrPixelsY = shape[0]
						NrPixelsZ = shape[1]
					nFramesPerFileVar.set(nFramesPerFile)
					nFramesMaxVar.set(nFramesPerFile)
					NrPixelsYVar.set(NrPixelsY)
					NrPixelsZVar.set(NrPixelsZ)
		except Exception as e:
			print(f"Error reading HDF5 info: {e}")
			
		return

	# TIFF and Binary share same stem_NNNNNN naming convention
	# Original logic for binary/tiff
	fileStem = '_'.join(fullfilename.split('_')[:-1])
	firstFileNumber = int(fullfilename.split('_')[-1])
	firstFileNrVar.set(firstFileNumber)
	padding = len(fullfilename.split('_')[-1])
	nDetectors = 1
	framenrvar.set('0')
	# Check if extension ends with geX (detector number), else use full_ext
	if full_ext.startswith('ge') and len(full_ext) == 3 and full_ext[-1].isdigit():
		detnumbvar.set(full_ext[-1])
	else:
		detnumbvar.set('-1')
		fnextvar.set(full_ext)
	folder = os.path.dirname(firstfilefullpath) + '/'
	statinfo = os.stat(firstfilefullpath)
	# nFrames calculation for binary
	# This might fail for compressed files if we use compressed size. 
	# User should input correct params if auto-detection fails.
	if not firstfilefullpath.endswith('.bz2'):
		nFramesPerFile = int((statinfo.st_size - Header)/(BytesPerPixel*NrPixelsY*NrPixelsZ))
		nFramesPerFileVar.set(nFramesPerFile)
		nFramesMaxVar.set(nFramesPerFile)

def darkFileSelector():
	global darkStem, darkNum, dark
	darkfilefullpath = selectFile()
	if not darkfilefullpath:
		return
	darkbasename = os.path.basename(darkfilefullpath)
	dot_idx = darkbasename.find('.')
	if dot_idx != -1:
		darkfullfilename = darkbasename[:dot_idx]
		dark_ext = darkbasename[dot_idx+1:]
	else:
		darkfullfilename = darkbasename
		dark_ext = ''
	parts = darkfullfilename.split('_')
	if parts[-1].isdigit():
		darkNum = int(parts[-1])
		darkStem = '_'.join(parts[:-1])
	else:
		print(f"Warning: could not extract file number from dark file: {darkbasename}")
		darkNum = 0
		darkStem = darkfullfilename
	dark = []
	var.set(1)

def replot():
	global initplot2
	global lines2
	global lines
	global bdata, refreshPlot
	global _b_artist, _b_logmode
	use_log = dolog.get() != 0
	threshold = float(thresholdvar.get())
	upperthreshold = float(maxthresholdvar.get())

	if bdata is not None:
		if not initplot2:
			lims = [b.get_xlim(), b.get_ylim()]
		if use_log:
			if threshold == 0:
				threshold = 1
			if upperthreshold == 0:
				upperthreshold = 1
			mask3 = np.copy(bdata)
			mask3 [ mask3 == 0 ] = 1
			display_data = np.log(mask3)
			clim = (np.log(threshold),np.log(upperthreshold))
		else:
			display_data = bdata
			clim = (threshold,upperthreshold)
		can_reuse = (not initplot2 and _b_artist is not None and
		             _b_logmode == use_log and
		             _b_artist.get_array().shape == display_data.shape)
		# Always rebuild colormap to pick up Color < Min / Color > Max changes
		cmap = plt.get_cmap('bone').copy()
		if colorMinVar.get():
			cmap.set_under('blue')
		if colorMaxVar.get():
			cmap.set_over('red')
		if can_reuse:
			_b_artist.set_data(display_data)
			_b_artist.set_clim(*clim)
			_b_artist.set_cmap(cmap)
			b.set_xlim([lims[0][0],lims[0][1]])
			b.set_ylim([lims[1][0],lims[1][1]])
		else:
			b.clear()
			_b_artist = b.imshow(display_data,cmap=cmap,interpolation='nearest',clim=clim)
			_b_logmode = use_log
			if initplot2:
				initplot2 = 0
				b.invert_yaxis()
			else:
				b.set_xlim([lims[0][0],lims[0][1]])
				b.set_ylim([lims[1][0],lims[1][1]])
		bcoord()
		b.title.set_text("Single Detector Display")
		refreshPlot = 1
	doRings()
	canvas.draw_idle()

# Main function
root = Tk.Tk()
root.wm_title("FF display v1.0 Dt. 2026/02/19 hsharma@anl.gov [scanning...]")

# Start async file auto-detection
_start_auto_detect_thread(root)
figur = Figure(figsize=(10,8),dpi=100)
canvas = FigureCanvasTkAgg(figur,master=root)
a = None # Removed 'a' entirely
# Rename 'b' to 'a' or just assign 'b' to this subplot?
# Original code used 'b' for single detector. 'a' for big det.
# We are keeping Single Detector.
# Let's alias 'b' to 'a' to minimize code changes in 'loadbplot' (which uses 'b')?
# No, 'loadbplot' uses global 'b'.
# usage: b.clear(), b.imshow().
# So we should create 'b' as the ONLY subplot.
b = figur.add_subplot(111,aspect='equal')
# a = None # Remove 'a'
b.title.set_text("Single Detector Display")
figrowspan = 10
figcolspan = 10
lsd = [0,0,0,0]
lsdlocal = 1000000
frameNr = 0
fileNumber = 0
getMax = 0
paramFN = 'PS.txt'

bdata = None

bigdetsize = 2048

initplot2 = 1

_b_artist = None

_b_logmode = False
origdetnum = 1
bclocal = [1024,1024]
ringRads = None
sg = 225
bcs = None
wl = 0.172979
px = 200
NrPixelsY = 2048
NrPixelsZ = 2048
Header = 8192
BytesPerPixel = 2
tempLsd = 1000000
tempMaxRingRad = 2000000
fileStem = ''
folder = os.getcwd() + '/'
padding = 6
darkStem = ''
darkNum = 0
dark = []
nDetectors = 1
startDetNr = 1
endDetNr = 1
nFilesPerLayer = 1
nFramesPerFile = 1
firstFileNumber = 1
firstFileNrVar = Tk.StringVar()
fnextvar = Tk.StringVar()
fnextvar.set('tif')
nFramesPerFileVar = Tk.StringVar()
firstFileNrVar.set(str(firstFileNumber))
nFramesPerFileVar.set(str(nFramesPerFile))
paramfilevar = Tk.StringVar()
paramfilevar.set(paramFN)
framenrvar = Tk.StringVar()
framenrvar.set(str(frameNr))
thresholdvar = Tk.StringVar()
threshold = 0
thresholdvar.set(str(threshold))
maxthresholdvar = Tk.StringVar()
maxthresholdvar.set(str(2000))
NrPixelsYVar = Tk.StringVar()
NrPixelsYVar.set(str(2048))
NrPixelsZVar = Tk.StringVar()
NrPixelsZVar.set(str(2048))
HeaderVar = Tk.IntVar()
HeaderVar.set(8192)
BytesVar = Tk.IntVar()
BytesVar.set(2)
LatticeConstant = np.zeros(6)
LatticeConstant[0] = 5.41116
LatticeConstant[1] = 5.41116
LatticeConstant[2] = 5.41116
LatticeConstant[3] = 90
LatticeConstant[4] = 90
LatticeConstant[5] = 90
lines = None
lines2 = None
DisplRingInfo = None
plotRingsVar = Tk.IntVar()
var = Tk.IntVar()
hydraVar = Tk.IntVar()
hydraVar.set(0)
sepfolderVar = Tk.IntVar()
getMaxVar = Tk.IntVar()
maskFNVar = Tk.StringVar()
maskFNVar.set("")
badPixelMask = None
_dark_cache = {}
colorMinVar = Tk.IntVar()
colorMaxVar = Tk.IntVar()
getSumVar = Tk.IntVar()
applyMaskVar = Tk.IntVar()
detnumbvar = Tk.StringVar()
detnumbvar.set('-1')
lsdlocalvar = Tk.StringVar()
lsdlocalvar.set(str(lsdlocal))
bclocalvar1 = Tk.StringVar()
bclocalvar1.set(str(bclocal[0]))
bclocalvar2 = Tk.StringVar()
bclocalvar2.set(str(bclocal[1]))
nFramesMaxVar = Tk.IntVar()
maxStartFrameNrVar = Tk.IntVar()
nFramesMaxVar.set(240)
maxStartFrameNrVar.set(0)
dolog = Tk.IntVar()
hflip = Tk.IntVar()
vflip = Tk.IntVar()
transpose = Tk.IntVar()
refreshPlot = 0
hdf5PathVar = Tk.StringVar()
hdf5PathVar.set('/exchange/data')
hdf5DarkPathVar = Tk.StringVar()
hdf5DarkPathVar.set('/exchange/dark')
hdf5_cached_datasets = []

canvas.get_tk_widget().pack(fill=Tk.BOTH, expand=True)

# GUI Layout Redesign

# Toolbar
toolbar_frame = Tk.Frame(root)
toolbar_frame.pack(side=Tk.BOTTOM, fill=Tk.X)
toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
toolbar.update()

# Main Control Frame
mainControlFrame = Tk.Frame(root)
mainControlFrame.pack(side=Tk.BOTTOM, fill=Tk.X, padx=2, pady=2)

# Font for readability
default_font = ("Helvetica", 14)

# 1. File I/O Frame
fileFrame = Tk.LabelFrame(mainControlFrame, text="File I/O", font=default_font)
fileFrame.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

Tk.Button(fileFrame, text='FirstFile', command=firstFileSelector, font=default_font).grid(row=0, column=0)
Tk.Button(fileFrame, text='DarkFile', command=darkFileSelector, font=default_font).grid(row=0, column=1)
Tk.Checkbutton(fileFrame, text="DarkCorr", variable=var, font=default_font).grid(row=0, column=2, columnspan=2)

Tk.Label(fileFrame, text="FileNr", font=default_font).grid(row=1, column=0)
Tk.Entry(fileFrame, textvariable=firstFileNrVar, width=5, font=default_font).grid(row=1, column=1)
Tk.Label(fileFrame, text="nFr/File", font=default_font).grid(row=1, column=2)
Tk.Entry(fileFrame, textvariable=nFramesPerFileVar, width=5, font=default_font).grid(row=1, column=3)

Tk.Label(fileFrame, text="H5 Data", font=default_font).grid(row=2, column=0)
Tk.Entry(fileFrame, textvariable=hdf5PathVar, width=12, font=default_font).grid(row=2, column=1, columnspan=2)
Tk.Button(fileFrame, text="Browse", command=lambda: selectHDF5Path(is_dark=False), font=default_font).grid(row=2, column=3)

Tk.Label(fileFrame, text="H5 Dark", font=default_font).grid(row=3, column=0)
Tk.Entry(fileFrame, textvariable=hdf5DarkPathVar, width=12, font=default_font).grid(row=3, column=1, columnspan=2)
Tk.Button(fileFrame, text="Browse", command=lambda: selectHDF5Path(is_dark=True), font=default_font).grid(row=3, column=3)

Tk.Button(fileFrame, text="MaskFile", command=lambda: maskFNVar.set(tkFileDialog.askopenfilename()), font=default_font).grid(row=4, column=0)
Tk.Entry(fileFrame, textvariable=maskFNVar, width=10, font=default_font).grid(row=4, column=1, columnspan=2)
Tk.Checkbutton(fileFrame, text="ApplyMask", variable=applyMaskVar, font=default_font).grid(row=4, column=3)

# 2. Image Settings Frame
imgFrame = Tk.LabelFrame(mainControlFrame, text="Image Settings", font=default_font)
imgFrame.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)

Tk.Label(imgFrame, text='NrPixelsHor', font=default_font).grid(row=0, column=0)
Tk.Entry(imgFrame, textvariable=NrPixelsZVar, width=5, font=default_font).grid(row=0, column=1)
Tk.Label(imgFrame, text='NrPixelsVert', font=default_font).grid(row=1, column=0)
Tk.Entry(imgFrame, textvariable=NrPixelsYVar, width=5, font=default_font).grid(row=1, column=1)

Tk.Label(imgFrame, text="HeadSize", font=default_font).grid(row=2, column=0)
Tk.Entry(imgFrame, textvariable=HeaderVar, width=5, font=default_font).grid(row=2, column=1)
Tk.Label(imgFrame, text="Bytes/Px", font=default_font).grid(row=3, column=0)
Tk.Entry(imgFrame, textvariable=BytesVar, width=5, font=default_font).grid(row=3, column=1)

Tk.Checkbutton(imgFrame, text="HFlip", variable=hflip, font=default_font).grid(row=4, column=0)
Tk.Checkbutton(imgFrame, text="VFlip", variable=vflip, font=default_font).grid(row=4, column=1)
Tk.Checkbutton(imgFrame, text="Transp", variable=transpose, font=default_font).grid(row=5, column=0, columnspan=2)

# 3. Display Control Frame
dispFrame = Tk.LabelFrame(mainControlFrame, text="Display Control", font=default_font)
dispFrame.grid(row=0, column=2, sticky="nsew", padx=2, pady=2)

Tk.Label(dispFrame, text='FrameNr', font=default_font).grid(row=0, column=0)
Tk.Entry(dispFrame, textvariable=framenrvar, width=5, font=default_font).grid(row=0, column=1)
Tk.Button(dispFrame, text='+', command=incr_plotupdater, font=default_font).grid(row=0, column=2)
Tk.Button(dispFrame, text='-', command=decr_plotupdater, font=default_font).grid(row=0, column=3)

Tk.Label(dispFrame, text='MinThresh', font=default_font).grid(row=1, column=0)
Tk.Entry(dispFrame, textvariable=thresholdvar, width=5, font=default_font).grid(row=1, column=1)
Tk.Checkbutton(dispFrame, text="Color < Min", variable=colorMinVar, font=default_font).grid(row=1, column=2, columnspan=2)

Tk.Label(dispFrame, text='MaxThresh', font=default_font).grid(row=2, column=0)
Tk.Entry(dispFrame, textvariable=maxthresholdvar, width=5, font=default_font).grid(row=2, column=1)
Tk.Checkbutton(dispFrame, text="Color > Max", variable=colorMaxVar, font=default_font).grid(row=2, column=2, columnspan=2)

Tk.Checkbutton(dispFrame, text="LogScale", variable=dolog, font=default_font).grid(row=3, column=0, columnspan=2)

# 4. Processing Frame
procFrame = Tk.LabelFrame(mainControlFrame, text="Processing", font=default_font)
procFrame.grid(row=0, column=3, sticky="nsew", padx=2, pady=2)

Tk.Checkbutton(procFrame, text="MaxOverFrames", variable=getMaxVar, font=default_font).grid(row=0, column=0)
Tk.Checkbutton(procFrame, text="SumOverFrames", variable=getSumVar, font=default_font).grid(row=1, column=0)

Tk.Label(procFrame, text="nFrames", font=default_font).grid(row=0, column=1)
Tk.Entry(procFrame, textvariable=nFramesMaxVar, width=5, font=default_font).grid(row=0, column=2)

Tk.Label(procFrame, text="StartFrame", font=default_font).grid(row=1, column=1)
Tk.Entry(procFrame, textvariable=maxStartFrameNrVar, width=5, font=default_font).grid(row=1, column=2)

# Rings
Tk.Button(procFrame, text="RingsMat", command=ringSelection, font=default_font).grid(row=2, column=0)
Tk.Checkbutton(procFrame, text='PlotRings', variable=plotRingsVar, command=clickRings, font=default_font).grid(row=2, column=1, columnspan=2)

# Detector Params
Tk.Label(procFrame, text='Lsd', font=default_font).grid(row=3, column=0)
Tk.Entry(procFrame, textvariable=lsdlocalvar, width=10, font=default_font).grid(row=3, column=1, columnspan=2)
Tk.Label(procFrame, text='BC', font=default_font).grid(row=4, column=0)
Tk.Entry(procFrame, textvariable=bclocalvar1, width=5, font=default_font).grid(row=4, column=1)
Tk.Entry(procFrame, textvariable=bclocalvar2, width=5, font=default_font).grid(row=4, column=2)

# Action button bar: Quit | Update Plot | Load
actionFrame = Tk.Frame(root)
actionFrame.pack(side=Tk.BOTTOM, fill=Tk.X, pady=5)
Tk.Button(actionFrame, text='Quit', command=_quit, font=("Helvetica", 18)).pack(side=Tk.LEFT, padx=20)
Tk.Button(actionFrame, text='Update Plot', command=replot, font=("Helvetica", 18), bg='lightblue').pack(side=Tk.LEFT, expand=True)
Tk.Button(actionFrame, text='Load', command=loadbplot, font=("Helvetica", 18), bg='lightgreen').pack(side=Tk.LEFT, expand=True)


if __name__ == "__main__":
	try:
		root.bind('<Control-w>', lambda event: root.destroy())
		Tk.mainloop()
	except KeyboardInterrupt:
		root.destroy()
