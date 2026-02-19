#!/usr/bin/env python

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

###### TODO: SET CLIM DRAGBAR FOR MIC FILE

import PIL
import matplotlib
matplotlib.use('TkAgg')

# Prefer tifffile for faster TIFF loading (direct to numpy), fall back to PIL
try:
	import tifffile
	HAS_TIFFFILE = True
except ImportError:
	HAS_TIFFFILE = False
import tempfile
import subprocess
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import sys
import tkinter as Tk
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import glob
import tkinter.filedialog as tkFileDialog
from math import sin, cos

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

# Helper constants
deg2rad = 0.0174532925199433
rad2deg = 57.2957795130823

## Some initialization values
NrPixelsY = 2048
NrPixelsZ = 2048
nrfilesperdistance = 720
padding = 6
ndistances = 6
background=0
folder = os.getcwd()
# fnstem is typically current folder name / current folder name
fnstem = os.path.basename(folder) + '/' + os.path.basename(folder)
folder = os.path.dirname(folder)
figcolspan=10
figrowspan=10

# Auto-detect startframenr from files matching fnstem pattern
_beampos_mode = False
_beampos_files = []
_auto_startframenr = None

_cwd_basename = os.path.basename(os.getcwd())
if 'BeamPos' in _cwd_basename or 'DetZBeamPos' in _cwd_basename:
	# BeamPos mode: files have varying stems, navigate by sorted index
	_beampos_mode = True
	_all_tifs = glob.glob(os.path.join(os.getcwd(), '*.tif'))
	def _extract_num(fp):
		try:
			return int(os.path.splitext(os.path.basename(fp))[0].split('_')[-1])
		except ValueError:
			return float('inf')
	_beampos_files = sorted(_all_tifs, key=_extract_num)
	folder = os.getcwd()
	fnstem = ''
	if _beampos_files:
		_auto_startframenr = 0
		print(f"BeamPos mode: found {len(_beampos_files)} files, navigating by index")
else:
	_stem_path = os.path.join(folder, fnstem + '_')
	_matching = glob.glob(_stem_path + '*.tif')
	if _matching:
		_nums = []
		for _f in _matching:
			_base = os.path.splitext(os.path.basename(_f))[0]
			try:
				_nums.append(int(_base.split('_')[-1]))
			except ValueError:
				continue
		if _nums:
			_auto_startframenr = min(_nums)
			print(f"Auto-detected start frame number: {_auto_startframenr}")

def _quit():
	root.quit()
	root.destroy()

def _safe_remove_colorbar():
	"""Clear the persistent colorbar axes for reuse."""
	global cb
	if cb is not None:
		cb = None
	_cb_ax.clear()
	_cb_ax.set_visible(False)

def _cleanup_click_handlers():
	"""Disconnect click handlers and hide selection UI elements."""
	global cid, lb1, button7, button8
	if button7 is not None:
		button7.grid_forget()
		button7 = None
	if button8 is not None:
		button8.grid_forget()
		button8 = None
	if cid is not None:
		canvas.mpl_disconnect(cid)
		cid = None
	if lb1 is not None:
		lb1.grid_forget()
		lb1 = None

def getfilenames():
	if _beampos_mode:
		idx = framenr
		if 0 <= idx < len(_beampos_files):
			return [_beampos_files[idx], None]
		else:
			print(f"Warning: frame index {idx} out of range (0-{len(_beampos_files)-1})")
			idx = max(0, min(idx, len(_beampos_files)-1))
			return [_beampos_files[idx], None]
	medianfn = folder + '/' + fnstem + "_Median_Background_Distance_" + str(dist) + ".bin"
	fnr = startframenr + framenr + dist*nrfilesperdistance
	filefn = folder + '/' + fnstem + '_' + str(fnr).zfill(padding) + '.tif'
	return [filefn, medianfn]

def draw_plot(): # always the initial framenr and distance, will calculate the correct framenr automatically
	global initplot
	global imarr2
	global _img_artist, _img_logmode
	use_log = dolog.get() != 0
	if not initplot:
		lims = [a.get_xlim(), a.get_ylim()]
	# --- Load image data ---
	if maxoverframes.get() == 0 or _beampos_mode:
		fns = getfilenames()
		if HAS_TIFFFILE:
			imarr = tifffile.imread(fns[0]).astype(np.uint16)
		else:
			im = PIL.Image.open(fns[0])
			imarr = np.array(im, dtype=np.uint16)
		print("Read file " + fns[0])
		doMedian = var.get()
		fnprint = fns[0].replace(folder,'')
		if doMedian == 1 and fns[1] is not None:
			f = open(fns[1],'rb')
			print("Read file " + fns[1])
			median = np.fromfile(f,dtype=np.uint16,count=(NrPixelsY*NrPixelsZ))
			f.close()
			median = np.reshape(median,(NrPixelsZ,NrPixelsY))
			imarr2 = np.subtract(imarr.astype(int),median.astype(int))
			imarr2[imarr2<background] = 0
		else:
			imarr2 = imarr
	else:
		if var.get() == 1:
			fnthis = folder + '/' + fnstem + '_MaximumIntensityMedianCorrected_Distance_' + str(dist) + '.bin'
		else:
			fnthis = folder + '/' + fnstem + '_MaximumIntensity_Distance_' + str(dist) + '.bin'
		f = open(fnthis,'rb')
		print('Read file ' + fnthis)
		fnprint = fnthis.replace(folder,'')
		imarr = np.fromfile(f,dtype=np.uint16,count=(NrPixelsY*NrPixelsZ))
		f.close()
		imarr2 = np.reshape(imarr,(NrPixelsZ,NrPixelsY))
		imarr2[imarr2<background] = 0
	imarr2 = imarr2[::-1, ::-1].copy()
	# --- Prepare display data ---
	if use_log:
		minC = max(float(minThreshvar.get()), 1)
		maxC = max(float(vali.get()), 1)
		display_data = np.log(np.maximum(imarr2, 1).astype(np.float32))
		clim = (np.log(minC), np.log(maxC))
	else:
		display_data = imarr2
		clim = (float(minThreshvar.get()), float(vali.get()))
	# --- Fast path: reuse existing artist if possible ---
	can_reuse = (not initplot and _img_artist is not None and
	             _img_logmode == use_log and
	             _img_artist.get_array().shape == display_data.shape)
	if can_reuse:
		_img_artist.set_data(display_data)
		_img_artist.set_clim(*clim)
		a.set_xlim([lims[0][0], lims[0][1]])
		a.set_ylim([lims[1][0], lims[1][1]])
	else:
		a.clear()
		_img_artist = a.imshow(display_data, cmap=plt.get_cmap('bone'),
		                       interpolation='nearest', clim=clim)
		_img_logmode = use_log
		if initplot:
			initplot = 0
			a.invert_xaxis()
			a.invert_yaxis()
		else:
			a.set_xlim([lims[0][0], lims[0][1]])
			a.set_ylim([lims[1][0], lims[1][1]])
	# --- Update coordinate display and title ---
	numrows, numcols = imarr2.shape
	def format_coord(x, y):
		col = int(x+0.5)
		row = int(y+0.5)
		if col>=0 and col<numcols and row>=0 and row<numrows:
			z = imarr2[row,col]
			return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x,y,z)
		else:
			return 'x=%1.4f, y=%1.4f'%(x,y)
	a.format_coord = format_coord
	a.title.set_text("Image: " + fnprint)
	canvas.draw_idle()
	canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)

def plot_updater():
	global initplot
	global framenr
	global dist
	global oldVar
	global oldmaxoverframes
	global initvali
	global folder
	global fnstem
	global ndistances
	global NrPixelsY
	global NrPixelsZ
	global lsd
	global minThresh
	global nrfilesperdistance
	global startframenr
	global logscaler
	newlsd = float(lsdvar.get())
	newVar = var.get()
	newframenr = int(r.get())
	newdist = int(r2.get())
	newvali = int(vali.get())
	newfolder = foldervar.get()
	newfnstem = fnstemvar.get()
	newndistances = int(ndistancesvar.get())
	newNrPixelsY = int(NrPixelsYvar.get())
	newNrPixelsZ = int(NrPixelsZvar.get())
	newminThresh = float(minThreshvar.get())
	newnrfilesperdistance = int(nrfilesvar.get())
	newstartframenr = int(startframenrvar.get())
	newmaxoverframes = maxoverframes.get()
	newlogscaler = dolog.get()
	if ((initplot == 1) or
		(newlogscaler != logscaler) or
		(newmaxoverframes != oldmaxoverframes) or
		(newminThresh != minThresh) or
		(newnrfilesperdistance != nrfilesperdistance) or
		(newframenr != framenr) or
		(newdist != dist) or
		(newVar != oldVar) or
		(newvali != initvali) or
		(newfolder != folder) or
		(newfnstem != fnstem) or
		(newndistances != ndistances) or
		(newNrPixelsY != NrPixelsY) or
		(newNrPixelsZ != NrPixelsZ) or
		(newstartframenr != startframenr) or
		(newlsd !=lsd)):
		oldmaxoverframes = newmaxoverframes
		logscaler = newlogscaler
		nrfilesperdistance = newnrfilesperdistance
		minThresh = newminThresh
		lsd = newlsd
		NrPixelsY = newNrPixelsY
		NrPixelsZ = newNrPixelsZ
		ndistances = newndistances
		fnstem = newfnstem
		folder = newfolder
		oldVar = newVar
		initvali = newvali
		framenr = newframenr
		startframenr = newstartframenr
		dist = newdist
		draw_plot()
		if bPlot is not None:
			plotb()
		if micPlot is not None:
			plotmic()

def plotb():
	global horvert
	global clickpos
	global cb
	global initplotb
	global micPlot, micfiledata, bPlot
	_safe_remove_colorbar()
	if cb is None:
		initplotb = 1
	if horvert == 1:
		b.clear()
		xs = [clickpos[0][0],clickpos[1][0]]
		y = (clickpos[0][1]+clickpos[1][1])/2
		smallx = int(xs[0])
		largex = int(xs[1])
		if xs[0] > xs[1]:
			smallx = int(xs[1])
			largex = int(xs[0])
		xvals = np.linspace(int(smallx),int(largex),num=largex-smallx+1)
		y = int(y)
		zs = []
		for xval in xvals:
			zs.append(imarr2[y,int(xval)])
		b.plot(np.array(xvals),zs)
		b.title.set_text("LineOutHor")
		canvas.draw()
		canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)
	elif horvert == 2:
		b.clear()
		ys = [clickpos[0][1],clickpos[1][1]]
		x = (clickpos[0][0]+clickpos[1][0])/2
		smally = int(ys[0])
		largey = int(ys[1])
		if ys[0] > ys[1]:
			smally = int(ys[1])
			largey = int(ys[0])
		yvals = np.linspace(int(smally),int(largey),num=largey-smally+1)
		x = int(x)
		zs = []
		for yval in yvals:
			zs.append(imarr2[int(yval),x])
		b.plot(np.array(yvals),zs)
		b.title.set_text("LineOutVert")
		canvas.draw()
		canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)
	b.set_aspect('auto')
	bPlot = 1
	micPlot = None
	micfiledata = None

def onclick(event):
	global clickpos
	global lb1
	global cid
	clickpos.append((event.xdata, event.ydata))
	if len(clickpos) == 2:
		canvas.mpl_disconnect(cid)
		cid = None
		lb1.grid_forget()
		plotb()
	return

def plotbBox():
	global horvert
	global clickpos
	global cb
	global initplotb
	_safe_remove_colorbar()
	if cb is None:
		initplotb = 1
	b.clear()
	xsmall = int(min(clickpos[0][0],clickpos[1][0]))
	xlarge = int(max(clickpos[0][0],clickpos[1][0]))
	ysmall = int(min(clickpos[0][1],clickpos[1][1]))
	ylarge = int(max(clickpos[0][1],clickpos[1][1]))
	xvals = np.linspace(xsmall,xlarge,num=-xsmall+xlarge+1)
	yvals = np.linspace(ysmall,ylarge,num=-ysmall+ylarge+1)
	zs = []
	if horvert == 1: # Horizontal
		for xval in xvals:
			ztr = 0
			for yval in yvals:
				ztr += imarr2[int(yval),int(xval)]
			zs.append(ztr)
		b.plot(np.array(xvals),zs)
		b.title.set_text('BoxOutHor')
	elif horvert == 2:
		for yval in yvals:
			ztr = 0
			for xval in xvals:
				ztr += imarr2[int(yval),int(xval)]
			zs.append(ztr)
		b.plot(np.array(yvals),zs)
		b.title.set_text('BoxOutVer')
	canvas.draw()
	canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)
	b.set_aspect('auto')

def onclickbox(event):
	global clickpos
	global lb1
	global cid
	clickpos.append((event.xdata, event.ydata))
	if len(clickpos) == 2:
		canvas.mpl_disconnect(cid)
		cid = None
		lb1.grid_forget()
		plotbBox()
	return

def boxhor():
	global cid
	global clickpos
	global horvert
	global lb1
	_cleanup_click_handlers()
	horvert = 1 # 1 for horizontal, 2 for vertical
	clickpos = []
	cid = canvas.mpl_connect('button_press_event',onclickbox)
	lb1 = Tk.Label(master=analysisFrame,text="Click two edges of the box")
	lb1.grid(row=1,column=nrthird+1,columnspan=3,sticky=Tk.W)

def boxver():
	global cid
	global clickpos
	global horvert
	global lb1
	_cleanup_click_handlers()
	horvert = 2 # 1 for horizontal, 2 for vertical
	clickpos = []
	cid = canvas.mpl_connect('button_press_event',onclickbox)
	lb1 = Tk.Label(master=analysisFrame,text="Click two edges of the box")
	lb1.grid(row=1,column=nrthird+1,columnspan=3,sticky=Tk.W)

def horline():
	global cid
	global clickpos
	global horvert
	global lb1
	_cleanup_click_handlers()
	horvert = 1 # 1 for horizontal, 2 for vertical
	clickpos = []
	cid = canvas.mpl_connect('button_press_event',onclick)
	lb1 = Tk.Label(master=analysisFrame,text="Click two (almost) horizontal points")
	lb1.grid(row=1,column=nrthird+1,columnspan=3,sticky=Tk.W)

def vertline():
	global cid
	global clickpos
	global horvert
	global lb1
	_cleanup_click_handlers()
	horvert = 2 # 1 for horizontal, 2 for vertical
	clickpos = []
	cid = canvas.mpl_connect('button_press_event',onclick)
	lb1 = Tk.Label(master=analysisFrame,text="Click two (almost) vertical points")
	lb1.grid(row=1,column=nrthird+1,columnspan=3,sticky=Tk.W)

def top_destroyer():
	global top2
	global varsStore
	global distDiff
	global topSelectSpotsWindow
	for dist in range(ndistances):
		bcs[dist][0] = float(varsStore[dist][0].get())
		bcs[dist][1] = float(varsStore[dist][1].get())
	distDiff = float(distDiffVar.get())
	top.destroy()
	if top2 is not None:
		top2.destroy()
		top2 = None
	if topSelectSpotsWindow is not None:
		topSelectSpotsWindow.destroy()
		topSelectSpotsWindow = None
	if selectingspots == 1:
		selectspotsfcn()

def getpos(event):
	# We need to get the ix and iy for center of mass of the spot. For now I will trust what the user clicked on.
	global ix
	global iy
	global cid2
	global button7
	ix,iy = event.xdata,event.ydata
	button7 = Tk.Button(master=analysisFrame,text='Confirm Selection',command=confirmselectspot)
	button7.grid(row=1,column=nrthird+1,sticky=Tk.E)

def loadnewdistance():
	global topNewDistance
	topNewDistance.destroy()
	plot_updater()

def confirmselectspot():
	global spots
	global topNewDistance
	global vali
	x = ix
	y = iy
	xbc = bcs[dist][0]
	ybc = bcs[dist][1]
	spots[dist][0] = x - xbc
	spots[dist][1] = y - ybc
	spots[dist][2] = math.sqrt((spots[dist][0]*spots[dist][0]) + (spots[dist][1]*spots[dist][1]))
	topNewDistance = Tk.Toplevel()
	topNewDistance.title("Select Distance")
	textdisplay = "Old distance was: " + str(dist) + " Which distance now?"
	Tk.Label(topNewDistance,text=textdisplay).grid(row=0,column=1)
	edist = Tk.Entry(master=topNewDistance,textvariable=r2)
	edist.grid(row=1,column=1)
	edist.focus_set()
	buttondist = Tk.Button(master=topNewDistance,text='Load',command=loadnewdistance)
	buttondist.grid(row=2,column=1)
	buttonkill = Tk.Button(master=topNewDistance,text='Finished',command=loadnewdistance)
	buttonkill.grid(row=3,column=1)

def computedistances():
	global topDistanceResult
	global lsd
	nsols = int(ndistances*(ndistances-1)/2)
	xs = np.zeros(nsols)
	ys = np.zeros(nsols)
	idx = 0
	for i in range(ndistances):
		for j in range(i+1,ndistances):
			z1 = spots[i][1]
			z2 = spots[j][1]
			y1 = spots[i][0]
			y2 = spots[j][0]
			x = distDiff * (j-i)
			xs[idx] = x*z1/(z2-z1) - (distDiff * i)
			ys[idx] = y1 + (y2-y1)*z1/(z1-z2)
			idx += 1
	topDistanceResult = Tk.Toplevel()
	topDistanceResult.title("Distance computation result")
	textdisplay = "Calculated distances are: " + str(xs) + " Calculated Ys are: " + str(ys)
	lsd = np.mean(xs)
	lsdvar.set(str(lsd))
	Tk.Label(topDistanceResult,text=textdisplay,font=("Helvetica",16)).grid(row=0)
	buttonclose = Tk.Button(master=topDistanceResult,text="Okay",command=topDistanceResult.destroy)
	buttonclose.grid(row=1)

def closeselectspotshelp():
	global topSelectSpotsWindow
	global cid2
	global button8
	topSelectSpotsWindow.destroy()
	cid2 = canvas.mpl_connect('button_press_event',getpos)
	button8 = Tk.Button(master=analysisFrame,text='Compute Distances',command=computedistances)
	button8.grid(row=1,column=nrthird+2,sticky=Tk.W)

def selectspotsfcn():
	global topSelectSpotsWindow
	global cid
	global lb1
	if cid is not None:
		canvas.mpl_disconnect(cid)
		lb1.grid_forget()
		lb1 = None
		cid = None
	topSelectSpotsWindow = Tk.Toplevel()
	topSelectSpotsWindow.title("Spot selection guide")
	Tk.Label(topSelectSpotsWindow,text="Use the following steps as a guide to select spots.").grid(row=0,columnspan=2)
	Tk.Label(topSelectSpotsWindow,text="1. Make sure you have the correct Beam centers and difference in distances, otheriwse click Enter Beam Centers.").grid(row=1, column=0)
	button5 = Tk.Button(master=topSelectSpotsWindow,text='Enter Beam Centers',command=bcwindow)
	button5.grid(row=1,column=1,sticky=Tk.W)
	Tk.Label(topSelectSpotsWindow,text="2. It is recommended to have median correction enabled.").grid(row=2,columnspan=2)
	Tk.Label(topSelectSpotsWindow,text="3. Starting from the first distance, click on or close to a diffraction spot and then click Confirm Selection.").grid(row=3,columnspan=2)
	Tk.Label(topSelectSpotsWindow,text="4. Repeat this for each distance and select the same spot.").grid(row=4,columnspan=2)
	Tk.Label(topSelectSpotsWindow,text="5. Click Compute Distances once finished.").grid(row=5,columnspan=2)
	button6 = Tk.Button(master=topSelectSpotsWindow,text="Ready!",command=closeselectspotshelp)
	button6.grid(row=6,columnspan=2)

def bcwindow():
	global bcs
	global top
	global varsStore
	global distDiffVar
	global cid
	global lb1
	global ndistances
	if cid is not None:
		canvas.mpl_disconnect(cid)
		lb1.grid_forget()
		lb1 = None
		cid = None
	ndistances = int(ndistancesvar.get())
	nRows = ndistances
	nCols = 2
	top = Tk.Toplevel()
	top.title("Enter beam center values (pixels)")
	Tk.Label(top,text="Enter beam center values (pixels). We assume center of the beam is the rotation axis.").grid(row=0,columnspan=3)
	varsStore = []
	for dist in range(nRows):
		labeltext = "Distance " + str(dist)
		Tk.Label(top,text=labeltext).grid(row=dist+1)
		var1 = Tk.StringVar()
		var2 = Tk.StringVar()
		var1.set(str(bcs[dist][0]))
		var2.set(str(bcs[dist][1]))
		Tk.Entry(top,textvariable=var1).grid(row=dist+1,column=1)
		Tk.Entry(top,textvariable=var2).grid(row=dist+1,column=2)
		varsStore.append([var1,var2])
	Tk.Label(top,text="Difference in distances eg. 1000 microns").grid(row=ndistances+1)
	distDiffVar = Tk.StringVar()
	distDiffVar.set(str(distDiff))
	Tk.Entry(top,textvariable=distDiffVar).grid(row=ndistances+1,column=1,columnspan=2)
	button = Tk.Button(top,text="Press this once done",command=top_destroyer)
	button.grid(row=ndistances+2,column=0,columnspan=3)

def top2destroyer():
	global top2
	top2.destroy()
	selectspotsfcn()

def selectspots():
	global top2
	global selectingspots
	global cid
	global lb1
	if cid is not None:
		canvas.mpl_disconnect(cid)
		lb1.grid_forget()
		lb1 = None
		cid = None
	if not np.any(bcs):
		top2 = Tk.Toplevel()
		top2.title("Warning")
		Tk.Label(top2,text="All beam centers are 0. Do you want to continue?").grid(row=0,sticky=Tk.W)
		Tk.Button(top2,text="Go ahead with zeros.",command=top2destroyer).grid(row=1,column=0,sticky=Tk.W)
		Tk.Button(top2,text="Edit beam centers.",command=bcwindow).grid(row=1,column=1,sticky=Tk.W)
		selectingspots = 1
	else:
		selectspotsfcn()

def folderselect():
	global folder
	global foldervar
	folder = tkFileDialog.askdirectory()
	foldervar.set(folder)

def incr_plotupdater():
	global r
	global framenr
	framenr = int(r.get())
	r.set(str(framenr+1))
	plot_updater()

def decr_plotupdater():
	global r
	global framenr
	framenr = int(r.get())
	r.set(str(framenr-1))
	plot_updater()

def killtopGetGrain():
	global topGetGrain
	global om,pos,latC,wl,startome,omestep, sg, maxringrad
	for i in range(9):
		om[i] = float(omvar[i].get())
	for i in range(3):
		pos[i] = float(posvar[i].get())
	for i in range(6):
		latC[i] = float(latCvar[i].get())
	wl = float(wlvar.get())
	startome = float(startomevar.get())
	omestep = float(omestepvar.get())
	sg = int(sgvar.get())
	maxringrad = float(maxringradvar.get())
	topGetGrain.destroy()
	makespots()

def getgrain():
	# open new toplevel window, either read from file or take input, then close
	global topGetGrain
	global omvar,posvar,latCvar,wlvar,startomevar,omestepvar,sgvar,maxringradvar
	topGetGrain = Tk.Toplevel()
	topGetGrain.title("Load Grain into memory")
	Tk.Label(master=topGetGrain,text="Please enter the orientation matrix, position, lattice parameter, wavelength, startomega, omegastep, spacegroup, maxringrad",font=("Helvetica",12)).grid(row=1,columnspan=10)
	Tk.Label(master=topGetGrain,text="Orientation Matrix: ").grid(row=2,column=1)
	omvar = []
	posvar = []
	latCvar = []
	for i in range(9):
		var1 = Tk.StringVar()
		var1.set(str(om[i]))
		Tk.Entry(topGetGrain,textvariable=var1).grid(row=2,column=i+2)
		omvar.append(var1)
	Tk.Label(master=topGetGrain,text="Grain Position (microns): ").grid(row=3,column=1)
	for i in range(3):
		var1 = Tk.StringVar()
		var1.set(str(pos[i]))
		Tk.Entry(topGetGrain,textvariable=var1).grid(row=3,column=i+2)
		posvar.append(var1)
	Tk.Label(master=topGetGrain,text="Lattice parameter (Angstrom): ").grid(row=4,column=1)
	for i in range(6):
		var1 = Tk.StringVar()
		var1.set(str(latC[i]))
		Tk.Entry(topGetGrain,textvariable=var1).grid(row=4,column=i+2)
		latCvar.append(var1)
	Tk.Label(master=topGetGrain,text="Wavelength (Angstrom): ").grid(row=5,column=1)
	wlvar.set(str(wl))
	Tk.Entry(topGetGrain,textvariable=wlvar).grid(row=5,column=2)
	Tk.Label(master=topGetGrain,text="StartOmega (Degrees): ").grid(row=6,column=1)
	startomevar.set(str(startome))
	Tk.Entry(topGetGrain,textvariable=startomevar).grid(row=6,column=2)
	Tk.Label(master=topGetGrain,text="OmegaStep (Degrees): ").grid(row=7,column=1)
	omestepvar.set(str(omestep))
	Tk.Entry(topGetGrain,textvariable=omestepvar).grid(row=7,column=2)
	Tk.Label(master=topGetGrain,text="Space Group Number").grid(row=8,column=1)
	sgvar.set(str(sg))
	Tk.Entry(topGetGrain,textvariable=sgvar).grid(row=8,column=2)
	Tk.Label(master=topGetGrain,text="MaxRingRad (microns)").grid(row=9,column=1)
	maxringradvar.set(str(maxringrad))
	Tk.Entry(topGetGrain,textvariable=maxringradvar).grid(row=9,column=2)
	buttonConfirmGrainParams = Tk.Button(master=topGetGrain,text="Confirm",command=killtopGetGrain)
	buttonConfirmGrainParams.grid(row=10,columnspan=10)

def YZ4mREta(R,Eta):
	return -R*math.sin(Eta*deg2rad),R*math.cos(Eta*deg2rad)

def rotationTransforms(ts):
	txr = ts[0]*deg2rad
	tyr = ts[1]*deg2rad
	tzr = ts[2]*deg2rad
	Rx = np.array([[1,0,0],[0,cos(txr),-sin(txr)],[0,sin(txr),cos(txr)]])
	Ry = np.array([[cos(tyr),0,sin(tyr)],[0,1,0],[-sin(tyr),0,cos(tyr)]])
	Rz = np.array([[cos(tzr),-sin(tzr),0],[sin(tzr),cos(tzr),0],[0,0,1]])
	return np.dot(Rx,np.dot(Ry,Rz))

def DisplacementSpots(a, b, Lsd, yi, zi, omega):
    OmegaRad = deg2rad * omega
    sinOme = math.sin(OmegaRad)
    cosOme = math.cos(OmegaRad)
    xa = a*cosOme - b*sinOme
    ya = a*sinOme + b*cosOme
    t = 1 - (xa/Lsd)
    return [ya + (yi*t) , t*zi]

def plot_update_spot():
	global r, spotnrvar, spotnr, startframenr
	thislsd = float(lsdvar.get()) + float(r2.get()) * float(distDiffVar.get())
	simlsd = float(lsdvar.get())
	endome = startome + nrfilesperdistance * omestep
	minOme = min(startome,endome)
	maxOme = max(startome,endome)
	startframenr = int(startframenrvar.get())
	thisome = float(simulatedspots[spotnr-1].split(' ')[2])
	rad = float(simulatedspots[spotnr-1].split(' ')[0])
	rad = rad * thislsd/simlsd
	eta = float(simulatedspots[spotnr-1].split(' ')[1])
	frameNrToRead = int((float(thisome)-float(startome))/omestep)
	r.set(str(frameNrToRead))
	spotnrvar.set(str(spotnr))
	ys,zs = YZ4mREta(rad,eta)
	ya = pos[0]*math.sin(thisome*deg2rad) + pos[1]*math.cos(thisome*deg2rad)
	xa = -pos[1]*math.sin(thisome*deg2rad) + pos[0]*math.cos(thisome*deg2rad)
	yn = (ya + ys*(1-(xa/thislsd)))/pixelsize + bcs[dist][0]
	zn = (zs*(1-(xa/thislsd)))/pixelsize + bcs[dist][1]
	#~ print([pos[0], pos[1], ys, zs, xa, ya, yn, zn, rad, eta,thisome,frameNrToRead])
	while ((yn > NrPixelsY) or
		   (zn > NrPixelsZ) or
		   (yn < 0) or
		   (zn < 0) or
		   (thisome < minOme) or
		   (thisome > maxOme)):
		spotnr += 1
		thisome = float(simulatedspots[spotnr-1].split(' ')[2])
		rad = float(simulatedspots[spotnr-1].split(' ')[0])
		rad = rad * thislsd/simlsd
		eta = float(simulatedspots[spotnr-1].split(' ')[1])
		frameNrToRead = int((float(thisome)-float(startome))/omestep)
		r.set(str(frameNrToRead))
		spotnrvar.set(str(spotnr))
		ys,zs = YZ4mREta(rad,eta)
		ya =  pos[0]*math.sin(thisome*deg2rad) + pos[1]*math.cos(thisome*deg2rad)
		xa = -pos[1]*math.sin(thisome*deg2rad) + pos[0]*math.cos(thisome*deg2rad)
		yn = (ya + ys*(1-(xa/thislsd)))/pixelsize + bcs[dist][0]
		zn = (zs*(1-(xa/thislsd)))/pixelsize + bcs[dist][1]
		#~ print([pos[0], pos[1], ys, zs, xa, ya, yn, zn, rad, eta,thisome,frameNrToRead])
	plot_updater()
	a.scatter(yn,zn,s=25,color='red')
	# Show Beam Center, unrotated Grain position, rotated Grain Position, Undisplaced Spot Position
	a.scatter(bcs[dist][0],bcs[dist][1],s=30,color='blue',marker=(5,0))
	a.scatter(bcs[dist][0]+(pos[1]/pixelsize),bcs[dist][1],s=30,color='green',marker=(5,0))
	a.scatter(bcs[dist][0]+(ya/pixelsize),bcs[dist][1],s=30,color='yellow',marker=(5,0))
	a.scatter(bcs[dist][0]+(ys/pixelsize),bcs[dist][1]+(zs/pixelsize),s=30,color='magenta',marker=(5,0))
	print("Look at the red spot for the diffraction spot position of SpotNr : "+str(spotnr))
	canvas.draw()
	canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)

def incr_spotnr():
	global spotnr
	spotnr += 1
	plot_update_spot()

def update_spotnr():
	global spotnr
	spotnr = int(spotnrvar.get())
	plot_update_spot()

def makespots():
	global spotnrvar
	global simulatedspots
	global r
	# Use TemporaryDirectory to contain output files
	with tempfile.TemporaryDirectory() as temp_dir:
		pfname = os.path.join(temp_dir, "ps.txt")
		with open(pfname, 'w') as f:
			f.write("SpaceGroup %d\n"%(sg))
			f.write("Wavelength %lf\n"%(wl))
			lsd = float(lsdvar.get())
			f.write("Lsd %lf\n"%(lsd))
			f.write("MaxRingRad %lf\n"%(maxringrad))
			f.write("LatticeConstant %lf %lf %lf %lf %lf %lf\n"%(latC[0],latC[1],latC[2],latC[3],latC[4],latC[5]))
		
		if midas_config and midas_config.MIDAS_NF_BIN_DIR:
			hklpath = os.path.join(midas_config.MIDAS_NF_BIN_DIR, 'GetHKLList')
		else:
			hklpath = os.path.expanduser('~/opt/MIDAS/NF_HEDM/bin/GetHKLList')
		
		subprocess.run([hklpath, pfname], check=True, cwd=temp_dir)
		
		if midas_config and midas_config.MIDAS_NF_BIN_DIR:
			genseedorpath = os.path.join(midas_config.MIDAS_NF_BIN_DIR, 'GenSeedOrientationsFF2NFHEDM')
			diffrspotspath = os.path.join(midas_config.MIDAS_NF_BIN_DIR, 'SimulateDiffractionSpots')
		else:
			genseedorpath = os.path.expanduser('~/opt/MIDAS/NF_HEDM/bin/GenSeedOrientationsFF2NFHEDM')
			diffrspotspath = os.path.expanduser('~/opt/MIDAS/NF_HEDM/bin/SimulateDiffractionSpots')

		orinfn = os.path.join(temp_dir, "orin.txt")
		oroutfn = os.path.join(temp_dir, "orout.txt")
		instr = "120 %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n"%(om[0],om[1],om[2],om[3],om[4],
				om[5],om[6],om[7],om[8],pos[0],pos[1],pos[2],latC[0],latC[1],latC[2],latC[3],latC[4],latC[5])
		
		with open(orinfn, 'w') as f:
			f.write(instr)

		subprocess.run([genseedorpath, orinfn, oroutfn], check=True, cwd=temp_dir)
		subprocess.run([diffrspotspath, str(lsd), oroutfn], check=True, cwd=temp_dir)
		
		# Read back the results from temp_dir
		spotsfn = os.path.join(temp_dir, 'SimulatedDiffractionSpots.txt')
		if os.path.exists(spotsfn):
			with open(spotsfn, 'r') as f:
				simulatedspots = f.readlines()
		else:
			print("Error: SimulatedDiffractionSpots.txt not found.")
			simulatedspots = []

	Tk.Label(master=analysisFrame,text="SpotNumber").grid(row=1,column=nrthird+1,sticky=Tk.W)
	Tk.Entry(master=analysisFrame,textvariable=spotnrvar,width=3).grid(row=1,column=nrthird+2,sticky=Tk.W)
	if simulatedspots:
		plot_update_spot()

	# put a frame with +,- and load
	Tk.Button(master=analysisFrame,text="+",command=incr_spotnr,font=("Helvetica",12)).grid(row=1,column=nrthird+3,sticky=Tk.W)
	Tk.Button(master=analysisFrame,text="Load",command=update_spotnr,font=("Helvetica",12)).grid(row=1,column=nrthird+4,sticky=Tk.W)

def median():
	# Iterate sequentially instead of using Pool/Popen logic which is hard to secure
	# Since these are likely fast, sequential is safer for the GUI
	for thisdist in range(ndistances):
		# Create a temp file for params
		with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
			pfname = f.name
			f.write('extReduced bin\n')
			f.write('extOrig tif\n')
			f.write('WFImages 0\n')
			f.write(f'OrigFileName {fnstem}\n')
			tempnr = startframenr + thisdist*(nrfilesperdistance - int(nrfilesmedianvar.get()))
			f.write(f'NrFilesPerDistance {nrfilesmedianvar.get()}\n')
			f.write(f'NrPixelsY {NrPixelsY}\n')
			f.write(f'NrPixelsZ {NrPixelsZ}\n')
			f.write(f'DataDirectory {folder}\n')
			f.write(f'RawStartNr {tempnr}\n')
			f.write(f'ReducedFileName {fnstem}\n')

		if midas_config and midas_config.MIDAS_NF_BIN_DIR:
			cmd = os.path.join(midas_config.MIDAS_NF_BIN_DIR,'MedianImageLibTiff')
		else:
			cmd = os.path.expanduser('~/opt/MIDAS/NF_HEDM/bin/MedianImageLibTiff')
			
		# subprocess.run([cmd, pfname, str(thisdist+1), '10'], check=True)
		# NOTE: original used Popen. Median calculation might be slow. 
		# Implementing fully blocking here will freeze GUI. 
		# But original used Pool(len(processes)).map(...) which blocked until ALL returned anyway! 
		# So blocking here is behaviorally equivalent.
		try:
			subprocess.run([cmd, pfname, str(thisdist+1), '10'], check=True)
			# Clean up temp file
			os.remove(pfname)
		except subprocess.CalledProcessError as e:
			print(f"Error running MedianImageLibTiff: {e}")

	print('Calculated median for all distances.')

def micfileselect():
	global micfile
	micfile = tkFileDialog.askopenfilename(
		title='Select the compressed binary file.',
		initialdir=folder if folder else os.getcwd(),
	)
	if not micfile:
		return False
	return True

def plotmic():
	global initplotb
	global colVar
	global cb
	global micfiledatacut
	global micPlot, bPlot
	is_first_plot = initplotb
	micPlot = 1
	bPlot = None
	if not initplotb:
		lims = [b.get_xlim(), b.get_ylim()]
		_safe_remove_colorbar()
	b.clear()
	col = colVar.get()
	if micfiletype == 1:
		micfiledatacut = np.copy(micfiledata)
		micfiledatacut = micfiledatacut[ micfiledatacut[:,10] > float(cutconfidencevar.get()) , :]
		_safe_remove_colorbar()
		if col == 10:
			sc = b.scatter(micfiledatacut[:,3],micfiledatacut[:,4],c=micfiledatacut[:,col],lw=0,cmap=plt.get_cmap('jet'),vmax=float(maxConfVar.get()))
		else:
			sc = b.scatter(micfiledatacut[:,3],micfiledatacut[:,4],c=micfiledatacut[:,col],lw=0,cmap=plt.get_cmap('jet'))
		if initplotb:
			initplotb = 0
		else:
			b.set_xlim([lims[0][0],lims[0][1]])
			b.set_ylim([lims[1][0],lims[1][1]])
		if col == 7:
			b.title.set_text("MicFile (Euler0)")
		elif col == 8:
			b.title.set_text("MicFile (Euler1)")
		elif col == 9:
			b.title.set_text("MicFile (Euler2)")
		elif col == 10:
			b.title.set_text("MicFile (Confidence Coloring)")
		elif col == 0:
			b.title.set_text("MicFile (OrientationID)")
		elif col == 11:
			b.title.set_text("MicFile (PhaseNr)")
	elif micfiletype == 2:
		micfiledatacut = np.copy(micfiledata)
		badcoords = micfiledatacut[:sizeX*sizeY]
		badcoords = badcoords < float(cutconfidencevar.get())
		exten = [refX,refX+sizeX,refY+sizeY,refY]
		#~ micfiledatacut = micfiledatacut[micfiledatacut[:,0] > float(cutconfidencevar.get()), :]
		_safe_remove_colorbar()
		if col == 7: # Euler0
			micfiledatacut = micfiledatacut[sizeX*sizeY:sizeX*sizeY*2]
			micfiledatacut[badcoords] = -15.0
			micfiledatacut = micfiledatacut.reshape((sizeY,sizeX))
			sc = b.imshow(np.ma.masked_where(micfiledatacut == -15.0,micfiledatacut),cmap=plt.get_cmap('jet'),interpolation='nearest',extent=exten)
			b.title.set_text("MicMap (Euler0)")
		if col == 8: # Euler1
			micfiledatacut = micfiledatacut[sizeX*sizeY*2:sizeX*sizeY*3]
			micfiledatacut[badcoords] = -15.0
			micfiledatacut = micfiledatacut.reshape((sizeY,sizeX))
			sc = b.imshow(np.ma.masked_where(micfiledatacut == -15.0,micfiledatacut),cmap=plt.get_cmap('jet'),interpolation='nearest',extent=exten)
			b.title.set_text("MicMap (Euler1)")
		if col == 9: # Euler2
			micfiledatacut = micfiledatacut[sizeX*sizeY*3:sizeX*sizeY*4]
			micfiledatacut[badcoords] = -15.0
			micfiledatacut = micfiledatacut.reshape((sizeY,sizeX))
			sc = b.imshow(np.ma.masked_where(micfiledatacut == -15.0,micfiledatacut),cmap=plt.get_cmap('jet'),interpolation='nearest',extent=exten)
			b.title.set_text("MicMap (Euler2)")
		if col == 10: # Confidence
			micfiledatacut = micfiledatacut[:sizeX*sizeY]
			micfiledatacut[badcoords] = -15.0
			micfiledatacut = micfiledatacut.reshape((sizeY,sizeX))
			sc = b.imshow(np.ma.masked_where(micfiledatacut == -15.0,micfiledatacut),cmap=plt.get_cmap('jet'),interpolation='nearest',extent=exten,vmax=float(maxConfVar.get()))
			b.title.set_text("MicMap (Confidence)")
		if col == 0: # OrientationID
			micfiledatacut = micfiledatacut[sizeX*sizeY*4:sizeX*sizeY*5]
			micfiledatacut[badcoords] = -15.0
			micfiledatacut = micfiledatacut.reshape((sizeY,sizeX))
			sc = b.imshow(np.ma.masked_where(micfiledatacut == -15.0,micfiledatacut),cmap=plt.get_cmap('jet'),interpolation='nearest',extent=exten)
			b.title.set_text("MicMap (OrientationID)")
		if col == 11: # PhaseNr
			micfiledatacut = micfiledatacut[sizeX*sizeY*5:sizeX*sizeY*6]
			micfiledatacut[badcoords] = -15.0
			micfiledatacut = micfiledatacut.reshape((sizeY,sizeX))
			sc = b.imshow(np.ma.masked_where(micfiledatacut == -15.0,micfiledatacut),cmap=plt.get_cmap('jet'),interpolation='nearest',extent=exten)
			b.title.set_text("MicMap (PhaseNr)")
		if col == 12: # KAM
			kamfile = micfile + ".kam"
			if os.path.exists(kamfile):
				with open(kamfile, 'rb') as f:
					kamdata = np.fromfile(f, dtype=np.double)[4:]
				kamdata[badcoords] = -15.0
				kamdata = kamdata.reshape((sizeY,sizeX))
				sc = b.imshow(np.ma.masked_where(kamdata == -15.0,kamdata),cmap=plt.get_cmap('jet'),interpolation='nearest',extent=exten)
				b.title.set_text("MicMap (KAM)")
			else:
				print(f"KAM file not found: {kamfile}")
		if col == 13: # GROD
			grodfile = micfile + ".grod"
			if os.path.exists(grodfile):
				with open(grodfile, 'rb') as f:
					groddata = np.fromfile(f, dtype=np.double)[4:]
				groddata[badcoords] = -15.0
				groddata = groddata.reshape((sizeY,sizeX))
				sc = b.imshow(np.ma.masked_where(groddata == -15.0,groddata),cmap=plt.get_cmap('jet'),interpolation='nearest',extent=exten)
				b.title.set_text("MicMap (GROD)")
			else:
				print(f"GROD file not found: {grodfile}")
		if col == 14: # GrainID
			grainfile = micfile + ".grainId"
			if os.path.exists(grainfile):
				with open(grainfile, 'rb') as f:
					graindata = np.fromfile(f, dtype=np.double)[4:]
				graindata[badcoords] = -15.0
				graindata = graindata.reshape((sizeY,sizeX))
				# GrainIDs are discrete integers, keep jet or use tab20c etc.
				sc = b.imshow(np.ma.masked_where((graindata == -15.0) | (graindata == 0), graindata),cmap=plt.get_cmap('jet'),interpolation='nearest',extent=exten)
				b.title.set_text("MicMap (GrainID File)")
			else:
				print(f"GrainID file not found: {grainfile}")
		if initplotb:
			initplotb = 0
			b.invert_yaxis()
		else:
			b.set_xlim([lims[0][0],lims[0][1]])
			b.set_ylim([lims[1][0],lims[1][1]])
	_cb_ax.set_visible(True)
	cb = figur.colorbar(sc, cax=_cb_ax)
	b.set_aspect('equal')
	if is_first_plot:
		figur.tight_layout()
	canvas.draw()
	canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)

def load_mic():
	global micfiledata, initplotb, micfiletype, sizeX, sizeY, refX, refY
	initplotb = 1
	if not micfileselect():
		return
	print(micfile)
	if (micfile[-3:] == 'map'):
		micfiletype = 2
		f = open(micfile, 'rb')
		sizeX = int(np.fromfile(f,dtype=np.double,count=1)[0])
		sizeY = int(np.fromfile(f,dtype=np.double,count=1)[0])
		refX = int(np.fromfile(f,dtype=np.double,count=1)[0])
		refY = int(np.fromfile(f,dtype=np.double,count=1)[0])
		micfiledata = np.fromfile(f,dtype=np.double)
		f.close()
		print([sizeX,sizeY,micfiledata.size])
		if (micfiledata.size/7) != (sizeX*sizeY):
			print("Size of the map file is not correct. Please check that the file was written properly.")
	else:
		micfiletype = 1
		with open(micfile, 'r') as f:
			micfiledata = np.genfromtxt(f,skip_header=4)
	plotmic()

def euler2orientmat(Euler):
    psi = Euler[0]
    phi = Euler[1]
    theta = Euler[2]
    cps = math.cos(psi)
    cph = math.cos(phi)
    cth = math.cos(theta)
    sps = math.sin(psi)
    sph = math.sin(phi)
    sth = math.sin(theta)
    m_out = np.zeros(9)
    m_out[0] = cth * cps - sth * cph * sps
    m_out[1] = -cth * cph * sps - sth * cps
    m_out[2] = sph * sps
    m_out[3] = cth * sps + sth * cph * cps
    m_out[4] = cth * cph * cps - sth * sps
    m_out[5] = -sph * cps
    m_out[6] = sth * sph
    m_out[7] = cth * sph
    m_out[8] = cph
    return m_out

def calcSpots(clickpos):
	global om, pos
	xs = micfiledatacut[:,3]
	ys = micfiledatacut[:,4]
	xdiff = xs - clickpos[0]
	ydiff = ys - clickpos[1]
	lendiff = np.square(xdiff) + np.square(ydiff)
	rowbest = lendiff == min(lendiff)
	rowcontents = micfiledatacut[rowbest,:][0]
	Euler = rowcontents[7:10]
	om = euler2orientmat(Euler)
	pos[0] = rowcontents[3]
	pos[1] = rowcontents[4]
	pos[2] = 0
	getgrain()

def onclickmicfile(event):
	clickpos = [event.xdata, event.ydata]
	calcSpots(clickpos)
	canvas.mpl_disconnect(ce2)

def killlsdwindow():
	global toplsd
	toplsd.destroy()
	lsd = float(lsdvar.get())

def lsdwindow():
	global toplsd
	toplsd = Tk.Toplevel()
	toplsd.title('Sample to Detector Distance Window')
	Tk.Label(master=toplsd,text='Please input the first sample to detector distance (in um): ').grid(row=0,column=0)
	Tk.Entry(master=toplsd,textvariable=lsdvar,width=10).grid(row=0,column=1)
	Tk.Button(master=toplsd,text='Confirm',command=killlsdwindow).grid(row=1,column=0,columnspan=2)

def selectpoint():
	global ce2
	if micfiledata is None:
		load_mic()
	if bcs[0][0] == 0:
		bcwindow()
	if float(lsdvar.get()) == 0:
		lsdwindow()
	ce2 = canvas.mpl_connect('button_press_event',onclickmicfile)

def selectfirstfile():
	global firstfile
	firstfile = tkFileDialog.askopenfilename()

def firstfileselect():
	global folder, foldervar, fnstem, fnstemvar, startframenr, startframenrvar, r, framenr, padding
	selectfirstfile()
	# get folder, fnstem, startfilenrfirstlayer, framenr=0
	framenr = 0
	r.set(str(framenr))
	idx = 0
	while idx < len(firstfile):
		idx = firstfile.find('/',idx)
		if idx == -1:
			break
		folderpos = idx
		idx += 1
	folder = firstfile[:folderpos]
	foldervar.set(folder)
	# we want to get padding count now
	fntot = firstfile[folderpos+1:]
	padding = len(fntot.split('_')[-1].split('.')[0])
	extlen = len(fntot.split('_')[-1].split('.')[1])
	fnstem = firstfile[folderpos+1:][:-(2 + padding + extlen)]
	fnstemvar.set(fnstem)
	startframenr = int(firstfile[folderpos+1:][-(padding+extlen+1):-(extlen+1)])
	startframenrvar.set(str(startframenr))

def findOrientation():
	paramsfile = tkFileDialog.askopenfilename(title='Select the parameter file to use.',filetypes=(('Txt files','*.txt'),('All Files','*.*')))

# Global constants initialization
imarr2 = None
_img_artist = None
_img_logmode = False
initplot = 1
framenr = 0
startframenr = _auto_startframenr if _auto_startframenr is not None else 0
sizeX = 0
sizeY = 0
refX = 0
refY = 0
dist = 0
horvert = 0
oldVar = 0
bcs = np.zeros((ndistances,2))
spots = np.zeros((ndistances,3))
pixelsize = 1.48
initvali=100
cid = None
cid2 = None
lb1 = None
top = None
top2 = None
button7 = None
button8 = None
distDiff = 0
ix = 0
iy = 0
cb = None
distDiffVar = None
topSelectSpotsWindow = None
topNewDistance = None
topDistanceResult = None
selectingspots = 0
clickpos = []
varsStore = []
lsd = 0
omvar = []
posvar = []
latCvar = []
om = np.zeros(9)
pos = np.zeros(3)
latC = np.zeros(6)
wl = 0
startome = 0
omestep = 0
sg = 0
maxringrad = 0
spotnr = 1
logscaler = 0

# Main funcion
root = Tk.Tk()
root.wm_title("NF display v0.2 Dt. 2024/02/10 hsharma@anl.gov")
default_font = ("Helvetica", 14)
root.option_add("*Font", default_font)
figur = Figure(figsize=(17,8),dpi=100)
canvas = FigureCanvasTkAgg(figur,master=root)
a = figur.add_subplot(121,aspect='equal')
b = figur.add_subplot(122)
from mpl_toolkits.axes_grid1 import make_axes_locatable
_divider = make_axes_locatable(b)
_cb_ax = _divider.append_axes("right", size="5%", pad=0.05)
_cb_ax.set_visible(False)  # Hidden until first mic plot
b.title.set_text("LineOuts/MicFile")
b.title.set_fontsize(16)
a.title.set_text("Image")
a.title.set_fontsize(16)
canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)
toolbar_frame = Tk.Frame(root)
toolbar_frame.grid(row=figrowspan+5,column=0,columnspan=5,sticky=Tk.W)
toolbar = NavigationToolbar2Tk( canvas, toolbar_frame )
toolbar.update()

vali = Tk.StringVar()
vali.set(str(100))
var = Tk.IntVar()
wlvar = Tk.StringVar()
startomevar = Tk.StringVar()
omestepvar = Tk.StringVar()
sgvar = Tk.StringVar()
maxringradvar = Tk.StringVar()
cutconfidencevar = Tk.StringVar()
cutoffconfidence = 0
cutconfidencevar.set(str(cutoffconfidence))
maxConfVar = Tk.StringVar()
maxConf = 1
maxConfVar.set(str(maxConf))
initplotb = 1
colVar = Tk.IntVar()
colVar.set(10)
bPlot = None
micPlot = None
micfiledata = None
dolog = Tk.IntVar()
r = Tk.StringVar()
r.set(str(framenr))
foldervar = Tk.StringVar()
foldervar.set(folder)
fnstemvar = Tk.StringVar()
fnstemvar.set(fnstem)
startframenrvar = Tk.StringVar()
startframenrvar.set(str(startframenr))
spotnrvar = Tk.StringVar()
ndistancesvar = Tk.StringVar()
ndistancesvar.set(str(ndistances))
NrPixelsYvar = Tk.StringVar()
NrPixelsYvar.set(str(NrPixelsY))
NrPixelsZvar = Tk.StringVar()
NrPixelsZvar.set(str(NrPixelsZ))
r2 = Tk.StringVar()
r2.set(str(0))
minThresh = 0
minThreshvar = Tk.StringVar()
minThreshvar.set(str(minThresh))
pxvar = Tk.StringVar()
pxvar.set(str(pixelsize))
lsdvar = Tk.StringVar()
lsdvar.set(str(lsd))
nrfilesvar = Tk.StringVar()
nrfilesvar.set(str(nrfilesperdistance))
nrfilesmedianvar = Tk.StringVar()
nrfilesmedianvar.set(str(nrfilesperdistance))
oldmaxoverframes = 0
maxoverframes = Tk.IntVar()
nrthird = 6
micfiletype = 1

## Main Control Frame
mainControlFrame = Tk.Frame(root)
mainControlFrame.grid(row=figrowspan+1,column=1,rowspan=4,sticky=Tk.W+Tk.N,padx=2,pady=2)

# --- File I/O Frame ---
fileFrame = Tk.LabelFrame(mainControlFrame, text="File I/O", font=default_font, padx=2, pady=2)
fileFrame.grid(row=0, column=0, sticky=Tk.W+Tk.E, padx=2, pady=2)
Tk.Button(master=fileFrame,text='FirstFile',command=firstfileselect,font=default_font).grid(row=0,column=0,sticky=Tk.W)
Tk.Label(master=fileFrame,text="Folder",font=default_font).grid(row=0,column=1,sticky=Tk.W)
Tk.Entry(master=fileFrame,textvariable=foldervar,width=20,font=default_font).grid(row=0,column=2,sticky=Tk.W)
Tk.Button(master=fileFrame,text="Browse",command=folderselect,font=default_font).grid(row=0,column=3,sticky=Tk.W)
Tk.Label(master=fileFrame,text="FNStem",font=default_font).grid(row=0,column=4,sticky=Tk.W)
Tk.Entry(master=fileFrame,textvariable=fnstemvar,width=12,font=default_font).grid(row=0,column=5,sticky=Tk.W)
Tk.Label(master=fileFrame,text="NrPixelsY",font=default_font).grid(row=0,column=6,sticky=Tk.W)
Tk.Entry(master=fileFrame,textvariable=NrPixelsYvar,width=5,font=default_font).grid(row=0,column=7,sticky=Tk.W)
Tk.Label(master=fileFrame,text="NrPixelsZ",font=default_font).grid(row=0,column=8,sticky=Tk.W)
Tk.Entry(master=fileFrame,textvariable=NrPixelsZvar,width=5,font=default_font).grid(row=0,column=9,sticky=Tk.W)

# --- Image Settings Frame ---
imgFrame = Tk.LabelFrame(mainControlFrame, text="Image Settings", font=default_font, padx=2, pady=2)
imgFrame.grid(row=1, column=0, sticky=Tk.W+Tk.E, padx=2, pady=2)
Tk.Label(master=imgFrame,text="FrameNr",font=default_font).grid(row=0,column=0,sticky=Tk.W)
Tk.Entry(master=imgFrame,textvariable=r,width=6,font=default_font).grid(row=0,column=1,sticky=Tk.W)
Tk.Button(master=imgFrame,text='+',command=incr_plotupdater,font=("Helvetica",12)).grid(row=0,column=2,sticky=Tk.W)
Tk.Button(master=imgFrame,text='-',command=decr_plotupdater,font=("Helvetica",12)).grid(row=0,column=3,sticky=Tk.W)
Tk.Label(master=imgFrame,text="nDist",font=default_font).grid(row=0,column=4,sticky=Tk.W)
Tk.Entry(master=imgFrame,textvariable=ndistancesvar,width=3,font=default_font).grid(row=0,column=5,sticky=Tk.W)
Tk.Label(master=imgFrame,text="DistNr",font=default_font).grid(row=0,column=6,sticky=Tk.W)
Tk.Entry(master=imgFrame,textvariable=r2,width=3,font=default_font).grid(row=0,column=7,sticky=Tk.W)
Tk.Label(master=imgFrame,text="nFl/Dist",font=default_font).grid(row=0,column=8,sticky=Tk.W)
Tk.Entry(master=imgFrame,textvariable=nrfilesvar,width=4,font=default_font).grid(row=0,column=9,sticky=Tk.W)
Tk.Label(master=imgFrame,text="MinThr",font=default_font).grid(row=0,column=10,sticky=Tk.W)
Tk.Entry(master=imgFrame,textvariable=minThreshvar,width=4,font=default_font).grid(row=0,column=11,sticky=Tk.W)
Tk.Label(master=imgFrame,text="MaxThr",font=default_font).grid(row=0,column=12,sticky=Tk.W)
Tk.Entry(master=imgFrame,textvariable=vali,width=5,font=default_font).grid(row=0,column=13,sticky=Tk.W)
Tk.Checkbutton(master=imgFrame,text="Log",variable=dolog,font=default_font).grid(row=0,column=14,sticky=Tk.W)
Tk.Label(master=imgFrame,text="PxSz(um)",font=default_font).grid(row=0,column=15,sticky=Tk.W)
Tk.Entry(master=imgFrame,textvariable=pxvar,width=5,font=default_font).grid(row=0,column=16,sticky=Tk.W)

# --- Processing Frame ---
procFrame = Tk.LabelFrame(mainControlFrame, text="Processing", font=default_font, padx=2, pady=2)
procFrame.grid(row=2, column=0, sticky=Tk.W+Tk.E, padx=2, pady=2)
Tk.Label(master=procFrame,text="StartFileNr",font=default_font).grid(row=0,column=0,sticky=Tk.W)
Tk.Entry(master=procFrame,textvariable=startframenrvar,width=6,font=default_font).grid(row=0,column=1,sticky=Tk.W)
Tk.Button(master=procFrame,text='CalcMedian',command=median,font=default_font).grid(row=0,column=2,sticky=Tk.W)
Tk.Label(master=procFrame,text="nMedianFl",font=default_font).grid(row=0,column=3,sticky=Tk.W)
Tk.Entry(master=procFrame,textvariable=nrfilesmedianvar,width=4,font=default_font).grid(row=0,column=4,sticky=Tk.W)
Tk.Checkbutton(master=procFrame,text="MaxOverFr",variable=maxoverframes,font=default_font).grid(row=0,column=5,sticky=Tk.W)
Tk.Checkbutton(master=procFrame,text="SubtMedian",variable=var,font=default_font).grid(row=0,column=6,sticky=Tk.W)
Tk.Label(master=procFrame,text="Lsd(um)",font=default_font).grid(row=0,column=7,sticky=Tk.W)
Tk.Entry(master=procFrame,textvariable=lsdvar,width=8,font=default_font).grid(row=0,column=8,sticky=Tk.W)

# --- Analysis Frame ---
analysisFrame = Tk.LabelFrame(mainControlFrame, text="Analysis", font=default_font, padx=2, pady=2)
analysisFrame.grid(row=3, column=0, sticky=Tk.W+Tk.E, padx=2, pady=2)
Tk.Button(master=analysisFrame,text='LineOutH',command=horline,font=default_font).grid(row=0,column=0,sticky=Tk.W)
Tk.Button(master=analysisFrame,text='LineOutV',command=vertline,font=default_font).grid(row=0,column=1,sticky=Tk.W)
Tk.Button(master=analysisFrame,text='BoxH',command=boxhor,font=default_font).grid(row=0,column=2,sticky=Tk.W)
Tk.Button(master=analysisFrame,text='BoxV',command=boxver,font=default_font).grid(row=0,column=3,sticky=Tk.W)
Tk.Button(master=analysisFrame,text='BeamCenter',command=bcwindow,font=default_font).grid(row=0,column=4,sticky=Tk.W)
Tk.Button(master=analysisFrame,text='SelectSpots',command=selectspots,font=default_font).grid(row=0,column=5,sticky=Tk.W)

# --- Load / Quit Buttons ---
Tk.Button(master=root,text='Load',command=plot_updater,font=("Helvetica",20)).grid(row=figrowspan+1,column=2,rowspan=4,sticky=Tk.W,padx=10)
Tk.Button(master=root,text='Quit',command=_quit,font=("Helvetica",20)).grid(row=figrowspan+1,column=0,rowspan=4,sticky=Tk.W,padx=10)

# --- Mic File Frame ---
micFrame = Tk.LabelFrame(root, text="Mic File", font=default_font, padx=2, pady=2)
micFrame.grid(row=figrowspan+1,column=3,rowspan=4,sticky=Tk.W+Tk.N,padx=2,pady=2)

# Row 0: Load/Reload/Select
Tk.Button(master=micFrame,text='LoadMic',command=load_mic,font=default_font).grid(row=0,column=0,sticky=Tk.W)
Tk.Button(master=micFrame,text='ReloadMic',command=plotmic,font=default_font).grid(row=0,column=1,sticky=Tk.W)
Tk.Button(master=micFrame,text='SelectPoint',command=selectpoint,font=default_font).grid(row=0,column=2,sticky=Tk.W)

# Row 1-2: Radio buttons for coloring
Tk.Radiobutton(master=micFrame,text='Confidence',variable=colVar,value=10,font=default_font).grid(row=1,column=0,sticky=Tk.W)
Tk.Radiobutton(master=micFrame,text='GrainID',variable=colVar,value=0,font=default_font).grid(row=1,column=1,sticky=Tk.W)
Tk.Radiobutton(master=micFrame,text='PhaseNr',variable=colVar,value=11,font=default_font).grid(row=1,column=2,sticky=Tk.W)
Tk.Radiobutton(master=micFrame,text='Euler0',variable=colVar,value=7,font=default_font).grid(row=2,column=0,sticky=Tk.W)
Tk.Radiobutton(master=micFrame,text='Euler1',variable=colVar,value=8,font=default_font).grid(row=2,column=1,sticky=Tk.W)
Tk.Radiobutton(master=micFrame,text='Euler2',variable=colVar,value=9,font=default_font).grid(row=2,column=2,sticky=Tk.W)

# Row 3: New Derived Map Radio Buttons
Tk.Radiobutton(master=micFrame,text='KAM',variable=colVar,value=12,font=default_font).grid(row=3,column=0,sticky=Tk.W)
Tk.Radiobutton(master=micFrame,text='GROD',variable=colVar,value=13,font=default_font).grid(row=3,column=1,sticky=Tk.W)
Tk.Radiobutton(master=micFrame,text='Grain Map',variable=colVar,value=14,font=default_font).grid(row=3,column=2,sticky=Tk.W)

# Row 4: Confidence cutoffs
Tk.Label(master=micFrame,text='ConfCutoffs',font=default_font).grid(row=4,column=0,sticky=Tk.W)
Tk.Entry(master=micFrame,textvariable=cutconfidencevar,width=5,font=default_font).grid(row=4,column=1,sticky=Tk.W)
Tk.Entry(master=micFrame,textvariable=maxConfVar,width=5,font=default_font).grid(row=4,column=2,sticky=Tk.W)

# Row 5: Grain tools
Tk.Button(master=micFrame,text='LoadGrain',command=getgrain,font=default_font).grid(row=5,column=0,sticky=Tk.W)
Tk.Button(master=micFrame,text='MakeSpots',command=makespots,font=default_font).grid(row=5,column=1,sticky=Tk.W)
Tk.Button(master=micFrame,text='FindOrient',command=findOrientation,font=default_font).grid(row=5,column=2,sticky=Tk.W)

if __name__ == "__main__":
	root.bind('<Control-w>', lambda event: root.destroy())
	Tk.mainloop()
