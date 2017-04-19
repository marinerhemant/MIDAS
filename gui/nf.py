#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

import PIL
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import sys
import Tkinter as Tk
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from scipy import stats
import tkFileDialog
import math
from subprocess import Popen, PIPE, STDOUT
from multiprocessing.dummy import Pool

# Helper constants
deg2rad = 0.0174532925199433
rad2deg = 57.2957795130823

## Some initialization values
NrPixels = 2048
nrfilesperdistance = 720
padding = 6
ndistances = 3
background=0
fnstem = 'Au1'
folder = '/var/host/media/removable/UNTITLED/Au/'
figcolspan=10
figrowspan=10

def _quit():
	root.quit()
	root.destroy()

def getfilenames():
	medianfn = folder + '/' + fnstem + "_Median_Background_Distance_" + str(dist) + ".bin"
	fnr = startframenr + framenr + dist*nrfilesperdistance
	filefn = folder + '/' + fnstem + '_' + str(fnr).zfill(padding) + '.tif' # '_{:06d}'.format(fnr) + '.tif'
	return [filefn, medianfn]

def draw_plot(): # always the initial framenr and distance, will calculate the correct framenr automatically
	global initplot
	global imarr2
	if not initplot:
		lims = [a.get_xlim(), a.get_ylim()]
	a.clear()
	if maxoverframes.get() == 0:
		fns = getfilenames()
		im = PIL.Image.open(fns[0])
		print "Read file " + fns[0]
		imarr = np.array(im,dtype=np.uint16)
		doMedian = var.get()
		fnprint = fns[0].replace(folder,'')
		if doMedian == 1:
			f = open(fns[1],'rb')
			print "Read file " + fns[1]
			median = np.fromfile(f,dtype=np.uint16,count=(NrPixels*NrPixels))
			f.close()
			median = np.reshape(median,(NrPixels,NrPixels))
			imarr2 = np.subtract(imarr.astype(int),median.astype(int))
			imarr2 = stats.threshold(imarr2,threshmin=background)
		else:
			imarr2 = imarr
	else:
		if var.get() == 1:
			fnthis = folder + '/' + fnstem + '_MaximumIntensityMedianCorrected_Distance_' + str(dist) + '.bin'
		else:
			fnthis = folder + '/' + fnstem + '_MaximumIntensity_Distance_' + str(dist) + '.bin'
		f = open(fnthis,'rb')
		print 'Read file ' + fnthis
		fnprint = fnthis.replace(folder,'')
		imarr = np.fromfile(f,dtype=np.uint16,count=(NrPixels*NrPixels))
		f.close()
		imarr2 = np.reshape(imarr,(NrPixels,NrPixels))
		imarr2 = stats.threshold(imarr2,threshmin=background)
	imarr2 = np.flipud(np.fliplr(imarr2))
	if dolog.get() == 0:
		a.imshow(imarr2,cmap=plt.get_cmap('bone'),interpolation='nearest',clim=(float(minThreshvar.get()),float(vali.get())))
	else:
		minC = float(minThreshvar.get())
		maxC = float(vali.get())
		if minC == 0:
			minC = 1
		if maxC == 0:
			maxC = 1
		imarr2plt = np.copy(imarr2)
		imarr2plt [imarr2plt == 0] = 1
		a.imshow(np.log(imarr2plt),cmap=plt.get_cmap('bone'),interpolation='nearest',clim=(np.log(minC),np.log(maxC)))
	if initplot:
		initplot = 0
		a.invert_xaxis()
		a.invert_yaxis()
	else:
		a.set_xlim([lims[0][0],lims[0][1]])
		a.set_ylim([lims[1][0],lims[1][1]])
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
	canvas.show()
	canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)#pack(side=Tk.TOP,fill=Tk.BOTH,expand=1)

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
	global NrPixels
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
	newNrPixels = int(NrPixelsvar.get())
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
		(newNrPixels != NrPixels) or 
		(newstartframenr != startframenr) or
		(newlsd !=lsd)):
		oldmaxoverframes = newmaxoverframes
		logscaler = newlogscaler
		nrfilesperdistance = newnrfilesperdistance
		minThresh = newminThresh
		lsd = newlsd
		NrPixels = newNrPixels
		ndistances = newndistances
		fnstem = newfnstem
		folder = newfolder
		oldVar = newVar
		initvali = newvali
		framenr = newframenr
		startframenr = newstartframenr
		dist = newdist
		draw_plot()
		plotb()

def plotb():
	global horvert
	global clickpos
	global cb
	global initplotb
	if cb is not None:
		cb.remove()
		cb = None
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
		canvas.show()
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
		canvas.show()
		canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)
	b.set_aspect('auto')
	
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
	if cb is not None:
		cb.remove()
		cb = None
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
				print imarr2[yval,xval]
				ztr += imarr2[yval,xval]
			zs.append(ztr)
		b.plot(np.array(xvals),zs)
		b.title.set_text('BoxOutHor')
	elif horvert == 2:
		for yval in yvals:
			ztr = 0
			for xval in xvals:
				print imarr2[yval,xval]
				ztr += imarr2[yval,xval]
			zs.append(ztr)
		b.plot(np.array(yvals),zs)
		b.title.set_text('BoxOutVer')
	canvas.show()
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
	global button7
	global  button8
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
	horvert = 1 # 1 for horizontal, 2 for vertical
	clickpos = []
	cid = canvas.mpl_connect('button_press_event',onclickbox)
	lb1 = Tk.Label(master=thirdRowFrame,text="Click two edges of the box")
	lb1.grid(row=1,column=nrthird+1,columnspan=3,sticky=Tk.W)

def boxver():
	global cid
	global clickpos
	global horvert
	global lb1
	global button7
	global  button8
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
	horvert = 2 # 1 for horizontal, 2 for vertical
	clickpos = []
	cid = canvas.mpl_connect('button_press_event',onclickbox)
	lb1 = Tk.Label(master=thirdRowFrame,text="Click two edges of the box")
	lb1.grid(row=1,column=nrthird+1,columnspan=3,sticky=Tk.W)

def horline():
	global cid
	global clickpos
	global horvert
	global lb1
	global button7
	global  button8
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
	horvert = 1 # 1 for horizontal, 2 for vertical
	clickpos = []
	cid = canvas.mpl_connect('button_press_event',onclick)
	lb1 = Tk.Label(master=thirdRowFrame,text="Click two (almost) horizontal points")
	lb1.grid(row=1,column=nrthird+1,columnspan=3,sticky=Tk.W)

def vertline():
	global cid
	global clickpos
	global horvert
	global lb1
	global button7
	global  button8
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
	horvert = 2 # 1 for horizontal, 2 for vertical
	clickpos = []
	cid = canvas.mpl_connect('button_press_event',onclick)
	lb1 = Tk.Label(master=thirdRowFrame,text="Click two (almost) vertical points")
	lb1.grid(row=1,column=nrthird+1,columnspan=3,sticky=Tk.W)#pack(side=Tk.BOTTOM)

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
	# We need to get the ix and iy for center of mass of the spot. For now I will trust what goes on.
	global ix
	global iy
	global cid2
	global button7
	ix,iy = event.xdata,event.ydata
	button7 = Tk.Button(master=thirdRowFrame,text='Confirm Selection',command=confirmselectspot)
	button7.grid(row=1,column=nrthird+1,sticky=Tk.E)#pack(side=Tk.LEFT)

def loadnewdistance():
	global topNewDistance
	topNewDistance.destroy()
	plot_updater()

def confirmselectspot():
	global spots
	global topNewDistance
	global vali
	x = ix # = NrPixels -ix
	y = iy # = NrPixels - iy
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
	edist.grid(row=1,column=1)#pack(side=Tk.LEFT)
	edist.focus_set()
	buttondist = Tk.Button(master=topNewDistance,text='Load',command=loadnewdistance)
	buttondist.grid(row=2,column=1)#pack(side=Tk.LEFT)
	buttonkill = Tk.Button(master=topNewDistance,text='Finished',command=loadnewdistance)
	buttonkill.grid(row=3,column=1)#pack(side=Tk.LEFT)

def computedistances():
	global topDistanceResult
	global lsd
	nsols = ndistances*(ndistances-1)/2
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
	button8 = Tk.Button(master=thirdRowFrame,text='Compute Distances',command=computedistances)
	button8.grid(row=1,column=nrthird+2,sticky=Tk.W)#pack(side=Tk.LEFT)

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
	button5.grid(row=1,column=1,sticky=Tk.W)#pack(side=Tk.LEFT)
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
	if cid is not None:
		canvas.mpl_disconnect(cid)
		lb1.grid_forget()
		lb1 = None
		cid = None
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

def getgrain():
	# open new toplevel window, either read from file or take input, then close
	global topGetGrain
	global omvar,posvar,latCvar,wlvar,startomevar,omestepvar,sgvar,maxringradvar
	topGetGrain = Tk.Toplevel()
	topGetGrain.title("Load Grain into memory")
	Tk.Label(master=topGetGrain,text="Please enter the orientation matrix, position, lattice parameter, wavelength, startomega, omegastep, spacegroup, maxringrad",font=("Helvetica",12)).grid(row=1,columnspan=10)
	Tk.Label(master=topGetGrain,text="Orientation Matrix: ").grid(row=2,column=1)
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

def plot_update_spot():
	global r, spotnrvar, spotnr
	thisome = float(simulatedspots[spotnr-1].split(' ')[2])
	rad = float(simulatedspots[spotnr-1].split(' ')[0])
	eta = float(simulatedspots[spotnr-1].split(' ')[1])
	ys,zs = YZ4mREta(rad,eta)
	filenrToRead = startframenr + int((float(thisome)-float(startome))/omestep)
	r.set(str(filenrToRead))
	spotnrvar.set(str(spotnr))
	ya = pos[0]*math.sin(thisome) + pos[1]*math.cos(thisome)
	xa = -pos[1]*math.sin(thisome) + pos[0]*math.cos(thisome)
	yn = (ya + ys*(1-(xa/lsd)))/pixelsize - bcs[dist][0] #NrPixels - (ya + ys*(1-(xa/lsd)))/pixelsize - bcs[dist][0]
	zn = (zs*(1-(xa/lsd)))/pixelsize - bcs[dist][1] #NrPixels - (zs*(1-(xa/lsd)))/pixelsize - bcs[dist][1]
	print [ys, ya, yn, zs, zn, rad, eta,thisome,filenrToRead]
	while ((abs(eta) > 90) or (yn > NrPixels) or (zn > NrPixels)):
		spotnr += 1
		thisome = float(simulatedspots[spotnr-1].split(' ')[2])
		rad = float(simulatedspots[spotnr-1].split(' ')[0])
		eta = float(simulatedspots[spotnr-1].split(' ')[1])
		filenrToRead = startframenr + int((float(thisome)-float(startome))/omestep)
		r.set(str(filenrToRead))
		spotnrvar.set(str(spotnr))
		# calculate spot position and make a blip on the detector
		ys,zs = YZ4mREta(rad,eta)
		ya = pos[0]*math.sin(thisome) + pos[1]*math.cos(thisome)
		xa = -pos[1]*math.sin(thisome) + pos[0]*math.cos(thisome)
		yn = (ya + ys*(1-(xa/lsd)))/pixelsize - bcs[dist][0] #NrPixels - (ya + ys*(1-(xa/lsd)))/pixelsize - bcs[dist][0]
		zn = (zs*(1-(xa/lsd)))/pixelsize - bcs[dist][1] #NrPixels - (zs*(1-(xa/lsd)))/pixelsize - bcs[dist][1]
		print [ys, ya, yn, zs, zn, rad, eta,thisome,filenrToRead]
	plot_updater()
	a.scatter(yn,zn,s=5,color='red')
	canvas.show()
	canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)#pack(side=Tk.TOP,fill=Tk.BOTH,expand=1)

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
	pfname = "ps.txt"
	f = open(pfname,'w')
	f.write("SpaceGroup %d\n"%(sg))
	f.write("Wavelength %lf\n"%(wl))
	f.write("Lsd %lf\n"%(lsd))
	f.write("MaxRingRad %lf\n"%(maxringrad))
	f.write("LatticeConstant %lf %lf %lf %lf %lf %lf\n"%(latC[0],latC[1],latC[2],latC[3],latC[4],latC[5]))
	f.close()
	hklpath = '~/opt/MIDAS/NF_HEDM/bin/GetHKLList '
	os.system(hklpath+pfname)
	genseedorpath = '~/opt/MIDAS/NF_HEDM/bin/GenSeedOrientationsFF2NFHEDM '
	diffrspotspath = '~/opt/MIDAS/NF_HEDM/bin/SimulateDiffractionSpots '
	orinfn = 'orin.txt'
	oroutfn = 'orout.txt'
	instr = "120 %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n"%(om[0],om[1],om[2],om[3],om[4],
			om[5],om[6],om[7],om[8],pos[0],pos[1],pos[2],latC[0],latC[1],latC[2],latC[3],latC[4],latC[5])
	f = open(orinfn,'w')
	f.write(instr)
	f.close()
	os.system(genseedorpath+orinfn+' '+oroutfn)
	os.system(diffrspotspath+str(lsd)+' '+oroutfn)
	spotsfn = 'SimulatedDiffractionSpots.txt'
	simulatedspots = open(spotsfn,'r').readlines()
	Tk.Label(master=root,text="SpotNumber").grid(row=figrowspan+3,column=3,sticky=Tk.E,padx=50)
	Tk.Entry(root,textvariable=spotnrvar,width=3).grid(row=figrowspan+3,column=3,sticky=Tk.E)
	plot_update_spot()
	# put a frame with +,- and load
	Tk.Button(master=root,text="+",command=incr_spotnr,font=("Helvetica",12)).grid(row=figrowspan+3,column=4,sticky=Tk.W)
	Tk.Button(master=root,text="Load",command=update_spotnr,font=("Helvetica",12)).grid(row=figrowspan+3,column=4,sticky=Tk.W,padx=70)

def median():
	cmdout = []
	for thisdist in range(ndistances):
		pfname = folder + 'ps.txt' + str(thisdist)
		f = open(pfname,'w')
		f.write('extReduced bin\n')
		f.write('extOrig tif\n')
		f.write('WFImages 0\n')
		f.write('OrigFileName '+fnstem+'\n')
		tempnr = startframenr + thisdist*(nrfilesperdistance - int(nrfilesmedianvar.get()))
		f.write('NrFilesPerDistance '+nrfilesmedianvar.get()+'\n')
		f.write('NrPixels '+str(NrPixels)+'\n')
		f.write('DataDirectory '+folder+'\n')
		f.write('RawStartNr '+str(tempnr)+'\n')
		f.close()
		cmdout.append('~/opt/MIDAS/NF_HEDM/Cluster/MedianImageParallel.sh '+pfname+' '+str(thisdist+1))
	processes = [Popen(cmdname,shell=True,
				stdin=PIPE, stdout=PIPE, stderr=STDOUT,close_fds=True) for cmdname in cmdout]
	def get_lines(process):
		return process.communicate()[0].splitlines()
	outputs = Pool(len(processes)).map(get_lines,processes)
	print 'Calculated median for all distances.'

def micfileselect():
	global micfile
	micfile = tkFileDialog.askopenfilename()

def plotmic():
	global initplotb
	global colVar
	global cb
	if not initplotb:
		lims = [b.get_xlim(), b.get_ylim()]
		cb.remove()
		cb = None
	b.clear()
	col = colVar.get()
	micfiledatacut = np.copy(micfiledata)
	micfiledatacut = micfiledatacut[ micfiledatacut[:,10] > float(cutconfidencevar.get()) , :]
	print float(cutconfidencevar.get())
	if cb is not None:
		cb.remove()
	sc = b.scatter(micfiledatacut[:,3],micfiledatacut[:,4],c=micfiledatacut[:,col],lw=0)
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
	cb = figur.colorbar(sc,ax=b)
	b.set_aspect('equal')
	figur.tight_layout()
	canvas.show()
	canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)

def load_mic():
	global micfiledata, initplotb
	initplotb = 1
	micfileselect()
	f = open(micfile,'r')
	micfiledata = np.genfromtxt(f,skip_header=4)
	f.close()
	plotmic()

# Global constants initialization
imarr2 = None
initplot = 1
framenr = 0
startframenr = 0
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
root.wm_title("NF display v0.1 Dt. 2017/03/25 hsharma@anl.gov")
figur = Figure(figsize=(20,8.5),dpi=100)
canvas = FigureCanvasTkAgg(figur,master=root)
a = figur.add_subplot(121,aspect='equal')
b = figur.add_subplot(122)
b.title.set_text("LineOuts/MicFile")
a.title.set_text("Image")
canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)#pack(side=Tk.TOP,fill=Tk.BOTH)
toolbar_frame = Tk.Frame(root)
toolbar_frame.grid(row=figrowspan+4,column=0,columnspan=5,sticky=Tk.W)
toolbar = NavigationToolbar2TkAgg( canvas, toolbar_frame )
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
initplotb = 1
colVar = Tk.IntVar()
colVar.set(10)
micfiledata = None
dolog = Tk.IntVar()

firstRowFrame = Tk.Frame(root)
firstRowFrame.grid(row=figrowspan+1,column=1,sticky=Tk.W)

Tk.Label(master=firstRowFrame,text="Folder").grid(row=1,column=1,sticky=Tk.W)#pack(side=Tk.LEFT)
foldervar = Tk.StringVar()
foldervar.set(folder)
e0 = Tk.Entry(master=firstRowFrame,textvariable=foldervar,width=50)
e0.grid(row=1,column=2,sticky=Tk.W)#pack(side=Tk.LEFT)

buttonfolder = Tk.Button(master=firstRowFrame,text="Select",command=folderselect)
buttonfolder.grid(row=1,column=3,sticky=Tk.W)

Tk.Label(master=firstRowFrame,text="FNStem").grid(row=1,column=4,sticky=Tk.W)#pack(side=Tk.LEFT)
fnstemvar = Tk.StringVar() 
fnstemvar.set(fnstem)
efns = Tk.Entry(master=firstRowFrame,textvariable=fnstemvar,width=15)
efns.grid(row=1,column=5,sticky=Tk.W)#pack(side=Tk.LEFT)

Tk.Label(master=firstRowFrame,text="nDistances").grid(row=1,column=6,sticky=Tk.W)#pack(side=Tk.LEFT)
ndistancesvar = Tk.StringVar()
ndistancesvar.set(str(ndistances))
endistances = Tk.Entry(master=firstRowFrame,textvariable=ndistancesvar,width=3)
endistances.grid(row=1,column=7,sticky=Tk.W)#pack(side=Tk.LEFT)

Tk.Label(master=firstRowFrame,text="NrPixels").grid(row=1,column=8,sticky=Tk.W)#pack(side=Tk.LEFT)
NrPixelsvar = Tk.StringVar()
NrPixelsvar.set(str(NrPixels))
enrpixels = Tk.Entry(master=firstRowFrame,textvariable=NrPixelsvar,width=5)
enrpixels.grid(row=1,column=9,sticky=Tk.W)#pack(side=Tk.LEFT)

Tk.Label(master=firstRowFrame,text="StartFileNumberFirstLayer").grid(row=1,column=10,sticky=Tk.W)
startframenrvar = Tk.StringVar()
startframenrvar.set(str(startframenr))
sfnrfiles = Tk.Entry(master=firstRowFrame,textvariable=startframenrvar,width=6)
sfnrfiles.grid(row=1,column=11,sticky=Tk.W)

Tk.Label(master=firstRowFrame,text="FrameNumber").grid(row=1,column=12,sticky=Tk.W)#pack(side=Tk.LEFT)
r = Tk.StringVar() 
r.set(str(framenr))
e1 = Tk.Entry(master=firstRowFrame,textvariable=r,width=5)
e1.grid(row=1,column=13,sticky=Tk.W)
e1.focus_set()

buttonIncr = Tk.Button(master=firstRowFrame,text='+',command=incr_plotupdater,font=("Helvetica",12))
buttonIncr.grid(row=1,column=14,sticky=Tk.W)
buttonDecr = Tk.Button(master=firstRowFrame,text='-',command=decr_plotupdater,font=("Helvetica",12))
buttonDecr.grid(row=1,column=15,sticky=Tk.W)

secondRowFrame = Tk.Frame(root)
secondRowFrame.grid(row=figrowspan+2,column=1,sticky=Tk.W)

Tk.Label(master=secondRowFrame,text="DistanceNr").grid(row=1,column=1,sticky=Tk.W)
r2 = Tk.StringVar() 
r2.set(str(0))
e2 = Tk.Entry(master=secondRowFrame,textvariable=r2,width=3)
e2.grid(row=1,column=2,sticky=Tk.W)
e2.focus_set()

minThresh = 0
Tk.Label(master=secondRowFrame,text="MinThresh(cnts)").grid(row=1,column=3,sticky=Tk.W)
minThreshvar = Tk.StringVar()
minThreshvar.set(str(minThresh))
emt = Tk.Entry(master=secondRowFrame,textvariable=minThreshvar,width=4)
emt.grid(row=1,column=4,sticky=Tk.W)

Tk.Label(master=secondRowFrame,text="MaxThresh").grid(row=1,column=5,sticky=Tk.W)
e3 = Tk.Entry(master=secondRowFrame,textvariable=vali,width=4)
e3.grid(row=1,column=6,sticky=Tk.W)
e3.focus_set()

Tk.Checkbutton(master=secondRowFrame,text="LogScale",variable=dolog).grid(row=1,column=7,sticky=Tk.W)

Tk.Label(master=secondRowFrame,text="PixelSize(um)").grid(row=1,column=8,sticky=Tk.W)
pxvar = Tk.StringVar()
pxvar.set(str(pixelsize))
epx = Tk.Entry(master=secondRowFrame,textvariable=pxvar,width=5)
epx.grid(row=1,column=9,sticky=Tk.W)

Tk.Label(master=secondRowFrame,text="FirstLsd(um)").grid(row=1,column=10,sticky=Tk.W)
lsdvar = Tk.StringVar()
lsdvar.set(str(lsd))
elsd = Tk.Entry(master=secondRowFrame,textvariable=lsdvar,width=10)
elsd.grid(row=1,column=11,sticky=Tk.W)

Tk.Label(master=secondRowFrame,text="nFiles/Distance").grid(row=1,column=12,sticky=Tk.W)
nrfilesvar = Tk.StringVar()
nrfilesvar.set(str(nrfilesperdistance))
enrfiles = Tk.Entry(master=secondRowFrame,textvariable=nrfilesvar,width=4)
enrfiles.grid(row=1,column=13,sticky=Tk.W)

buttonmedian = Tk.Button(master=secondRowFrame,text='CalcMedian',command=median)
buttonmedian.grid(row=1,column=14,sticky=Tk.W)

Tk.Label(master=secondRowFrame,text="nFilesMedianCalc").grid(row=1,column=15,sticky=Tk.W)
nrfilesmedianvar = Tk.StringVar()
nrfilesmedianvar.set(str(nrfilesperdistance))
enrfilesmedian = Tk.Entry(master=secondRowFrame,textvariable=nrfilesmedianvar,width=4)
enrfilesmedian.grid(row=1,column=16,sticky=Tk.W)

oldmaxoverframes = 0
maxoverframes = Tk.IntVar()

chkMaxOverFrames = Tk.Checkbutton(master=secondRowFrame,text="LoadMaxOverFrames",variable=maxoverframes)
chkMaxOverFrames.grid(row=1,column=17,sticky=Tk.W)

c = Tk.Checkbutton(master=secondRowFrame,text="Subtract Median",variable=var)
c.grid(row=1,column=18,sticky=Tk.W)

thirdRowFrame = Tk.Frame(root)
thirdRowFrame.grid(row=figrowspan+3,column=1,sticky=Tk.W)
nrthird = 8

button3 = Tk.Button(master=thirdRowFrame,text='LineOutHor',command=horline)
button3.grid(row=1,column=1,sticky=Tk.W)

button4 = Tk.Button(master=thirdRowFrame,text='LineOutVert',command=vertline)
button4.grid(row=1,column=2,sticky=Tk.W)

Tk.Button(master=thirdRowFrame,text='BoxOutHor',command=boxhor).grid(row=1,column=3,sticky=Tk.W)
Tk.Button(master=thirdRowFrame,text='BoxOutVer',command=boxver).grid(row=1,column=4,sticky=Tk.W)

button5 = Tk.Button(master=thirdRowFrame,text='BeamCenter',command=bcwindow)
button5.grid(row=1,column=5,sticky=Tk.W)

button6 = Tk.Button(master=thirdRowFrame,text='Select Spots',command=selectspots)
button6.grid(row=1,column=6,sticky=Tk.W)

buttongetgrain = Tk.Button(master=thirdRowFrame,text='LoadGrain',command=getgrain)
buttongetgrain.grid(row=1,column=7,sticky=Tk.W)

spotnrvar = Tk.StringVar()
bMakeSpots = Tk.Button(master=thirdRowFrame,text="MakeSpots",command=makespots)
bMakeSpots.grid(row=1,column=8,sticky=Tk.W)

button2 = Tk.Button(master=root,text='Load',command=plot_updater,font=("Helvetica",20))
button2.grid(row=figrowspan+1,column=2,rowspan=3,sticky=Tk.W)

loadmicframe = Tk.Frame(root)
loadmicframe.grid(row=figrowspan+1,column=3,sticky=Tk.W)

buttonLoadMicFile = Tk.Button(master=loadmicframe,text='LoadMicrostructure',command=load_mic,font=("Helvetica",11))
buttonLoadMicFile.grid(row=1,column=1,sticky=Tk.W)

buttonReplot = Tk.Button(master=loadmicframe,text='ReloadMic',command=plotmic,font=("Helvetica",11))
buttonReplot.grid(row=1,column=2,sticky=Tk.W)

radioframe = Tk.Frame(root)
radioframe.grid(row=figrowspan+2,column=3,sticky=Tk.W)

Tk.Radiobutton(master=radioframe,text='Confidence',variable=colVar,value=10).grid(row=1,column=1,sticky=Tk.W)
Tk.Radiobutton(master=radioframe,text='Euler0',variable=colVar,value=7).grid(row=1,column=2,sticky=Tk.W)
Tk.Radiobutton(master=radioframe,text='Euler1',variable=colVar,value=8).grid(row=1,column=3,sticky=Tk.W)
Tk.Radiobutton(master=radioframe,text='Euler2',variable=colVar,value=9).grid(row=1,column=4,sticky=Tk.W)

micframethirdrow = Tk.Frame(root)
micframethirdrow.grid(row=figrowspan+3,column=3,sticky=Tk.W)

Tk.Label(master=micframethirdrow,text='CutoffConfidence').grid(row=1,column=1,sticky=Tk.W)
Tk.Entry(master=micframethirdrow,textvariable=cutconfidencevar,width=4).grid(row=1,column=2,sticky=Tk.W)

def selectpoint():
	if micfiledata == None:
		load_mic()
	

Tk.Button(master=micframethirdrow,text='SelectPoint',command=selectpoint).grid(row=1,column=3)

button = Tk.Button(master=root,text='Quit',command=_quit,font=("Helvetica",20))
button.grid(row=figrowspan+1,column=0,rowspan=3,sticky=Tk.W)

Tk.mainloop()
