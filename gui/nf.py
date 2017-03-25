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

def _quit():
	root.quit()
	root.destroy()

NrPixels = 2048
nrfilesperdistance = 720
ndistances = 3
background=100
fnstem = 'Au1'
folder = '/var/host/media/removable/UNTITLED/Au/'

figcolspan=10
figrowspan=10

def getfilenames(startFrameNr,distanceNr):
	medianfn = folder + fnstem + "_Median_Background_Distance_" + str(distanceNr) + ".bin"
	fnr = startFrameNr + distanceNr*nrfilesperdistance
	filefn = folder + fnstem + '_{:06d}'.format(fnr) + '.tif'
	return [filefn, medianfn]

def draw_plot(frameNr,distanceNr): # always the initial framenr and distance, will calculate the correct framenr automatically
	global initplot
	global framenr
	global imarr2
	if not initplot:
		lims = [a.get_xlim(), a.get_ylim()]
	a.clear()
	fns = getfilenames(frameNr,distanceNr)
	im = PIL.Image.open(fns[0])
	print "Read file " + fns[0]
	imarr = np.array(im,dtype=np.uint16)
	doMedian = var.get()
	if doMedian == 1:
		f = open(fns[1],'rb')
		print "Read file " + fns[1]
		median = np.fromfile(f,dtype=np.uint16,count=(NrPixels*NrPixels))
		median = np.reshape(median,(NrPixels,NrPixels))
		imarr2 = np.subtract(imarr.astype(int),median.astype(int))
		imarr2 = stats.threshold(imarr2,threshmin=background)
	else:
		imarr2 = imarr
	a.imshow(imarr2,cmap=plt.get_cmap('bone'),interpolation='nearest',clim=(0,int(vali.get())))
	if initplot:
		initplot = 0
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
	a.title.set_text("Image")
	canvas.show()
	canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)#pack(side=Tk.TOP,fill=Tk.BOTH,expand=1)

def plot_updater():
	global initplot
	global framenr
	global dist
	global oldVar
	global initvali
	global folder
	global fnstem
	global ndistances
	global NrPixels
	newVar = var.get()
	newframenr = int(r.get())
	newdist = int(r2.get())
	newvali = int(vali.get())
	newfolder = foldervar.get()
	newfnstem = fnstemvar.get()
	newndistances = int(ndistancesvar.get())
	newNrPixels = int(NrPixelsvar.get())
	if initplot == 1:
		draw_plot(framenr,newdist)
	if (newframenr != framenr) or (newdist != dist) or (newVar != oldVar) or (newvali != initvali) or (newfolder != folder) or (newfnstem != fnstem) or (newndistances != ndistances) or (newNrPixels != NrPixels):
		NrPixels = newNrPixels
		ndistances = newndistances
		fnstem = newfnstem
		folder = newfolder
		oldVar = newVar
		initvali = newvali
		framenr = newframenr
		dist = newdist
		draw_plot(framenr,newdist)

def onclick(event):
	global clickpos
	global horvert
	global imarr2
	global lb1
	global cid
	clickpos.append((event.xdata, event.ydata))
	if len(clickpos) == 2:
		canvas.mpl_disconnect(cid)
		cid = None
		lb1.grid_forget()#pack_forget()
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
			canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)#pack(side=Tk.TOP,fill=Tk.BOTH,expand=1)
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
			canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)#pack(side=Tk.TOP,fill=Tk.BOTH,expand=1)
	return

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
	horvert = 1 # 1 for horizontal, 2 for vertical
	clickpos = []
	cid = canvas.mpl_connect('button_press_event',onclick)
	lb1 = Tk.Label(master=root,text="Click two (almost) horizontal points")
	lb1.grid(row=figrowspan+3,column=3,sticky=Tk.W)#pack(side=Tk.BOTTOM)

def vertline():
	global cid
	global clickpos
	global horvert
	global lb1
	horvert = 2 # 1 for horizontal, 2 for vertical
	clickpos = []
	cid = canvas.mpl_connect('button_press_event',onclick)
	lb1 = Tk.Label(master=root,text="Click two (almost) vertical points")
	lb1.grid(row=figrowspan+3,column=3,sticky=Tk.W)#pack(side=Tk.BOTTOM)

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
	button7 = Tk.Button(master=root,text='Confirm Selection',command=confirmselectspot)
	button7.grid(row=figrowspan+3,column=3,sticky=Tk.W)#pack(side=Tk.LEFT)

def loadnewdistance():
	global topNewDistance
	topNewDistance.destroy()
	plot_updater()

def confirmselectspot():
	global spots
	global topNewDistance
	global vali
	x = NrPixels -ix
	y = NrPixels - iy
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
			#r1 = spots[i][2]
			#r2 = spots[j][2]
			x = distDiff * (j-i)
			xs[idx] = x*z1/(z2-z1) - (distDiff * i)#(r1*x/(r2-r1)) - (distDiff * i)
			ys[idx] = y1 + (y2-y1)*z1/(z1-z2)
			idx += 1
	topDistanceResult = Tk.Toplevel()
	topDistanceResult.title("Distance computation result")
	textdisplay = "Calculated distances are: " + str(xs) + " Calculated Ys are: " + str(ys)
	Tk.Label(topDistanceResult,text=textdisplay,font=("Helvetica",16)).grid(row=0)
	buttonclose = Tk.Button(master=topDistanceResult,text="Okay",command=topDistanceResult.destroy)
	buttonclose.grid(row=1)

def closeselectspotshelp():
	global topSelectSpotsWindow
	global cid2
	global button8
	topSelectSpotsWindow.destroy()
	cid2 = canvas.mpl_connect('button_press_event',getpos)
	button8 = Tk.Button(master=root,text='Compute Distances',command=computedistances)
	button8.grid(row=figrowspan+3,column=4,sticky=Tk.W)#pack(side=Tk.LEFT)

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
	Tk.Label(topSelectSpotsWindow,text="2. Make sure you have median correction enabled.").grid(row=2,columnspan=2)
	Tk.Label(topSelectSpotsWindow,text="3. Starting from the first distance, click on or close to a diffraction spot and then click Confirm Selection.").grid(row=3,columnspan=2)
	Tk.Label(topSelectSpotsWindow,text="4. Repeat this for each distance.").grid(row=4,columnspan=2)
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
	Tk.Label(top,text="Enter beam center values (pixels) REMEMBER to subtract NrPixels from raw coordinates").grid(row=0,columnspan=3)
	varsStore = []
	for dist in range(nRows):
		labeltext = "Distance " + str(dist)
		Tk.Label(top,text=labeltext).grid(row=dist+1)
		var1 = Tk.StringVar()
		var2 = Tk.StringVar()
		var1.set(str(0))
		var2.set(str(0))
		Tk.Entry(top,textvariable=var1).grid(row=dist+1,column=1)
		Tk.Entry(top,textvariable=var2).grid(row=dist+1,column=2)
		varsStore.append([var1,var2])
	Tk.Label(top,text="Difference in distances eg. 1000 microns").grid(row=ndistances+1)
	distDiffVar = Tk.StringVar()
	distDiffVar.set(str(0))
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
		Tk.Label(top2,text="All beam centers are 0. Do you want to continue?").grid(row=0)
		Tk.Button(top2,text="Go ahead with zeros.",command=top2destroyer).grid(row=1,column=0,sticky=Tk.W)
		Tk.Button(top2,text="Edit beam centers.",command=bcwindow).grid(row=1,column=1,sticky=Tk.W)
		selectingspots = 1
	else:
		selectspotsfcn()

root = Tk.Tk()
root.wm_title("NF display v0.1 Dt. 2017/03/25 hsharma@anl.gov")
figur = Figure(figsize=(20,7.5),dpi=100)
canvas = FigureCanvasTkAgg(figur,master=root)
a = figur.add_subplot(121,aspect='equal')
b = figur.add_subplot(122)
b.title.set_text("LineOuts")
imarr2 = None
var = Tk.IntVar()
initplot = 1
framenr = 20
dist = 0
horvert = 1
oldVar = 0
bcs = np.zeros((ndistances,2))
spots = np.zeros((ndistances,3))
pixelsize = 1.48
vali = Tk.StringVar() 
vali.set(str(100))
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
distDiffVar = None
topSelectSpotsWindow = None
topNewDistance = None
topDistanceResult = None
selectingspots = 0
clickpos = []
varsStore = []
canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)#pack(side=Tk.TOP,fill=Tk.BOTH)
toolbar_frame = Tk.Frame(root)
toolbar_frame.grid(row=figrowspan+5,column=0,columnspan=5,sticky=Tk.W)
toolbar = NavigationToolbar2TkAgg( canvas, toolbar_frame )
toolbar.update()

Tk.Label(master=root,text="Folder").grid(row=figrowspan+1,column=1,sticky=Tk.W)#pack(side=Tk.LEFT)
foldervar = Tk.StringVar() 
foldervar.set(folder)
e0 = Tk.Entry(master=root,textvariable=foldervar,width=35)
e0.grid(row=figrowspan+1,column=2,sticky=Tk.W)#pack(side=Tk.LEFT)

def folderselect():
	global folder
	global foldervar
	folder = tkFileDialog.askdirectory()
	foldervar.set(folder)

buttonfolder = Tk.Button(master=root,text="Select",command=folderselect)
buttonfolder.grid(row=figrowspan+1,column=1,sticky=Tk.E)

Tk.Label(master=root,text="FNStem").grid(row=figrowspan+1,column=2,sticky=Tk.E,padx=10)#pack(side=Tk.LEFT)
fnstemvar = Tk.StringVar() 
fnstemvar.set(fnstem)
efns = Tk.Entry(master=root,textvariable=fnstemvar,width=15)
efns.grid(row=figrowspan+1,column=3,sticky=Tk.W,padx=10)#pack(side=Tk.LEFT)

Tk.Label(master=root,text="nDistances").grid(row=figrowspan+1,column=3,sticky=Tk.E,padx=10)#pack(side=Tk.LEFT)
ndistancesvar = Tk.StringVar()
ndistancesvar.set(str(ndistances))
endistances = Tk.Entry(master=root,textvariable=ndistancesvar,width=3)
endistances.grid(row=figrowspan+1,column=4,sticky=Tk.W,padx=10)#pack(side=Tk.LEFT)

Tk.Label(master=root,text="NrPixels").grid(row=figrowspan+1,column=4,sticky=Tk.W,padx=50)#pack(side=Tk.LEFT)
NrPixelsvar = Tk.StringVar()
NrPixelsvar.set(str(NrPixels))
enrpixels = Tk.Entry(master=root,textvariable=NrPixelsvar,width=5)
enrpixels.grid(row=figrowspan+1,column=4,sticky=Tk.W,padx=120)#pack(side=Tk.LEFT)

Tk.Label(master=root,text="File Number").grid(row=figrowspan+1,column=4,sticky=Tk.E,padx=90)#pack(side=Tk.LEFT)
r = Tk.StringVar() 
r.set(str(framenr))
e1 = Tk.Entry(master=root,textvariable=r,width=5)
e1.grid(row=figrowspan+1,column=4,sticky=Tk.E,padx=30)#pack(side=Tk.LEFT)
e1.focus_set()

def incr_plotupdater():
	global r
	r.set(str(framenr+1))
	plot_updater()

def decr_plotupdater():
	global r
	r.set(str(framenr-1))
	plot_updater()

buttonIncr = Tk.Button(master=root,text='+',command=incr_plotupdater,font=("Helvetica",12))
buttonIncr.grid(row=figrowspan+1,column=4,sticky=Tk.E)#pack(side=Tk.LEFT)
buttonDecr = Tk.Button(master=root,text='-',command=decr_plotupdater,font=("Helvetica",12))
buttonDecr.grid(row=figrowspan+1,column=5,sticky=Tk.W,padx=5)#pack(side=Tk.LEFT)

Tk.Label(master=root,text="Distance Nr").grid(row=figrowspan+2,column=1,sticky=Tk.W)#pack(side=Tk.LEFT)
r2 = Tk.StringVar() 
r2.set(str(0))
e2 = Tk.Entry(master=root,textvariable=r2,width=3)
e2.grid(row=figrowspan+2,column=1,sticky=Tk.W,padx=90)#pack(side=Tk.LEFT)
e2.focus_set()

Tk.Label(master=root,text="Max Intensity for Color").grid(row=figrowspan+2,column=1,sticky=Tk.E,padx=10)#pack(side=Tk.LEFT)
e3 = Tk.Entry(master=root,textvariable=vali,width=4)
e3.grid(row=figrowspan+2,column=2,sticky=Tk.W)#pack(side=Tk.LEFT)
e3.focus_set()

c = Tk.Checkbutton(master=root,text="Subtract Median",variable=var)
c.grid(row=figrowspan+2,column=2,sticky=Tk.W,padx=70)#pack(side=Tk.LEFT)

button2 = Tk.Button(master=root,text='Load',command=plot_updater,font=("Helvetica",20))
button2.grid(row=figrowspan+1,column=5,rowspan=3,sticky=Tk.E,padx=10)#pack(side=Tk.LEFT)

Tk.Label(master=root,text="Pixel Size (microns)").grid(row=figrowspan+2,column=2,sticky=Tk.E)#pack(side=Tk.LEFT)
pxvar = Tk.StringVar()
pxvar.set(str(pixelsize))
epx = Tk.Entry(master=root,textvariable=pxvar,width=5)
epx.grid(row=figrowspan+2,column=3,sticky=Tk.W,padx=10)#pack(side=Tk.LEFT)

def paramfileselect():
	global paramfile
	global parameters
	paramfile = tkFileDialog.askopenfile()
	filecontents = paramfile.readlines()
	print filecontents

def getparams():
	# open new toplevel window, either read from file or take input, then close
	global topGetParams
	global paramfile
	topGetParams = Tk.Toplevel()
	topGetParams.title("Load Parameters into memory")
	Tk.Label(master=topGetParams,text="How do you want to load the parameters? File or input manually?",font=("Helvetica",12)).grid(row=1)
	bFile = Tk.Button(master=topGetParams,text="Choose File",command=paramfileselect)
	bFile.grid(row=2)
	x = 0
paramfile = ""
parameters = {}
buttongetparams = Tk.Button(master=root,text='LoadParameters',command=getparams)
buttongetparams.grid(row=figrowspan+2,column=3,sticky=Tk.E,padx=10)#pack(side=Tk.LEFT)

def saveparams():
	# open new toplevel window, either read from file or take input, then close
	x = 0
buttonsaveparams = Tk.Button(master=root,text='SaveParameters',command=saveparams)
buttonsaveparams.grid(row=figrowspan+2,column=4,sticky=Tk.W,padx=10)#pack(side=Tk.LEFT)

button3 = Tk.Button(master=root,text='LineOutHor',command=horline)
button3.grid(row=figrowspan+3,column=1,sticky=Tk.W,padx=40)#pack(side=Tk.LEFT)

button4 = Tk.Button(master=root,text='LineOutVert',command=vertline)
button4.grid(row=figrowspan+3,column=1,sticky=Tk.E,padx=40)#pack(side=Tk.LEFT)

button5 = Tk.Button(master=root,text='BeamCenter',command=bcwindow)
button5.grid(row=figrowspan+3,column=2,sticky=Tk.W,padx=10)#pack(side=Tk.LEFT)

button6 = Tk.Button(master=root,text='Select Spots',command=selectspots)
button6.grid(row=figrowspan+3,column=2,sticky=Tk.E,padx=90)#pack(side=Tk.LEFT)

button = Tk.Button(master=root,text='Quit',command=_quit,font=("Helvetica",20))
button.grid(row=figrowspan+1,column=0,rowspan=3,sticky=Tk.W,padx=10)#pack(side=Tk.LEFT)

Tk.mainloop()
