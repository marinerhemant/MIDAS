import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import sys
import tkinter as Tk
from tkinter import ttk
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import tkinter.filedialog as tkFileDialog
import math
import scipy
import scipy.ndimage as ndimage

def _quit():
    root.quit()
    root.destroy()

def getfn_1(fstem):
 return fstem+'/'+samplename+'method1_FileNrs_'+str(StartNr)+'_'+str(EndNr)+'_'+str(Resulttype_m1)+'_Rad_'+str(Rad_m1)+'_pm_'+str(Rwidth)+'_Eta_'+str(Eta_m1)+'_pm_'+str(Ewidth)+'_size_'+str(nframe)+'x'+str(nfile)+'_filter_'+str(filter)+'_float32.bin'

def getfn_2(fstem):
 return fstem+'/'+samplename+'method2_FileNrs_'+str(StartNr)+'_'+str(EndNr)+'_'+str(Resulttype_m2)+'_Rad_'+str(Rad_m2)+'_pm_'+str(Rwidth)+'_Eta_'+str(Eta_m2)+'_pm_'+str(Ewidth)+'_size_'+str(nframe)+'x'+str(nfile)+'_float32.bin'

def getData(fn,reconSize,Methodnum):
    if Methodnum ==1:
     fn=getfn_1(fileStem_m1)
    else:
     fn=getfn_2(fileStem_m2)
    print("Reading file: " + fn)
    f = open(fn,'rb')
    data = np.fromfile(f,dtype=np.uint16,count=(reconSize*reconSize))
    f.close()
    data = np.reshape(data,(reconSize,reconSize))
    data = data.astype(float)
    return data

def loadplot_m1():
    global refreshPlot
    global data_m1
    threshold = float(thresholdvar_m1.get())
    upperthreshold = float(maxthresholdvar_m1.get())
    data_m1 = getData(fileStem_m1,NrPixels_m1,1)
    refreshPlot = 1
    m1.imshow(data_m1,cmap=plt.get_cmap('bone'),interpolation='nearest',clim=(threshold,upperthreshold))
    m1.set_xlim([lims[0][0],lims[0][1]])
    m1.set_ylim([lims[1][0],lims[1][1]])
    m1.title.set_text("Method 1")
    canvas.draw()
    canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)

def loadplot_m2():
    global refreshPlot
    global data_m2
    global NrPixels_m2
    threshold = float(thresholdvar_m2.get())
    upperthreshold = float(maxthresholdvar_m2.get())
    NrPixels_m2 = int(NrPixelsvar_m2.get())
    data_m2 = getData(fileStem_m2,NrPixels_m2,2)
    refreshPlot = 1
    m2.imshow(data_m2,cmap=plt.get_cmap('bone'),interpolation='nearest',clim=(threshold,upperthreshold))
    m2.set_xlim([lims[0][0],lims[0][1]])
    m2.set_ylim([lims[1][0],lims[1][1]])
    m2.title.set_text("Method 2")
    canvas.draw()
    canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)

def readParams():
    global paramFN
    paramFN = paramfilevar.get()
    global NrPixels_m1,startNr,endNr,samplename,nframe,Rad,Rwidth,Eta,etaWidth,nfile
    paramContents = open(paramFN,'r').readlines()
    for line in paramContents:
        if line == '\n':
            continue
        if line[0] == '#':
            continue
        if 'detYdim' == line.split()[0]:
            NrPixels_m1 = int(line.split()[1])
        if 'startNr' == line.split()[0]:
            StartNr = int(line.split()[1])
        if 'endNr' == line.split()[0]:
            EndNr = int(line.split()[1])
        if 'fstem' == line.split()[0]:
            samplename = str((line.split()[1]).split('/')[-1])
        if 'nFrames' == line.split()[0]:
            nframe = int(line.split()[1])
        if 'rads'==line.split()[0]:
         lineRadarray=line.split()
         for radN in range(len(lineRadarray)-1):
          Rad[radN] =lineRadarray[1+radN]
        if 'rWidth'==line.split()[0]:
            Rwidth=int(line.split()[1])
        if 'eta'==line.split()[0]:
         lineEtaarray=line.split()
         for etaN in range(len(lineEtaarray)-1):
          Eta[etaN] =lineEtaarray[1+etaN]
        if 'etaWidth'==line.split()[0]:
            etaWidth=int(line.split()[1])
        if 'filter'==line.split()[0]:
            filterNr=int(line.split()[1])
    nfile=EndNr-StartNr+1
    NrPixels_m2=nfile
            
def paramfileselect():
    global paramFN
    global paramfilevar
    paramFN = tkFileDialog.askopenfilename()
    paramfilevar.set(paramFN)

def folderSelector_m1():
    global fileStem_m1
    fileStem_m1 = tkFileDialog.askdirectory()
 
def folderSelector_m2():
    global fileStem_m2
    fileStem_m2 = tkFileDialog.askdirectory()
       
def replot_m1():
    global data_m1, refreshPlot
    threshold = float(thresholdvar_m1.get())
    upperthreshold = float(maxthresholdvar_m1.get())
    if data_m1 is not None:
     lims = [data_m1.get_xlim(), data_m1.get_ylim()]
     m1.imshow(data_m1,cmap=plt.get_cmap('bone'),interpolation='nearest',clim=(threshold,upperthreshold))
     m1.title.set_text("Method 1")
    refreshPlot = 1
    canvas.draw()
    canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)

def replot_m2():
    global data_m2, refreshPlot
    threshold = float(thresholdvar_m2.get())
    upperthreshold = float(maxthresholdvar_m2.get())
    if data_m1 is not None:
     lims = [data_m2.get_xlim(), data_m2.get_ylim()]
     m2.imshow(data_m2,cmap=plt.get_cmap('bone'),interpolation='nearest',clim=(threshold,upperthreshold))
     m2.title.set_text("Method 2")
    refreshPlot = 1
    canvas.draw()
    canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)
    
def func1(event):
    Resulttype_m1=cbox_rs_1.get()

def func2(event):
    Rad_m1=cbox_rs_1.get()

def func3(event):
    Eta_m1=cbox_rs_1.get()

def func4(event):
    Resulttype_m2=cbox_rs_2.get()

def func5(event):
    Rad_m2=cbox_rs_2.get()

def func6(event):
    Eta_m2=cbox_rs_2.get()

root = Tk.Tk()
root.wm_title(" DT Reconstruction display ")
figur = Figure(figsize=(19.5,8.5),dpi=100)
canvas = FigureCanvasTkAgg(figur,master=root)
m1 = figur.add_subplot(121,aspect='equal')
m2 = figur.add_subplot(122,aspect='equal')
m1.title.set_text("Method 1 Display")
m2.title.set_text("Method 2 Display")
figrowspan = 10
figcolspan = 10
paramFN = 'filepara.txt'
data_m1 = None
data_m2 = None
NrPixels_m1 = 128
NrPixels_m2 = 55
Rad= None
Eta=None
paramfilevar = Tk.StringVar()
paramfilevar.set(paramFN)
thresholdvar_m1 = Tk.StringVar()
threshold_m1 = 0
thresholdvar_m1.set(str(threshold_m1))
maxthresholdvar_m1 = Tk.StringVar()
maxthresholdvar_m1.set(str(50))

thresholdvar_m2 = Tk.StringVar()
threshold_m2 = 0
thresholdvar_m2.set(str(threshold_m2))
maxthresholdvar_m2 = Tk.StringVar()
maxthresholdvar_m2.set(str(50))

Resultvalue=["RMEAN", "MixFactor", "SigmaG", "SigmaL", "MaxInt", "BGFit",
           "BGSimple", "MeanError", "FitIntegratedIntensity", "TotalIntensity", "TotalIntensityBackgroundCorr", "MaxIntensityObs"]
refreshPlot = 0

canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)
toolbar_frame = Tk.Frame(root)
toolbar_frame.grid(row=figrowspan+4,column=0,columnspan=10,sticky=Tk.W)
toolbar = NavigationToolbar2Tk( canvas, toolbar_frame )
toolbar.update()

Tk.Button(master=root,text='Quit',command=_quit,font=("Helvetica",20)).grid(row=figrowspan+1,column=0,rowspan=3,sticky=Tk.W,padx=10)

firstRowFrame = Tk.Frame(root)
firstRowFrame.grid(row=figrowspan+1,column=1,sticky=Tk.W)
Tk.Label(master=firstRowFrame,text="ParamFile").grid(row=1,column=2,sticky=Tk.W)
Tk.Button(master=firstRowFrame,text="Select",command=paramfileselect).grid(row=1,column=3,sticky=Tk.W)
Tk.Entry(master=firstRowFrame,textvariable=paramfilevar,width=20).grid(row=1,column=4,sticky=Tk.W)
Tk.Button(master=firstRowFrame,text="LoadParams",command=readParams).grid(row=1,column=5,sticky=Tk.W)

secondRowFrame = Tk.Frame(root)
secondRowFrame.grid(row=figrowspan+2,column=1,sticky=Tk.W)
Tk.Label(master=secondRowFrame,text="Method 1:",font=('Helvetica',15)).grid(row=1,column=1,sticky=Tk.W)
Tk.Button(master=secondRowFrame,text='OutputFolderM1',command=folderSelector_m1,font=("Helvetica",12)).grid(row=1,column=2,sticky=Tk.W)
Tk.Label(master=secondRowFrame,text='ResultStyle_m1').grid(row=1,column=3,sticky=Tk.W)

cbox_rs_1 = ttk.Combobox(root)
cbox_rs_1 = ttk.Combobox(master=secondRowFrame,value=Resultvalue)
cbox_rs_1.grid(row=1,column=4,sticky=Tk.W)
cbox_rs_1.current(9)
cbox_rs_1.bind("<<ComboboxSelected>>",func1)
Tk.Label(master=secondRowFrame,text='Rad_m1').grid(row=1,column=5,sticky=Tk.W)
cbox_Rad_1 = ttk.Combobox(master=secondRowFrame,value=Rad)
cbox_Rad_1.grid(row=1,column=6,sticky=Tk.W)
cbox_Rad_1.bind("<<ComboboxSelected>>",func2)
Tk.Label(master=secondRowFrame,text='Eta_m1').grid(row=1,column=7,sticky=Tk.W)
cbox_Eta_1 = ttk.Combobox(master=secondRowFrame,value=Eta)
cbox_Eta_1.grid(row=1,column=8,sticky=Tk.W)
cbox_Eta_1.bind("<<ComboboxSelected>>",func3)
Tk.Label(master=secondRowFrame,text='MinThresh_m1').grid(row=1,column=9,sticky=Tk.W)
Tk.Entry(master=secondRowFrame,textvariable=thresholdvar_m1,width=5).grid(row=1,column=10,sticky=Tk.W)
Tk.Label(master=secondRowFrame,text='MaxThresh_m1').grid(row=1,column=11,sticky=Tk.W)
Tk.Entry(master=secondRowFrame,textvariable=maxthresholdvar_m1,width=5).grid(row=1,column=12,sticky=Tk.W)
Tk.Button(master=secondRowFrame,text='UpdThresh_m1',command=replot_m1).grid(row=1,column=13,sticky=Tk.W)
Tk.Button(master=secondRowFrame,text='Load Method1',command=loadplot_m1).grid(row=1,column=14,rowspan=3,sticky=Tk.W)

thirdRowFrame = Tk.Frame(root)
thirdRowFrame.grid(row=figrowspan+3,column=1,sticky=Tk.W)
Tk.Label(master=thirdRowFrame,text="Method 2:",font=('Helvetica',15)).grid(row=1,column=1,sticky=Tk.W)
Tk.Button(master=thirdRowFrame,text='OutputFolderM2',command=folderSelector_m2,font=("Helvetica",12)).grid(row=1,column=2,sticky=Tk.W)
Tk.Label(master=thirdRowFrame,text='ResultStyle_m2').grid(row=1,column=3,sticky=Tk.W)
cbox_rs_2 = ttk.Combobox(master=thirdRowFrame,value=Resultvalue)
cbox_rs_2.grid(row=1,column=4,sticky=Tk.W)
cbox_rs_2.current(9)
cbox_rs_2.bind("<<ComboboxSelected>>",func4)
Tk.Label(master=thirdRowFrame,text='Rad_m2').grid(row=1,column=5,sticky=Tk.W)
cbox_Rad_2 = ttk.Combobox(master=thirdRowFrame,value=Rad)
cbox_Rad_2.grid(row=1,column=6,sticky=Tk.W)
cbox_Rad_2.bind("<<ComboboxSelected>>",func5)
Tk.Label(master=thirdRowFrame,text='Eta_m2').grid(row=1,column=7,sticky=Tk.W)
cbox_Eta_2 = ttk.Combobox(master=thirdRowFrame,value=Eta)
cbox_Eta_2.grid(row=1,column=8,sticky=Tk.W)
cbox_Eta_2.bind("<<ComboboxSelected>>",func6)
Tk.Label(master=thirdRowFrame,text='MinThresh_m2').grid(row=1,column=9,sticky=Tk.W)
Tk.Entry(master=thirdRowFrame,textvariable=thresholdvar_m2,width=5).grid(row=1,column=10,sticky=Tk.W)
Tk.Label(master=thirdRowFrame,text='MaxThresh_m2').grid(row=1,column=11,sticky=Tk.W)
Tk.Entry(master=thirdRowFrame,textvariable=maxthresholdvar_m2,width=5).grid(row=1,column=12,sticky=Tk.W)
Tk.Button(master=thirdRowFrame,text='UpdThresh_m2',command=replot_m2).grid(row=1,column=13,sticky=Tk.W)
Tk.Button(master=thirdRowFrame,text='Load Method2',command=loadplot_m2).grid(row=1,column=14,rowspan=3,sticky=Tk.W)

Tk.mainloop()
