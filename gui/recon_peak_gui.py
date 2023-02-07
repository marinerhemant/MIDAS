
def getImage(fn,reconSize):
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
    #global Header
    #Header = HeaderVar.get()
    rec_Nr = int(rec_nrvar.get())
    threshold = float(thresholdvar_m1.get())
    upperthreshold = float(maxthresholdvar_m1.get())
   # bytesToSkip = Header + framesToSkip*(BytesPerPixel*NrPixelsY*NrPixelsZ)
    bdata_m1 = getImage(fileStem_m1,NrPixels_m1)
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
    #global Header
    #Header = HeaderVar.get()
    threshold = float(thresholdvar_m2.get())
    upperthreshold = float(maxthresholdvar_m2.get())
   # bytesToSkip = Header + framesToSkip*(BytesPerPixel*NrPixelsY*NrPixelsZ)
   
    NrPixels_m2 = int(NrPixelsvar_m2.get())
    data_m2 = getImage(fileStem_m2,NrPixels_m2)
    refreshPlot = 1
    m2.imshow(data_m2,cmap=plt.get_cmap('bone'),interpolation='nearest',clim=(threshold,upperthreshold))
    m2.set_xlim([lims[0][0],lims[0][1]])
    m2.set_ylim([lims[1][0],lims[1][1]])
    m2.title.set_text("Method 2")
    canvas.draw()
    canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)

def incr_plotupdater():
    global rec_Nr
    global rec_nrvar
    rec_Nr = int(rec_nrvar.get())
    rec_Nr += 1
    rec_nrvar.set(str(rec_Nr))
    loadplot_m1()

def decr_plotupdater():
    global rec_Nr
    global rec_nrvar
    rec_Nr = int(rec_nrvar.get())
    rec_Nr -= 1
    rec_nrvar.set(str(rec_Nr))
    loadplot_m1()

def readParams():
    global paramFN
    paramFN = paramfilevar.get()
    global NrPixels_m1
    paramContents = open(paramFN,'r').readlines()
    for line in paramContents:
        if line == '\n':
            continue
        if line[0] == '#':
            continue
        if 'detYdim' == line.split()[0]:
            NrPixels_m1 = int(line.split()[1])
        
def paramfileselect():
    global paramFN
    global paramfilevar
    paramFN = tkFileDialog.askopenfilename()
    paramfilevar.set(paramFN)
    
def selectFile():
    return tkFileDialog.askopenfilename()

def firstFileSelector():
    global fileStem_m1, folder, padding,firstReconNr
   
    #global Header, BytesPerPixel
    
   # Header = HeaderVar.get()
    BytesPerPixel = BytesVar.get()
    firstfilefullpath = selectFile()
    fullfilename = firstfilefullpath.split('/')[-1].split('.')[0]
    fileStem_m1 = '_'.join(fullfilename.split('_')[:-1])
    firstReconNr = int(fullfilename.split('_')[-1])
    firstReconNrVar.set(firstReconNr)
    padding = len(fullfilename.split('_')[-1])
    folder = os.path.dirname(firstfilefullpath) + '/'
   
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

def selectFittingResult():


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
rec_Nr = 0
paramFN = 'tomo_config.txt'
data_m1 = None
data_m2 = None
NrPixels_m1 = 128
NrPixels_m2 = 55

#Header = 8192
BytesPerPixel = 2

firstReconNrVar = Tk.StringVar()
firstReconNrVar.set(str(1))
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

NrPixelsvar_m2 = Tk.StringVar()
NrPixelsvar_m2.set(str(55))


#HeaderVar = Tk.IntVar()
#HeaderVar.set(8192)
#BytesVar = Tk.IntVar()
#BytesVar.set(2)
refreshPlot = 0

canvas.get_tk_widget().grid(row=0,column=0,columnspan=figcolspan,rowspan=figrowspan,sticky=Tk.W+Tk.E+Tk.N+Tk.S)
toolbar_frame = Tk.Frame(root)
toolbar_frame.grid(row=figrowspan+4,column=0,columnspan=10,sticky=Tk.W)
toolbar = NavigationToolbar2Tk( canvas, toolbar_frame )
toolbar.update()

Tk.Button(master=root,text='Quit',command=_quit,font=("Helvetica",20)).grid(row=figrowspan+1,column=0,rowspan=3,sticky=Tk.W,padx=10)

firstRowFrame = Tk.Frame(root)
firstRowFrame.grid(row=figrowspan+1,column=1,sticky=Tk.W)
Tk.Label(master=firstRowFrame,text="Method 1:",font=('Helvetica',15)).grid(row=1,column=1,sticky=Tk.W)
Tk.Label(master=firstRowFrame,text="ParamFile").grid(row=1,column=3,sticky=Tk.W)
Tk.Button(master=firstRowFrame,text="Select",command=paramfileselect).grid(row=1,column=4,sticky=Tk.W)
Tk.Entry(master=firstRowFrame,textvariable=paramfilevar,width=20).grid(row=1,column=5,sticky=Tk.W)
Tk.Button(master=firstRowFrame,text="LoadParams",command=readParams).grid(row=1,column=6,sticky=Tk.W)
Tk.Button(master=firstRowFrame,text='FirstReconFile',command=firstFileSelector,font=("Helvetica",12)).grid(row=1,column=1,sticky=Tk.W)
Tk.Label(master=firstRowFrame,text='firstReconNr').grid(row=1,column=4,sticky=Tk.W)
Tk.Entry(master=firstRowFrame,textvariable=firstReconNrVar,width=5).grid(row=1,column=5,sticky=Tk.W)
Tk.Button(master=firstRowFrame,text='+',command=incr_plotupdater,font=("Helvetica",12)).grid(row=1,column=10,sticky=Tk.W)
Tk.Button(master=firstRowFrame,text='-',command=decr_plotupdater,font=("Helvetica",12)).grid(row=1,column=11,sticky=Tk.W)
Tk.Checkbutton(master=firstRowFrame,text="Shepp\Logan",variable=,command=).grid(row=1,column=13,sticky=Tk.W)
Tk.Checkbutton(master=firstRowFrame,text="Hann",variable=,command=).grid(row=1,column=14,sticky=Tk.W)
Tk.Checkbutton(master=firstRowFrame,text="Hamming",variable=,command=).grid(row=1,column=15,sticky=Tk.W)
Tk.Checkbutton(master=firstRowFrame,text="Ramp",variable=,command=).grid(row=1,column=16,sticky=Tk.W)

cbox = ttk.Combobox(Tk)
cbox.grid(row = 1, sticky="NW")
cbox['value'] = ("RMEAN", "MixFactor", "SigmaG", "SigmaL", "MaxInt", "BGFit",
           "BGSimple", "MeanError", "FitIntegratedIntensity", "TotalIntensity", "TotalIntensityBackgroundCorr", "MaxIntensityObs")
cbox.current(9)

def func(event):
    text.insert('insert',cbox.get()+"\n")

cbox.bind("<<ComboboxSelected>>",func)
text = tkinter.Text(win)
text.grid(pady = 5)






Tk.Label(master=firstRowFrame,text='MinThresh_m1').grid(row=1,column=5,sticky=Tk.W)
Tk.Entry(master=firstRowFrame,textvariable=thresholdvar_m1,width=5).grid(row=1,column=6,sticky=Tk.W)
Tk.Label(master=firstRowFrame,text='MaxThresh_m1').grid(row=1,column=7,sticky=Tk.W)
Tk.Entry(master=firstRowFrame,textvariable=maxthresholdvar_m1,width=5).grid(row=1,column=8,sticky=Tk.W)
Tk.Button(master=firstRowFrame,text='UpdThresh_m1',command=replot_m1).grid(row=1,column=9,sticky=Tk.W)
Tk.Button(master=firstRowFrame,text='Load Method1',command=loadplot_m1).grid(row=figrowspan+1,column=2,rowspan=3,sticky=Tk.W)

secondRowFrame = Tk.Frame(root)
secondRowFrame.grid(row=figrowspan+2,column=1,sticky=Tk.W)
Tk.Label(master=secondRowFrame,text="Method 2:",font=('Helvetica',15)).grid(row=1,column=1,sticky=Tk.W)
Tk.Label(master=secondRowFrame,text='TotalFilesNr').grid(row=1,column=4,sticky=Tk.W)
Tk.Entry(master=secondRowFrame,textvariable=NrPixelsvar_m2,width=5).grid(row=1,column=5,sticky=Tk.W)
scroll down
Tk.Label(master=secondRowFrame,text='MinThresh_m2').grid(row=1,column=5,sticky=Tk.W)
Tk.Entry(master=secondRowFrame,textvariable=thresholdvar_m2,width=5).grid(row=1,column=6,sticky=Tk.W)
Tk.Label(master=secondRowFrame,text='MaxThresh_m2').grid(row=1,column=7,sticky=Tk.W)
Tk.Entry(master=secoondRowFrame,textvariable=maxthresholdvar_m2,width=5).grid(row=1,column=8,sticky=Tk.W)
Tk.Button(master=secondRowFrame,text='UpdThresh_m2',command=replot_m2).grid(row=1,column=9,sticky=Tk.W)
Tk.Button(master=secondRowFrame,text='Load Method2',command=loadplot_m2).grid(row=figrowspan+1,column=2,rowspan=3,sticky=Tk.W)

Tk.mainloop()
