#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

type file;

app (file ep) runPeaks (string paramsfn, int fnr, int ringnr)
{
	peaks paramsfn fnr ringnr stderr=filename(ep);
}

app (file out, file err) runProcessPeaks (string paramsf, int RNr, file DummyA[])
{
	processPeaks paramsf RNr stdout=filename(out) stderr=filename(err);
}

app (file out, file err) mergerings (string pfname, file dummy[])
{
	mergeRings pfname stdout=filename(out) stderr=filename(err);
}

app (file out) datareader (string fn, file dm)
{
	cat fn stdout=filename(out);
}

app (file out, file err) postpeaks (string foldername, string pfname, file dummy)
{
	postPeaks foldername pfname stdout=filename(out) stderr=filename(err);
}

app (file out, file err) postpeaks2 (string foldername, string pfname)
{
	postPeaks foldername pfname stdout=filename(out) stderr=filename(err);
}

app (file out, file err) indexrefine (string foldername, int spotsinput)
{
	indexstrains spotsinput foldername stdout=filename(out) stderr=filename(err);
}

app (file out) processgrains (string foldername, string pfname, file dummy[])
{
	processGrains foldername pfname stdout=filename(out);
}

# Parameters to be modified #############

int startnr = toInt(arg("startnr","1"));
int endnr = toInt(arg("endnr","600"));
string ringfile = arg("ringfile","RingInfo.txt");
string seedfolder = arg("SeedFolder","/clhome/FolderNames.txt");
int dopeaksearch = toInt(arg("DoPeakSearch","1"));
int nLayers = toInt(arg("nLayers","1"));

# End parameters ########################

int rings[] = readData(ringfile);
string folderNames[] = readData(strcat(seedfolder,"/FolderNames.txt"));
string PFNames[] = readData(strcat(seedfolder,"/PFNames.txt"));

if (dopeaksearch == 1) {
	###### Change to iterate instead of foreach
	iterate ix {
	#foreach foldername,ix in folderNames { 
		string foldername = folderNames[ix];
		# equivalent to layernr
		string paramfilenamefile = strcat(foldername,"/ParamFileNames.txt");
		string paramFileNames[] = readData(paramfilenamefile);
		file simBerr[]<simple_mapper;location=strcat(foldername,"/output"),prefix=strcat("ProcessPeaks_",ix,"_"),suffix=".err">;
		file simBout[]<simple_mapper;location=strcat(foldername,"/output"),prefix=strcat("ProcessPeaks_",ix,"_"),suffix=".out">;
		foreach Ring,idx in rings {
			string parameterfilename = paramFileNames[idx];
			string PreFix1 = strcat("PeaksPerFile_",Ring);
			file simAerr[];
			foreach i in [startnr:endnr] {
				file simx<simple_mapper;location=strcat(foldername,"/output"),prefix=strcat(PreFix1,"_",i,"_"),suffix=".err">;
				simx = runPeaks(parameterfilename,i,Ring);
				if (i %% 100 == 0){
					simAerr[i%/100] = simx;
				}
			}
			(simBout[idx],simBerr[idx]) = runProcessPeaks(parameterfilename,Ring,simAerr);
		}
		# take the output of this, run mergemultiple rings for each layer
		string pfname = PFNames[ix];
		file simCerr<simple_mapper;location=strcat(foldername,"/output"),prefix=strcat("MergeRings_",ix),suffix=".err">;
		file simCout<simple_mapper;location=strcat(foldername,"/output"),prefix=strcat("MergeRings_",ix),suffix=".out">;
		(simCout,simCerr) = mergerings(pfname, simBout);
		# take the output, run shmoperators
		file simDerr<simple_mapper;location=strcat(foldername,"/output"),prefix=strcat("PostPeaksSHM_",ix),suffix=".err">;
		file simDout<simple_mapper;location=strcat(foldername,"/output"),prefix=strcat("PostPeaksSHM_",ix),suffix=".out">;
		int spots[];
		(simDout,simDerr) = postpeaks(foldername,pfname,simCout);
		file simCatOut<single_file_mapper;file=strcat(foldername,"/SpotsCopy.csv")>;
		simCatOut = datareader(strcat(foldername,"/SpotsToIndex.csv"),simDout);
		spots = readData(simCatOut);
		file all[];
		foreach i in spots {
			file simEerr<simple_mapper;location=strcat(foldername,"/output"),prefix=strcat("IndexRefine_",ix,"_"),suffix=".err">;
			file simEout<simple_mapper;location=strcat(foldername,"/output"),prefix=strcat("IndexRefine_",ix,"_"),suffix=".out">;
			(simEout,simEerr) = indexrefine(i);
			if (i %% 100 == 0){
				all[i %/ 100] = simEout;
			}
		}
		file processgrainsout <single_file_mapper;file=strcat(foldername,"output/processgrains.txt")>;
		processgrainsout = processgrains(foldername,pfname,all);
	#}
	} until (ix == length(folderNames));
} else {
	###### Change to iterate instead of foreach
	iterate ix {
	#foreach foldername,ix in folderNames { 
		string foldername = folderNames[ix];
		string pfname = PFNames[ix];
		# equivalent to layernr
		file simDerr<simple_mapper;location=strcat(foldername,"/output"),prefix=strcat("PostPeaksSHM_",ix),suffix=".err">;
		file simDout<simple_mapper;location=strcat(foldername,"/output"),prefix=strcat("PostPeaksSHM_",ix),suffix=".out">;
		int spots[];
		(simDout,simDerr) = postpeaks2(foldername,pfname);
		file simCatOut<single_file_mapper;file=strcat(foldername,"/SpotsCopy.csv")>;
		simCatOut = datareader(strcat(foldername,"/SpotsToIndex.csv"),simDout);
		spots = readData(simCatOut);
		file all[];
		foreach i in spots {
			file simEerr<simple_mapper;location=strcat(foldername,"/output"),prefix=strcat("IndexRefine_",ix,"_"),suffix=".err">;
			file simEout<simple_mapper;location=strcat(foldername,"/output"),prefix=strcat("IndexRefine_",ix,"_"),suffix=".out">;
			(simEout,simEerr) = indexrefine(i);
			if (i %% 100 == 0){
				all[i %/ 100] = simEout;
			}
		}
		file processgrainsout <single_file_mapper;file=strcat(foldername,"output/processgrains.txt")>;
		processgrainsout = processgrains(foldername,pfname,all);
	#}
	} until (ix == length(folderNames));
}
