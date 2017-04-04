#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

type file;

app (file ep) runPeaks (string paramsfn, int fnr, int ringnr)
{
	peaks paramsfn fnr ringnr stderr=filename(ep);
}

app (file err) runProcessPeaks (string paramsf, int RNr, file DummyA[])
{
	processPeaks paramsf RNr stderr=filename(err);
}

app (file err) mergerings (string pfname, file dummy[])
{
	mergeRings pfname stderr=filename(err);
}

app (file err, file spotsfile) postpeaks (string foldername, string pfname, file dummy)
{
	postPeaksHydra foldername pfname filename(spotsfile) stderr=filename(err);
}

app (file err) indexrefine (string foldername, int spotsinput, file dm)
{
	indexstrains spotsinput foldername stderr=filename(err);
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

# End parameters ########################

int rings[] = readData(ringfile);
string folderNames[] = readData(strcat(seedfolder,"/FolderNames.txt"));
string PFNames[] = readData(strcat(seedfolder,"/PFNames.txt"));

iterate ix {
	string foldername = folderNames[ix];
	foreach detector in [1:4]{
		string paramfilenamefile = 
	}
	string paramfilenamefile = strcat(
}until (ix == length(foldernames));
