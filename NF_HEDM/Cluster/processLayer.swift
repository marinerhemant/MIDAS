#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

type file;

app (file out, file err) Medians (string pf, int layernr)
{
	runmedianparallel pf layernr stdout=@filename(out) stderr=@filename(err);
}

app (file outm, file errm) Images (string paramf, int layern, int filenr, file inp)
{
	runimageprocessingparallel paramf layern filenr stdout=@filename(outm) stderr=@filename(errm);
}

app (file done) PlaceHolder (string prefix, file out[])
{
	echo prefix stdout=@filename(done);
}

app (file done) PlaceHolder2 (string prefix, file out)
{
	echo prefix stdout=@filename(done);
}

app runfitorientation (string pf, int nr, string micfn, file mmapdone)
{
	fitorientation pf nr micfn;
}

app (file mmapdone) mmapcode (string paramfn, string dire, file imagedone)
{
	mmaps paramfn dire stdout=@filename(mmapdone);
}

# Parameters to be modified ############

string paramfile = arg("ParameterFileName","/data/tomo1/NFTest/ParametersGoldApril14.txt");
int NrLayers = toInt(arg("NrLayers","3"));
int NrFilesPerLayer = toInt(arg("NrFilesPerLayer","180"));
int startnr = toInt(arg("StartNumber","1"));
int endnr = toInt(arg("EndNumber","100"));
string micfn = arg("MicFileName","microstructure.mic");
int DoPeakSearch = toInt(arg("DoPeakSearch","1"));
string direct = arg("DataDirectory","/data/tomo1/NFTest/");

# End parameters #######################


## Whether do peak search or not
file imagesdone <"imageprocessing.txt">;
if (DoPeakSearch == 1){
	trace("Doing peaksearch.\n");
	string prefix2 = strcat("ImageProcessing_");
	file simBout[]<simple_mapper;location="output",prefix=prefix2,suffix=".out">;
	file simBerr[]<simple_mapper;location="output",prefix=prefix2,suffix=".err">;
	foreach layer in [1:NrLayers] {
		string prefix1 = strcat("Median_",layer);
		file simAout <simple_mapper;location="output",prefix=prefix1,suffix=".out">;
		file simAerr <simple_mapper;location="output",prefix=prefix1,suffix=".err">;
		(simAout,simAerr) = Medians(paramfile,layer);
		foreach FileNr in [0:(NrFilesPerLayer-1)]{
			(simBout[(layer-1)*NrFilesPerLayer + FileNr],simBerr[(layer-1)*NrFilesPerLayer + FileNr]) = Images(paramfile, layer, FileNr,simAout);
		}
	}
	imagesdone = PlaceHolder("Done",simBout);
} else {
	trace("Not doing peaksearch");
	string prefix2 = "ImageProcessing";
	file simBout<simple_mapper;location="output",prefix=prefix2,suffix=".out">;
	imagesdone = PlaceHolder2(prefix2,simBout);
}

## Now MMap Images
file mmapdone <"output/mmapdone.txt">;
mmapdone = mmapcode(parafile,direct,imagesdone);

## Now do FitOrientation
foreach i in [startnr:endnr] {
	runfitorientation(paramfile,i,micfn,mmapdone);
}
