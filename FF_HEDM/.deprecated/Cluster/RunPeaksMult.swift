#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

type file;

app (file op, file ep) runPeaks (string paramsfn, int fnr, int ringnr)
{
 peaks paramsfn fnr ringnr stdout=filename(op) stderr=filename(ep);
}

app runProcessPeaks (string paramsf, int RNr, file DummyA[], file hkl)
{
 processPeaks paramsf RNr;
}

# Parameters to be modified #############

int startnr = toInt(arg("startnr","1"));
int endnr = toInt(arg("endnr","600"));
string parameterfilestem = arg("paramsfile","/clhome/TOMO1/PeaksAnalysisHemant/PeaksFittingCode/90_33ParamsFile1.txt");
string ringfile = arg("ringfile","RingInfo.txt");
string fstm = arg("fstm","PS.txt");

# End parameters ########################

file hkl <"hkls.csv">;

int rings[] = readData(ringfile);

foreach Ring in rings {
    string PFst1 = strcat(parameterfilestem,Ring);
    string parameterfilename = strcat(PFst1,"_",fstm);
    tracef("%s\n",parameterfilename);
    string PreFix1 = strcat("PeaksPerFile_",Ring);
    file simAerr[]<simple_mapper;location="output",prefix=PreFix1,suffix=".err">;
    file simAout[]<simple_mapper;location="output",prefix=PreFix1,suffix=".out">;
    foreach i in [startnr:endnr] {
       (simAout[i],simAerr[i]) = runPeaks(parameterfilename,i,Ring);
    }
    runProcessPeaks(parameterfilename,Ring,simAerr,hkl);
}
