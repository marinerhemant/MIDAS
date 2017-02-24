#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

type file;

app indexrefine (int spotsinput, string folder)
{
   indexstrains spotsinput folder;
}

string fldr = arg("folder","/data/tomo1/NFTest/");

foreach i in [1:2000] {
    indexrefine(i, fldr);
}
