#!/bin/bash -eu

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

for i in *.out; do
   tail -n 1 "$i"
done
