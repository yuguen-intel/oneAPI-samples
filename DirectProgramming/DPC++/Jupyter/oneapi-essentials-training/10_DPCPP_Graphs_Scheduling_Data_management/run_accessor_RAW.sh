#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module10 -- DPCPP Graphs and Dependencies sample - 3 of 11 accessors_RAW.cpp
dpcpp lab/accessors_RAW.cpp
if [ $? -eq 0 ]; then ./a.out; fi

