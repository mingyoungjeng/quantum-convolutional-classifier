#!/bin/bash

A="B"
B=3

echo ${!A}

C=$(echo $A | awk '{print tolower($0)}')
echo --$C

# qcc --help

filename="cli/convolution_pooling.sh"
source $filename
VARLIST=$(cat $filename | grep -v "#" | cut -d "=" -f 1)
for i in $VARLIST
do
   echo $i, ${!i}
done

