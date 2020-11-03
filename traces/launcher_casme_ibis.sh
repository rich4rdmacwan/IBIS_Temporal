#!/bin/bash
#This launcher should be placed in its own directory, e.g. CasmeSq/traces. 
#Otherwise, the path to casmedb should be passed.
if [[ "$#" -lt 1 ]]; then
    echo "Usage ./ibislauncher <path_to_ibistemporal_executable> [path_to_casmedb]."
    echo "If the second argument is not given, this launcher should be placed in CasmeSq/traces directory."
    exit
fi

casmedbpath=../rawvideo/*
if [[ "$#" = 2 ]]; then
	#Check optional casmedbpath argument
	casmedbpath=$2
fi
for d in casmedbpath; do
  if [ -d "$d" ]; then
    	#Subject directory
	echo $d
	for f in `ls "$d"`; do
		$1 100 50 $d/$f
	done
  fi
done
