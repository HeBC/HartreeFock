#!/bin/bash
date
export LD_LIBRARY_PATH=/vol6/intel_composer_xe_2013.0.079_lib:$LD_LIBRARY_PATH
yhrun -N2 -n 24 -p TH_SR1 ./NPSM_pn
date
