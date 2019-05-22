#!/bin/bash

for disp in 0.0 1.0 2.5 5.0
do
    ./setSimParameters_tipDispl.sh $disp
    ./AFM p22_3fold.top p22_3fold.enm p22_3fold.gaussian options.in p22_3fold_disp$disp
done
