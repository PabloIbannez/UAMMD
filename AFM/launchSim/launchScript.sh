#!/bin/bash

k=1000
e=0.1

for s in 0.1 0.5 1.0 1.5 3.0 5.0 
do
    echo $e $k $s
    ./setSimParameters_k_e_s.sh $k $e $s
    ./AFM
done
