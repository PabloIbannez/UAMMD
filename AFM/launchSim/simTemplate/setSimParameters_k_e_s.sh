#!/bin/bash

sed -i "s/^k .*$/k $1/" options.in
sed -i "s/^epsilonGaussian .*$/epsilonGaussian $2/" options.in
sed -i "s/^sigmaGaussian .*$/sigmaGaussian $3/" options.in
