import sys
import os
import numpy as np


d = np.loadtxt(sys.argv[1])

filename, file_extension = os.path.splitext(sys.argv[1])
filename = filename+file_extension+"P"
fout = open(filename,"w+")

offset = d[0][0]
currentInd = offset
mean = 0.0
measCount = 0;

for data in d:
    if(data[0] == currentInd):
        mean += data[1]
        measCount = measCount + 1
    else:
        #print offset - currentInd, mean/measCount
        fout.write('{} {}\n'.format(currentInd, mean/measCount))
        currentInd = data[0]
        mean = 0
        measCount = 0
        mean += data[1]
        measCount = measCount + 1
        
        

