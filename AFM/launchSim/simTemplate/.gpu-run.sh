#!/bin/bash
#$ -S /bin/bash
#$ -N afm_3fold_cargo 
#$ -o OUTPUT
#$ -e ERRORS
#$ -l gpu=1
#$ -V
#$ -cwd

echo "";
echo "--------------------------- Job Info ------------------------------";
echo "Starting job:" afm_3fold_cargo;
echo "Machine name:" `hostname`;
echo "Date:" `date`;
echo "User:" `whoami`;
cd /scratch/`whoami`/afm_3fold_cargo-$JOB_ID;
echo "Working directory:" `pwd`;
echo "Working directory content:" `ls`;
echo "";
echo "... Job running on `hostname` ...";
echo "";
#!/bin/bash

./AFM p22_3fold_200egfp.top p22_3fold_200egfp.enm p22_3fold_200egfp.gaussian options.in p22_3fold_200egfp_fullInteraction
