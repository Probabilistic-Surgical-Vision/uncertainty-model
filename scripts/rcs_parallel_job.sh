#!/bin/bash

#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=8:ngpus=2:mem=75gb:gpu_type=RTX6000

cd $PBS_O_WORKDIR

module load anaconda3/personal
source activate tukra-env

python parallel_main.py config.yml da-vinci --epochs 50 \
    --save-model-to trained/da-vinci --save-model-every 10 \
    --save-evaluation-to results/da-vinci --evaluate-every 10 \
    --home /rds/general/user/lem3617/home --number-of-gpus 2