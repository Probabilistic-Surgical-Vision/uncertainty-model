#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=lem3617

export PATH=/vol/bitbucket/${USER}/tukra-model-env/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

python main.py config.yaml da-vinci --no-pbar