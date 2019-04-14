#!/bin/bash

#$ -l rt_G.small=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -cwd
#$ -m ea
#$ -M dnakhoa.aist@gmail.com

source ~/ds/bin/activate

source /etc/profile.d/modules.sh
module load python/3.6/3.6.5
module load cuda/10.0/10.0.130
module load cudnn/7.5/7.5.0
module load nccl/2.4/2.4.2-1

python main.py train
