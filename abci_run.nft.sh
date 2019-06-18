#!/bin/bash

#$ -l rt_G.small=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -cwd
#$ -m ea
#$ -M dnakhoa.aist@gmail.com

source /etc/profile.d/modules.sh
module load python/3.6/3.6.5
module load cuda/10.0/10.0.130
module load cudnn/7.5/7.5.0
module load nccl/2.4/2.4.2-1

source ~/ml/bin/activate

bash neptune_api.sh

python -u main.py --train_file=corpora/WSJ-PTB/02-21.10way.clean.train --dev_file=corpora/WSJ-PTB/22.auto.clean.dev --test_file=corpora/WSJ-PTB/23.auto.clean.test --output_dir=outputs.nft --bert_model=models/bert-base-multilingual-cased --batch_size=32 --num_epochs=20 --learning_rate=3e-4 --freeze_bert > training.nft.log
