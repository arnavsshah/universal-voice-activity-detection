#!/usr/bin/env bash

#$ -wd /export/c01/ashah108/vad
#$ -V
#$ -N all-50-sincnet-musan-weighted-stop-early-false
#$ -j y -o logs/$JOB_NAME
#$ -M ashah108@jh.edu
#$ -m e
#$ -l ram_free=10G,mem_free=10G,gpu=4

# Submit to GPU queue
#$ -q g.q

# source /home/gqin2/scripts/acquire-gpu

# conda activate universal-vad
export PATH="/home/ashah108/miniconda3/envs/universal-vad/bin:$PATH"
source activate /home/ashah108/miniconda3/envs/universal-vad


# CUDA_LAUNCH_BLOCKING=1 python main.py

# export CUDA_VISIBLE_DEVICES=$(free-gpu -n 1)
# python main.py
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256'
CUDA_VISIBLE_DEVICES=$(free-gpu -n 4) python main.py