#!/usr/bin/env bash

#$ -wd /export/c01/ashah108/vad
#$ -V
#$ -N vad
#$ -j y -o logs/$JOB_NAME
#$ -M ashah108@jh.edu
#$ -m e
#$ -l ram_free=2G,mem_free=2G,gpu=1,hostname=b06

# Submit to GPU queue
#$ -q g.q

source /home/gqin2/scripts/acquire-gpu

conda activate vad

# CUDA_LAUNCH_BLOCKING=1 python main.py
CUDA_LAUNCH_BLOCKING=1 python test.py
