#!/usr/bin/env bash

#$ -wd /export/c01/ashah108/vad
#$ -V
#$ -N vad-test
#$ -j y -o logs/$JOB_NAME
#$ -M ashah108@jh.edu
#$ -m e
#$ -l ram_free=11G,mem_free=11G,gpu=1,hostname=c01

# #$ -l ram_free=5G,mem_free=5G,gpu=1,hostname=b1[123456789]|c0*|c1[123456789]

# Submit to GPU queue
#$ -q g.q

# source /home/gqin2/scripts/acquire-gpu

conda activate vad

# CUDA_LAUNCH_BLOCKING=1 python main.py

# export CUDA_VISIBLE_DEVICES=$(free-gpu -n 1)
# python main.py

CUDA_VISIBLE_DEVICES=$(free-gpu) python main.py