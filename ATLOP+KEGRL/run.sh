#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

getdate=$(date +"%Y%m%d_%H%M%S");

# change this option to run different settings
config=baseline

python train.py --config configs/${config}.yaml