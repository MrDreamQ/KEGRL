#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

ckpt=xxx.pth

config=baseline

python test.py --ckpt $ckpt --config configs/$config.yaml