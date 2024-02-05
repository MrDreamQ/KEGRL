#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=ContextAware
cfg=baseline
config=yamls/${cfg}.yaml
ckpt_path=xxx
input_theta=0.

python test.py \
    --model_name ${model_name} \
    --save_name ${ckpt_path} \
    --input_theta ${input_theta} \
    --config ${config} 
