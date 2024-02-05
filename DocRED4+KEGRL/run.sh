#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

cfg=baseline
model_name=ContextAware # CNN3 LSTM BiLSTM ContextAware
config=yamls/${cfg}.yaml
save_name=checkpoint_${model_name}_${cfg}

python train.py --config $config --save_name $save_name --model_name $model_name
