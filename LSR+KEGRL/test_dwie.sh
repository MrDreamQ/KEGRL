#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

model_name=LSR_bert # [ LSR, LSR_bert ]
dataset=dwie

save_name=xxx.pth
input_theta=0.
test_prefix=dev_test

sgraph=0
dgraph=0

hidden_dim=120
batch_size=200
data_path=./prepro_dwie_glove

if [[ $model_name = LSR_bert ]]
then
    hidden_dim=216
    batch_size=100
    data_path=./prepro_dwie_bert
fi

python test.py \
    --model_name ${model_name} \
    --hidden_dim ${hidden_dim} \
    --save_name ${save_name} \
    --input_theta ${input_theta} \
    --test_prefix $test_prefix \
    --hidden_dim ${hidden_dim} \
    --data_path ${data_path} \
    --dataset ${dataset} \
    --sgraph $sgraph \
    --dgraph $dgraph

