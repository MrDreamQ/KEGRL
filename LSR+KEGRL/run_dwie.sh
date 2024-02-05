#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

getdate=$(date +"%Y%m%d_%H%M%S")

model_name=LSR_bert  # [ LSR, LSR_bert ]
dataset=dwie

# run baseline when setting sgraph=0 and dgraph=0
sgraph=0
dgraph=0

epoch=400
hidden_dim=120
batch_size=5
grad_accu_step=4
data_path=./prepro_dwie_glove
lr=1e-3

if [[ $model_name = LSR_bert ]];
then
    hidden_dim=216
    batch_size=2
    grad_accu_step=10
    lr=1e-3
    data_path=./prepro_dwie_bert
fi

python train.py \
    --model_name ${model_name} \
    --hidden_dim ${hidden_dim} \
    --batch_size $batch_size \
    --grad_accu_step $grad_accu_step \
    --data_path $data_path \
    --save_name ${model_name}_${getdate}.pth \
    --lr ${lr} \
    --sgraph $sgraph \
    --dgraph $dgraph \
    --dataset $dataset \
    --num_epoch $epoch