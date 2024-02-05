#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

echo ======================================== Prepro DocRED ========================================
in_path=../dataset/DocRED
out_path=prepro_glove
python gen_data.py --in_path $in_path --out_path $out_path 

out_path=prepro_bert
python gen_data_bert.py --in_path $in_path --out_path $out_path 


echo ======================================== Prepro Re-DocRED ========================================
in_path=../dataset/Re-DocRED
out_path=prepro_re_glove
python gen_data.py --in_path $in_path --out_path $out_path 

out_path=prepro_re_bert
python gen_data_bert.py --in_path $in_path --out_path $out_path 


echo ======================================== Prepro DWIE ========================================
in_path=../dataset/DWIE
out_path=prepro_dwie_glove
python gen_data_dwie.py --in_path $in_path --out_path $out_path 

out_path=prepro_dwie_bert
python gen_data_bert_dwie.py --in_path $in_path --out_path $out_path 
