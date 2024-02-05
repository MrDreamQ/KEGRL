#!/bin/bash

echo ======================================== Prepro DocRED ========================================
in_path=../dataset/DocRED
out_path=prepro_data
python gen_data.py --in_path $in_path --out_path $out_path


echo ======================================== Prepro Re-DocRED ========================================
in_path=../dataset/Re-DocRED
out_path=prepro_re
python gen_data.py --in_path $in_path --out_path $out_path

echo ======================================== Prepro DWIE ========================================
in_path=../dataset/DWIE
out_path=prepro_dwie
python dwie_gen_data.py --in_path $in_path --out_path $out_path
