#!/bin/bash

echo 'music transformer shell script 4'
python /src/train.py
# python /src/train.py \
# --epochs 22 --max_seq 2048 --pickle_dir /pfs/out/pickle \
# --save_path /pfs/out/save --batch_size 2 $@
