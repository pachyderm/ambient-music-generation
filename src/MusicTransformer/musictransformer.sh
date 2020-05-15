#!/bin/bash

python /tf/MusicTransformer-tensorflow2.0/train.py \
--epochs 22 --max_seq 2048 --pickle_dir /pfs/out/pickle \
--save_path /pfs/out/save --batch_size 2
