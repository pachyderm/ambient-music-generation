#!/bin/bash

python3 /tf/MusicTransformer-tensorflow2.0/train.py \
  --epochs=200 --save_path=/save --max_seq=1024 \
  --pickle_dir=/pickle \
  --batch_size=1 --l_r=0.0001
