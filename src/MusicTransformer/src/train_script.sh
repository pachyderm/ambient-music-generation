#!/bin/bash

python3 /tf/MusicTransformer-tensorflow2.0/train.py \
  --epochs=22 --save_path=/save --max_seq=1024 \
  --pickle_dir=/pickle \
  --batch_size=1
