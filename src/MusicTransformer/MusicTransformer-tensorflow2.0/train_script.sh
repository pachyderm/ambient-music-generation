#!/bin/bash

python3 /notebooks/MusicTransformer-tensorflow2.0/train.py \
  --epochs=1 --save_path=./save --max_seq=2048 \
  --pickle_dir=./pickle --batch_size=2 --l_r=0.0001
