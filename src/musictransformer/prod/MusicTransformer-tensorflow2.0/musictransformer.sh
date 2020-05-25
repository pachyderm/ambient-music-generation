#!/bin/bash

echo 'music transformer shell script 0.1.9'
python /src/train.py --max_seq 1024 --input /pfs/transformer-preprocess
