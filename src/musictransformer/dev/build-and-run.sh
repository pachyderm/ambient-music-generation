#!/bin/bash

containername='musictransformer'

err=0
trap 'err=1' ERR
docker stop $containername
docker build -t $containername -f Dockerfile .

docker run --name $containername --rm -it -p 8889:8888 --runtime nvidia \
--shm-size=1g -e NVIDIA_VISIBLE_DEVICES=0,1 \
-v $(pwd)/src:/tf/src \
-v $(pwd)/audio:/tf/audio \
-v $(pwd)/pickle:/pickle \
-v $(pwd)/save:/save \
$containername /tf/src/train_script.sh
# $containername 
