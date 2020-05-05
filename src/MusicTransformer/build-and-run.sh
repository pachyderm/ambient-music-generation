#!/bin/bash

containername='musictransformer'

err=0
trap 'err=1' ERR
docker stop $containername
docker build -t $containername -f Dockerfile .

# test $err = 0 # Return non-zero if any command failed

# if [ $err == 0 ]
# then

    docker run --name $containername --rm -it -p 8889:8888 --runtime nvidia \
    --shm-size=1g -e NVIDIA_VISIBLE_DEVICES=0,1 \
    -v $(pwd)/MusicTransformer-pytorch:/notebooks/MusicTransformer-pytorch \
    -v $(pwd)/MusicTransformer-tensorflow2.0:/notebooks/MusicTransformer-tensorflow2.0 \
    -v $(pwd)/audio:/notebooks/audio \
    $containername
# else
#   echo 'error building docker file'
# fi
