#!/bin/bash


err=0
trap 'err=1' ERR
docker build -t onsets-and-frames -f Dockerfile .


test $err = 0 # Return non-zero if any command failed

if [ $err == 0 ]
then
    # docker run --name onsets-and-frames --rm -it -p 8889:8888 --runtime nvidia \
    # --shm-size=1g -e NVIDIA_VISIBLE_DEVICES=0,1 -v $(pwd)/src:/notebooks/src -v \
    # $(pwd)/../../audio/wave_files:/pfs/dev-audio-processed-wav -v $(pwd)/out:/pfs/out \
    # onsets-and-frames python3 /notebooks/src/transcribe.py

    docker run --name onsets-and-frames --rm -it \
    --shm-size=1g -v $(pwd)/src:/notebooks/src -v \
    $(pwd)/../../audio/wave_files:/pfs/dev-audio-processed-wav -v $(pwd)/out:/pfs/out \
    onsets-and-frames
else
  echo 'error building docker file'
fi
