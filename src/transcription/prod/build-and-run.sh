#!/bin/bash


err=0
trap 'err=1' ERR
docker build -t onsets-and-frames -f jupyter.Dockerfile .


test $err = 0 # Return non-zero if any command failed

if [ $err == 0 ]
then
  # To run with the Jupyter notebook, leave off the python script call
  # docker run --name onsets-and-frames -p 8888:8888 -v $(pwd)/src:/code/src -v $(pwd)/checkpoints:/checkpoints -v $(pwd)/samples:/samples --rm -it onsets-and-frames

  # To run training out of the box:
  # docker run --name onsets-and-frames -v $(pwd)/src:/code/src -v $(pwd)/checkpoints:/checkpoints -v $(pwd)/samples:/samples --rm -it onsets-and-frames ./src/transcribe

    docker run --name onsets-and-frames --rm -it -p 8889:8888 --runtime nvidia \
    --shm-size=1g -e NVIDIA_VISIBLE_DEVICES=0,1 -v $(pwd)/src:/notebooks/src -v \
    $(pwd)/audio:/pfs/audio-processed-wav -v $(pwd)/out:/pfs/out \
    onsets-and-frames python3 /notebooks/src/transcribe.py
else
  echo 'error building docker file'
fi
