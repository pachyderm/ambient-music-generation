#!/bin/bash


err=0
trap 'err=1' ERR
docker build -t onsets-and-frames .
test $err = 0 # Return non-zero if any command failed

if [ $err == 0 ]
then
  # To run with the Jupyter notebook, leave off the python script call
  # docker run --name onsets-and-frames -p 8888:8888 -v $(pwd)/src:/code/src -v $(pwd)/checkpoints:/checkpoints -v $(pwd)/samples:/samples --rm -it onsets-and-frames

  # To run training out of the box:
  docker run --name onsets-and-frames -v $(pwd)/src:/code/src -v $(pwd)/checkpoints:/checkpoints -v $(pwd)/samples:/samples --rm -it onsets-and-frames ./src/transcribe
else
  echo 'error building docker file'
fi
