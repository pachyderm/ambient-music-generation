#!/bin/bash


err=0
trap 'err=1' ERR
docker stop magenta
docker build -t magenta -f Dockerfile .


# test $err = 0 # Return non-zero if any command failed

# if [ $err == 0 ]
# then

#     docker run --name magenta --rm -it -p 8889:8888 --runtime nvidia \
#     --shm-size=1g -e NVIDIA_VISIBLE_DEVICES=0,1 -v $(pwd)/src:/notebooks/src -v \
#     $(pwd)/audio:/notebooks/audio \
#     magenta
# else
#   echo 'error building docker file'
# fi

docker run --name magenta --rm -it -p 8889:8888 --runtime nvidia \
--shm-size=1g -e NVIDIA_VISIBLE_DEVICES=0,1 -v $(pwd)/src:/notebooks/src -v \
$(pwd)/audio:/notebooks/audio \
magenta
