#!/bin/bash

docker build -t testing-musictransformer-piano -f research.Dockerfile ../..

docker run --name testing-musictransformer-piano \
    --rm -p 8889:8888 -p 6006:6006 -v $(pwd)/../../musictransformer/dev/:/notebooks/src \
    -v $(pwd)/../../MusicTransformer-tensorflow2.0:/notebooks/MusicTransformer-tensorflow2.0 \
    --gpus all testing-musictransformer-piano
