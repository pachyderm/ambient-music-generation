#!/bin/bash

# If containers fail to die, uncomment this line
# docker stop pachyderm-magenta-js || true && docker rm pachyderm-magenta-js || true

docker build -t pachyderm-magenta-js .

docker run \
  --name pachyderm-magenta-js \
  --rm \
  -v $(pwd)/src:/code/src \
  -v $(pwd)/../../audio/samples/:/samples \
  -v $(pwd)/outputs:/outputs \
  pachyderm-magenta-js
