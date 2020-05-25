#!/bin/bash

sh /notebooks/src/bash_scripts/tensorboard.sh &  PIDIOS=$!
sh /notebooks/src/bash_scripts/jupyter.sh &  PIDMIX=$!
wait $PIDIOS
wait $PIDMIX
