#!/bin/bash

sh /bash_scripts/tensorboard.sh &  PIDIOS=$!
sh /bash_scripts/jupyter.sh &  PIDMIX=$!
wait $PIDIOS
wait $PIDMIX
