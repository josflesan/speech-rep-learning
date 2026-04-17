#!/bin/bash

exp=$1

for i in {1..5}; do
    src/train.py --config $exp/train.conf \
        --param $exp/param-$((i-1)) \
        --param-output $exp/param-$i
done
