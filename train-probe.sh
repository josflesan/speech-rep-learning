#!/bin/bash

exp=$1

for i in {1..10}; do
    src/train_probe.py --config $exp/probe.conf \
        --pred-param-output $exp/pred/param-$i \
	> $exp/pred/log-$i
done
