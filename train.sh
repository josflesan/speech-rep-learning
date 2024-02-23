#!/bin/bash

exp=$1

for i in {1..20}; do
    src/train.py --config $exp/train.conf \
	--param $exp/param-$((i-1)) \
        --param-out $exp/param-$i \
	> $exp/log-$i
done

