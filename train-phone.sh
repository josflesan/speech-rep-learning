#!/bin/bash

exp=$1

for i in {1..10}; do
    src/train_phone.py --config $exp/phone/train.conf \
	--apc-param $exp/param-20 \
        --pred-param $exp/phone/param-$((i-1)) \
        --pred-param-output $exp/phone/param-$i \
	> $exp/phone/log-$i
done
