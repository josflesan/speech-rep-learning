#!/bin/bash

exp=$1

for i in {1..20}; do
	src/train_asr.py \
		--config $exp/train.conf \
		--asr-param $exp/param-$((i-1)) \
		--pred-param-output $exp/param-$i \
		> $exp/log-$i
done
