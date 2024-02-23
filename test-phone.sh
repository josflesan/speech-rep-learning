#!/bin/bash

exp=$1
epoch=$2

src/predict_phone.py \
       	--config $exp/phone/evaluate.conf \
       	--pred-param $exp/phone/param-$epoch \
       	--apc-param $exp/param-20 \
	> $exp/phone/eval-epoch-$epoch.log
