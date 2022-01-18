#!/bin/bash

#Run HPO using Ray, command line args are for stopping rules
#Used to generate curve model settings


python hpo.py \
	--run-target train_curve.py \
	--grace-period 300 \
	--max-iter 300 \
	--num-samples 1 \
	| tee logs/curve_resnet_hpo.txt

