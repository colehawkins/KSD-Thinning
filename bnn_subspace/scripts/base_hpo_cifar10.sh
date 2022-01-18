#!/bin/bash

#Run HPO using Ray, command line args are for stopping rules
#Used to generate SWA resnet settings


python hpo.py \
	--run-target pretrain.py \
	--problem-setting cifar10 \
	--grace-period 300 \
	--max-iter 300 \
	--num-samples 1 \
	| tee logs/swa_resnet_hpo.txt

