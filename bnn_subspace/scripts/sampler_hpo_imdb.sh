#!/bin/bash

#Run HPO using Ray, command line args are for stopping rules
#Used to generate SWA resnet settings


python hpo.py \
	--run-target sample.py \
	--problem-setting imdb hpo \
	--grace-period 1000 \
	--max-iter 1000 \
	--num-samples 1 \
	$1 
