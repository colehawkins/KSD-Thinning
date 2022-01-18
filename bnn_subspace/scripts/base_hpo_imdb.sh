#!/bin/bash

#Run HPO using Ray, command line args are for stopping rules
#Used to generate SWA resnet settings


python hpo.py \
	--run-target pretrain.py \
	--problem-setting imdb \
	--grace-period 30 \
	--max-iter 30 \
	--num-samples 1 \
	| tee logs/swa_cnn_lstm_hpo.txt

