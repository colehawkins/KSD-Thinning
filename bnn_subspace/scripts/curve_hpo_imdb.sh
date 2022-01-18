#!/bin/bash

#Run HPO using Ray, command line args are for stopping rules
#Used to generate curve model settings


python hpo.py \
	--run-target train_curve.py \
	--problem-setting imdb \
	--grace-period 30 \
	--max-iter 30 \
	--num-samples 1 \
	| tee logs/curve_cnn_lstm_hpo.txt

