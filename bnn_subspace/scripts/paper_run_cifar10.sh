#!/bin/bash

#Run HPO using Ray, command line args are for stopping rules


#CIFAR paper run
python hpo.py \
	--run-target sample.py \
	--problem-setting cifar10 paper \
	--num-samples 1 \
	$1 | tee logs/cifar_paper_run.txt
