#!/bin/bash

#Train base resnet models for subspace inference


for seed in 1 2 3;
do	python pretrain.py \
		--seed $seed \
		--num-epochs 300 \
		--batch-size 128 \
		--save-every 5 \
		--lr 1e-1 \
		--wd 1e-3 \
		--save-dir /data/frn_checkpoints/pretrain \
		--problem-setting cifar10;
done
