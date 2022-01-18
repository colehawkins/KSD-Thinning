#!/bin/bash

#Train base resnet models for subspace inference


for seed in 1 2 3;
do	python pretrain.py \
		--seed $seed \
		--num-epochs 30 \
		--batch-size 128 \
		--save-every 50 \
		--lr 1e-1 \
		--wd 1e-3 \
		--save-dir /data/cnn_lstm_checkpoints/pretrain \
		--problem-setting imdb;
done
