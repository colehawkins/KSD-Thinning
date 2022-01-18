#!/bin/bash

#Train curve midpoint for SWA resnets

ENDPOINT_1='/data/cnn_lstm_checkpoints/pretrain/epoch_29_seed_1'
ENDPOINT_2='/data/cnn_lstm_checkpoints/pretrain/epoch_29_seed_2'

python train_curve.py \
	--seed 1 \
	--num-epochs 30 \
	--lr 1e-2 \
	--wd 1e-4 \
	--save-dir /data/cnn_lstm_checkpoints/curve \
	--problem-setting imdb \
	--endpoint-1 $ENDPOINT_1 \
	--endpoint-2 $ENDPOINT_2 \
	--save-every 10 \
	$1

ENDPOINT_2='/data/cnn_lstm_checkpoints/pretrain/epoch_29_seed_3'

python train_curve.py \
	--seed 2 \
	--num-epochs 30 \
	--lr 1e-2 \
	--wd 1e-4 \
	--save-dir /data/cnn_lstm_checkpoints/curve \
	--problem-setting imdb \
	--endpoint-1 $ENDPOINT_1 \
	--endpoint-2 $ENDPOINT_2 \
	--save-every 10 \
	$1
