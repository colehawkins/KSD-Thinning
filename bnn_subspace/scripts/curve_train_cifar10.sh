#!/bin/bash

#Train curve midpoint for SWA resnets

ENDPOINT_1='/data/frn_checkpoints/pretrain/epoch_299_seed_1'
ENDPOINT_2='/data/frn_checkpoints/pretrain/epoch_299_seed_2'

python train_curve.py \
	--seed 1 \
	--num-epochs 300 \
	--lr 1e-2 \
	--wd 1e-4 \
	--save-dir /data/frn_checkpoints/curve \
	--endpoint-1 $ENDPOINT_1 \
	--endpoint-2 $ENDPOINT_2 \
	--save-every 10 \
	$1

ENDPOINT_2='/data/frn_checkpoints/pretrain/epoch_299_seed_3'

python train_curve.py \
	--seed 2 \
	--num-epochs 300 \
	--lr 1e-2 \
	--wd 1e-4 \
	--save-dir /data/frn_checkpoints/curve \
	--endpoint-1 $ENDPOINT_1 \
	--endpoint-2 $ENDPOINT_2 \
	--save-every 10 \
	$1
