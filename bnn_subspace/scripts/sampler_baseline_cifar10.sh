#!/bin/bash

#Run an MCMC sampler

for seed in 123456;
	do python sample.py --seed $seed \
		--num-steps 20000 \
		--save-every 1000 \
		--eval-every 1000 \
		--lr 3e-1 \
		--temperature 1.0 \
		--batch-size 256 \
		--save-dir /data/frn_checkpoints/sampling \
		--curve-checkpoint-path /data/frn_checkpoints/curve/epoch_299_seed_1;
	done
