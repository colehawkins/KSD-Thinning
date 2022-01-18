#!/bin/bash

#Run baseline MCMC chain
#Launches both long training runs simultaneously on one GPU
#Requires ~18GB memory

for seed in 123 1234;
	do python sample.py --seed $seed \
		--problem-setting imdb\
		--num-steps 20000 \
		--save-every 1000 \
		--eval-every 100 \
		--lr 3e-1 \
		--temperature 1.0 \
		--batch-size 256 \
		--save-dir /data/cnn_lstm_checkpoints/sampling \
		--curve-checkpoint-path /data/cnn_lstm_checkpoints/curve/epoch_29_seed_1 | tee logs/imdb_run_seed_${seed} &
	done
