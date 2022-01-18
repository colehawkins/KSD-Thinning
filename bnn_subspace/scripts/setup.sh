#!/bin/bash

python -m spacy download en_core_web_sm

for folder in 'frn_checkpoints' 'cnn_lstm_checkpoints';
do mkdir /data/${folder};
		for subfolder in 'pretrain' 'sampling' 'curve';
				do mkdir /data/${folder}/${subfolder};
				done
		done
mkdir results
mkdir results/linked_dicts
