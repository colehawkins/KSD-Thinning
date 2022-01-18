#!/bin/bash 

#Create a subset of the cardiac model data for smoke test loading

cd /data/stein_thinning/cardiac/Tempered_posterior/seed_1


NUM_ENTRIES=$((987*38*10))
#echo $NUM_ENTRIES
#echo $(ls )
head -c $NUM_ENTRIES THETA_unique_seed_1_temp_8.csv > smoke_test_theta.csv
head -c $NUM_ENTRIES GRAD_unique_seed_1_temp_8.csv > smoke_test_grad.csv

