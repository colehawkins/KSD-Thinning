#!/bin/bash

FOLDER='.ray_data'

for path in $(ls $FOLDER);
do for kernel in imq rbf;
do python plot_frontier.py --kernel-type $kernel --results-path ${FOLDER}/${path} --metrics 'Agreement' 'Total Variation' $1 $2;
done
done

for path in $(ls $FOLDER);
do for kernel in imq rbf;
do python plot_frontier.py --kernel-type $kernel --results-path ${FOLDER}/${path} --metrics 'Accuracy' 'ECE' $1 $2;
done
done

