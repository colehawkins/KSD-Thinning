#!/bin/bash

FOLDER='.ray_data'

#for file in $(ls $FOLDER);
#		do python plot_curves.py --results-path ${FOLDER}/${file} $1 $2;
#done

for file in $(ls $FOLDER);
do for kernel_type in imq rbf;
		do python plot_curves.py --results-path ${FOLDER}/${file} --kernel-types $kernel_type $1 $2;
done
done
