#!/bin/bash

GOODWIN_PATH='results/main/goodwin_Nov-05-2021.pkl'
CARDIAC_PATH='results/main/cardiac_Nov-05-2021.pkl'


for path in $CARDIAC_PATH;
do for kernel_type in imq rbf;
do for sampler_type in rwm;
do python plot_curves.py --kernel-type $kernel_type --sample-generation $sampler_type --results-path $path $1 $2;
done
done
done 

for path in $GOODWIN_PATH;
do for kernel_type in imq rbf;
do for sampler_type in rwm mala;
do python plot_curves.py --kernel-type $kernel_type --sample-generation $sampler_type --results-path $path $1 $2;
done
done
done 

