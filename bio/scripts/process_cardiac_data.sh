#!/bin/bash

#Transpose downloaded files for easier loading
#Removes old processed cardiac data and reprocesses

cd /data/stein_thinning/cardiac

echo $(ls)

POSTERIOR_TYPE='Tempered_posterior'

cd $POSTERIOR_TYPE
for seed in '1' '2';
do cd "seed_${seed}"; rm processed_*.csv;
		for filename in $(ls *);
		do echo $filename; csvtool transpose  $filename  > processed_${filename};
#		do echo $filename; perl -pe 's{,}{++$n % 38 == 0 ? "\n" : $&}ge' $filename  > processed_${filename};
		done;
	cd ..;
done
