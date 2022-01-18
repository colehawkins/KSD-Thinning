#!/bin/bash

#Download the goodwin oscillator data 

DEST='/data/stein_thinning'
WEBPATH='https://dataverse.harvard.edu/api/access/datafile/'

cd $DEST

mkdir goodwin
cd goodwin
#Goodwin oscillator MALA gradients
SUFFIX=4807854
wget ${WEBPATH}${SUFFIX} 
mv $SUFFIX mala_grads.csv

#Goodwin oscillator MALA samples
SUFFIX=4807850
wget ${WEBPATH}${SUFFIX}
mv $SUFFIX mala_samples.csv

#Goodwin oscillator RWM gradients
SUFFIX=4807838
wget ${WEBPATH}${SUFFIX}
mv $SUFFIX  rwm_grads.csv

#Goodwin oscillator RWM samples
SUFFIX=4807844
wget ${WEBPATH}${SUFFIX}
mv $SUFFIX  rwm_samples.csv
