#!/bin/bash

#Download the data used in several Stein methods papers
DEST='/data/stein_thinning'

mkdir $DEST

cd $DEST

mkdir goodwin
cd goodwin
#Goodwin oscillator MALA gradients
wget https://dataverse.harvard.edu/api/access/datafile/4807854 
mv 4807854 mala_grads.csv
#Goodwin oscillator MALA samples
wget https://dataverse.harvard.edu/api/access/datafile/4807850
mv 4807850 mala_samples.csv

#Goodwin oscillator RWM gradients
wget https://dataverse.harvard.edu/api/access/datafile/4807854 
mv 4807854 mala_grads.csv
#Goodwin oscillator MALA samples
wget https://dataverse.harvard.edu/api/access/datafile/4807850
mv 4807850 mala_samples.csv

mkdir goodwin
cd goodwin
#Goodwin oscillator MALA gradients
wget https://dataverse.harvard.edu/api/access/datafile/4807854 
mv 4807854 mala_grads.csv
#Goodwin oscillator MALA samples
wget https://dataverse.harvard.edu/api/access/datafile/4807850
mv 4807850 mala_samples.csv

cd ..

mkdir cardiac

cd cardiac
#tempered posterior MCMC 
wget https://dataverse.harvard.edu/api/access/datafile/4616786
unzip 4616786

rm -rf 4616786 __MACOSX


