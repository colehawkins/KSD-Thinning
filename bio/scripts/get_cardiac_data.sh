#!/bin/bash

#Download the data used in several Stein methods papers
DEST='/data/stein_thinning'

mkdir $DEST

cd $DEST

rm -rf cardiac
mkdir cardiac

cd cardiac
#tempered posterior MCMC 
wget https://dataverse.harvard.edu/api/access/datafile/4616786
unzip 4616786

rm -rf 4616786 __MACOSX


