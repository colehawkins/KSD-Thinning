#!/bin/bash

#install the prerequisites
conda env update -f requirements.yml

conda activate KSDP
#install the ksdp pruning package
cd ksdp
pip install -e .
