#!/bin/bash

python hpo.py  --problem cardiac \
	--run-target main.py \
	$1
