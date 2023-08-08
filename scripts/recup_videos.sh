#!/bin/bash

name_training=$1 # first argument should be the name of trainig like 07-08
name_runs=$2 # second argument should be the name of model like 07-08-pred-len-32

scp -r slurm-ext:./test_results/*${name_runs}* ./test_results/${name_training}/