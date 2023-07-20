#!/bin/bash
sbatch --gres=gpu:0 --time=1:00:00 --output=logs/1run/Auto/CPUout.stdout --error=logs/1run/Auto/CPUerr.stderr --job-name=AutoCPU scripts/1Run_FED_NTU/Autoformers_1run.sh

sbatch --gres=gpu:0 --time=1:00:00 --output=logs/1run/FED/CPUout.stdout --error=logs/1run/FED/CPUerr.stderr --job-name=FEDCPU scripts/1Run_FED_NTU/FEDformers_1run.sh

