#!/bin/bash

#SBATCH --partition=all
#SBATCH --qos=default
#SBATCH --output=logs/1run/FED/out.stdou
#SBATCH --error=logs/1run/FED/err.stderr
#SBATCH --job-name=FED_bournez

#SBATCH --gres=gpu:1

LOG_STDOUT="logs/1run/FED/out_$SLURM_JOB_ID.stdout"
LOG_STDERR="logs/1run/FED/err_$SLURM_JOB_ID.stderr"

function restart
{
    echo "Calling restart" >> $LOG_STDOUT
    scontrol requeue $SLURM_JOB_ID
    echo "Scheduled job for restart" >> $LOG_STDOUT
}

function ignore
{
    echo "Ignored SIGTERM" >> $LOG_STDOUT
}
trap restart USR1
trap ignore TERM

# start or restart experiment
data >> $LOG_STDOUT
which python >> $LOG_STDOUT
echo "---Begininng program---" >> $LOG_STDOUT
echo "Exp name       : test" >> $LOG_STDOUT
echo "Slurm Job ID   : $SLURM_JOB_ID" >> $LOG_STDOUT
echo "SBATCH script  : ?" >> $LOG_STDOUT
bash conda activate tslib_env
bash scripts\1Run_FED_NTU\FEDformers_1run.sh

wait $!