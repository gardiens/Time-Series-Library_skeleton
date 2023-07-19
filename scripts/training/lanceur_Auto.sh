#!/bin/bash

#SBATCH --partition=all
#SBATCH --qos=default
#SBATCH --output=logs/training/FED/out.stdou
#SBATCH --error=logs/training/FED/err.stderr
#SBATCH --job-name=FEDGF

#SBATCH --gres=gpu:4
MODEL_NAME=Auto
LOG_STDOUT="logs/1run/${MODEL_NAME}/out_$SLURM_JOB_ID.stdout"
LOG_STDERR="logs/1run/${MODEL_NAME}/err_$SLURM_JOB_ID.stderr"

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

bash scripts\training\${MODEL_NAME}_train.sh


wait $!