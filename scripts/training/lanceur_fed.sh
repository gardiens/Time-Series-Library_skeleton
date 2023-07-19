#!/bin/bash


MODEL_NAME=FEDformer
LOG_STDOUT="logs/training/${MODEL_NAME}/out_$SLURM_JOB_ID.stdout"
LOG_STDERR="logs/training/${MODEL_NAME}/err_$SLURM_JOB_ID.stderr"
NB_GPU=2
#SBATCH --partition=all
#SBATCH --qos=default
#SBATCH --output=logs/training/${MODEL_NAME}/out.stdout
#SBATCH --error=logs/training/${MODEL_NAME}/err.stderr
#SBATCH --job-name=${MODEL_NAME}GF1
#SBATCH --gres=gpu:${NB_GPU}

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