#!/bin/bash

#SBATCH --partition=all
#SBATCH --qos=default
#SBATCH --output=fed_out.stdout
#SBATCH --error=fed_err.stderr
#SBATCH --job-name=fed1
#SBATCH --gres=gpu:2

LOG_STDOUT="fed_out.stdout"
LOG_STDERR="fed_err.stderr"


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

echo "---START---" >> $LOG_STDOUT

data >> $LOG_STDOUT

which python >> $LOG_STDOUT

conda activate tslib_env

bash scripts/training/FEDformer_train.sh  >> $LOG_STDOUT

conda deactivate

echo "---END---" >> $LOG_STDOUT

echo

wait $!