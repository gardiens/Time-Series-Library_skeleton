#!/bin/bash
#SBATCH --partition=all
#SBATCH --qos=default
#SBATCH --output=logs/training/FEDFormer/outtest1.stdout
#SBATCH --error=logs/training/FEDFormer/errtest1.stderr
#SBATCH --job-name=TFFor1
#SBATCH --gres=gpu:1
#SBATCH --mail-user=pierrick.bournez@student-cs.fr
#SBATCH --mail-type=ALL
MODEL_NAME=FEDformer
LOG_STDOUT="logs/training/FEDFormer/out_test.stdout"
LOG_STDERR="logs/training/FEDFormer/err_test.stderr"
NB_GPU=2

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

which python >> $LOG_STDOUT
echo "---Begininng program---" >> $LOG_STDOUT
echo "Exp name       : test" >> $LOG_STDOUT
echo "Slurm Job ID   : $SLURM_JOB_ID" >> $LOG_STDOUT
echo "SBATCH script  : ?" >> $LOG_STDOUT

bash scripts/training/${MODEL_NAME}_train.sh >> $LOG_STDOUT


wait $!