#!/bin/bash
NB_GPU=1
NOM_RUNS=training

#C'est ici qu'on va submit toutes nos runs avec un sbatch, cela va permettre de sauvegarder ce qu'on fait ou non...
#donnée technique souvent modifiable
PATH_LOG= $PWD/logs/${NOM_RUNS}
PATH_SCRIPT= $PWD/scripts/${NOM_RUNS}
# Create directories if they don't exist
#mkdir -p $PATH_LOG/${MODEL_NAME}
#mkdir -p $PATH_SCRIPT/${MODEL_NAME}


#Moins utile car plus technique
PARTITION=all
QOS=all
MAIL=pierrick.bournez@student-cs.fr
MAIL_TYPE=ALL
TIME=10:00:00






JOB_NAME=${MODEL_NAME}_${NOM_RUNS}

# Changer ici les scripts si on veut!!
MODEL_NAME=Autoformer
mkdir -p $PATH_LOG/${MODEL_NAME}
OUTPUT=${PATH_LOG}/${MODEL_NAME}/output_%j.out
ERROR=${PATH_LOG}/${MODEL_NAME}/error_%j.err

sbatch $PATH_SCRIPT/Autoformer_train.sh --partition=${PARTITION} --qos=${QOS} --mail-user=${MAIL} --mail-type=${MAIL_TYPE} --time=${TIME} --job-name=${JOB_NAME} --output=${OUTPUT} --error=${ERROR} --gres=gpu:${NB_GPU} 
MODEL_NAME=FEDformer
mkdir -p $PATH_LOG/${MODEL_NAME}
OUTPUT=${PATH_LOG}/${MODEL_NAME}/output_%j.out
ERROR=${PATH_LOG}/${MODEL_NAME}/error_%j.err

sbatch $PATH_SCRIPT/FEDformer_train.sh --partition=${PARTITION} --qos=${QOS} --mail-user=${MAIL} --mail-type=${MAIL_TYPE} --time=${TIME} --job-name=${JOB_NAME} --output=${OUTPUT} --error=${ERROR} --gres=gpu:${NB_GPU} 
MODEL_NAME=NTS
mkdir -p $PATH_LOG/${MODEL_NAME}
OUTPUT=${PATH_LOG}/${MODEL_NAME}/output_%j.out
ERROR=${PATH_LOG}/${MODEL_NAME}/error_%j.err

sbatch $PATH_SCRIPT/NTS_train.sh --partition=${PARTITION} --qos=${QOS} --mail-user=${MAIL} --mail-type=${MAIL_TYPE} --time=${TIME} --job-name=${JOB_NAME} --output=${OUTPUT} --error=${ERROR} --gres=gpu:${NB_GPU} 
