#!/bin/bash


# main arguments
model_name=FEDformer
input_len=16
pred_len=32
label_len=32
data=NTU
#IMPORTANT: value that heavily change the behaviour of the model
cat_value=0
get_time_value=1
preprocess=1
model_id_name=TEMPLATE

# learning argument 
batch_size=256
learning_rate=3
lradj=sem_constant
epoch=14 
#technical argument. Must be the number of channel that will go into our models ( 75 for NTU)
enc_in=75
dec_in=${enc_in} # must be equal to enc_in
c_out=75 
split_train_test= action 



python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path './dataset/NTU_RGB+D/' \
  --data_path 'numpyed/'\
  --model_id ${model_id_name} \
  --model ${model_name} \
  --data ${data} \
  --features M \
  --seq_len ${input_len} \
  --label_len ${label_len} \
  --pred_len ${pred_len} \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 75 \
  --dec_in 75 \
  --c_out 75 \
  --des 'Exp' \
  --itr 1      \
  --dropout 0.1 \
  --embed timeNTU \
  --get_cat_value ${cat_value} \
  --get_time_value ${get_time_value} \
  --use_gpu 1 \
  --train_epochs ${epoch}\
  --lradj ${lradj} \
  --patience 4 \
  --batch_size ${batch_size}\
  --learning_rate $(echo "scale=10; 10^-${learning_rate}" | bc)\
  --split_train_test ${action} \
  --preprocess ${preprocess}\

#Some possibilities are :
#--augment
#prop 1.0,0.1,0.1,0.1
# --no_test
#--start_checkpoint
#--use_gpu usually True
