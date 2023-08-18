#!/bin/bash

for model in  Autoformer  Dlinear FEDformer Informer  Pyraformer  FEDWav NTS
do 

model_name=${model}
pred_len=16
label_len=16
data=NTU
batch_size=256
model_id_name=17-08-sanspre-${model_name}
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path './dataset/NTU_RGB+D/' \
  --data_path 'numpyed/'\
  --model_id ${model_id_name} \
  --model ${model_name} \
  --data ${data} \
  --features M \
  --seq_len 16 \
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
  --get_cat_value 0 \
  --get_time_value 1 \
  --use_gpu 1 \
  --train_epochs 10\
  --lradj sem_constant \
  --patience 3 \
  --batch_size ${batch_size}\
  --learning_rate $(echo "scale=10; 10^-3" | bc)\
  --split_train_test action \
  --no_test\
  --preprocess 6
wait 1800

done 
wait 3600
for model in  Autoformer  Dlinear FEDformer Informer  Pyraformer  FEDWav NTS
do 


model_name=${model}
pred_len=32
label_len=32
data=NTU
batch_size=256
model_id_name=17-08-${model_name} #c'est du pre
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path './dataset/NTU_RGB+D/' \
  --data_path 'numpyed/'\
  --model_id ${model_id_name} \
  --model ${model_name} \
  --data ${data} \
  --features M \
  --seq_len 16 \
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
  --get_cat_value 0 \
  --get_time_value 1 \
  --use_gpu 1 \
  --train_epochs 14\
  --lradj sem_constant \
  --patience 4 \
  --batch_size ${batch_size}\
  --learning_rate $(echo "scale=10; 10^-3" | bc)\
  --split_train_test action \
  --no_test\
  --preprocess 1
wait 1800


done 