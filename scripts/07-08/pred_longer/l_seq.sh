#!/bin/bash

for i in 1 2 3 4 5 6 7 8 9 10
do 

  model_name=FEDformer
  pred_len=$((32 * i))
  label_len=32
  data=NTU
  batch_size=256
  model_id_name=07-08-pred-len-${pred_len}
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
    --batch_size ${batch_size}\
    --learning_rate $(echo "scale=10; 10^-3" | bc)\
    --split_train_test action 


done 

