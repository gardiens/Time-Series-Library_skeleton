#!/bin/bash

model_name=FEDformer
pred_len=32
label_len=32
data=NTU
batch_size=256
model_id_name=08-08-run-model-100ep
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
  --itr 4      \
  --dropout 0.1 \
  --embed timeNTU \
  --get_cat_value 0 \
  --get_time_value 1 \
  --use_gpu 1 \
  --train_epochs 100\
  --lradj cosine \
  --patience 80 \
  --batch_size ${batch_size}\
  --learning_rate $(echo "scale=10; 10^-3" | bc)\
  --split_train_test action \
  --augment \
  --prop 0.1,0.1,0.1,0.1\

