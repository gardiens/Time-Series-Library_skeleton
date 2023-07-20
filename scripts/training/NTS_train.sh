#!/bin/bash
model_name=Nonstationary_Transformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path './dataset/NTU_RGB+D/' \
  --data_path 'numpyed/'\
  --model_id NTU_32_32_training_${model_name} \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 32 \
  --label_len 32 \
  --pred_len 32 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 75 \
  --dec_in 75 \
  --c_out 75 \
  --des 'Exp' \
  --itr 1 \
  --p_hidden_dims 256 256 \
  --p_hidden_layers 2 \
  --get_cat_value 0 \
  --get_time_value 1 \ 
