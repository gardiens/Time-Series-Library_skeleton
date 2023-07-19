#!/bin/bash
model_name=Autoformer
pred_len=32
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path './dataset/NTU_RGB+D/' \
  --data_path 'numpyed/'\
  --model_id NTU_32_32_1run \
  --model $model_name \
  --data NTU \
  --features M \
  --seq_len 96 \
  --label_len $pred_len \
  --pred_len $pred_len \
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
  --get_time_value 1 

