#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

model_name=FEDformer
pred_len=64
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path './dataset/NTU_RGB+D/' \
  --data_path 'numpyed/'\
  --model_id NTU_32_64_training_${model_name} \
  --model $model_name \
  --data NTU \
  --features M \
  --seq_len 32 \
  --label_len ${pred_len} \
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
  > "$OUTPUT" 2>"$ERROR"
  

