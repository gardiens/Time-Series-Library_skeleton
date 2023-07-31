#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

model_name=FEDWav
pred_len=32
model_id_name=NTU-16-16-training27-7${model_name}
data=NTU
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1\
  --root_path './dataset/NTU_RGB+D/' \
  --data_path 'numpyed/'\
  --model_id ${model_id_name} \
  --model ${model_name} \
  --data ${data} \
  --features M \
  --seq_len 16 \
  --label_len ${pred_len} \
  --pred_len ${pred_len} \
  --e_layers 3 \
  --d_layers 3 \
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
  --train_epochs 10