#!/bin/bash
model_name=Meta
pred_len=32
data=NTU

batch_size=256

i=3
epoch=14
model_id_name=NTU02-08FEDformerleg-lr-${i}-bs${batch_size}
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path './dataset/NTU_RGB+D/' \
  --data_path 'numpyed/'\
  --model_id ${model_id_name} \
  --model Meta \
  --data ${data} \
  --features M \
  --seq_len 16 \
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
  --use_gpu 1 \
  --train_epochs ${epoch} \
  --batch_size ${batch_size}\
  --learning_rate $(echo "scale=10; 10^-$i" | bc)\
  --split_train_test action \
  --lradj sem_constant \
  --patience 4 \
  --no_test
