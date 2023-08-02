model_name=FEDformer
pred_len=32
data=NTU_leg
batch_size=256

i=3
epoch=14
model_id_name=NTU02-08${model_name}leg-lr-${i}-bs${batch_size}
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
  --label_len ${pred_len} \
  --pred_len ${pred_len} \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 24 \
  --dec_in 24 \
  --c_out 24 \
  --des 'Exp' \
  --itr 1      \
  --dropout 0.1 \
  --embed timeNTU \
  --get_cat_value 0 \
  --get_time_value 1 \
  --use_gpu 1 \
  --train_epochs ${epoch}} \
  --batch_size ${batch_size}\
  --learning_rate $(echo "scale=10; 10^-$i" | bc)\
  --split_train_test action \
  --lradj sem_constant \
  --patience 4 \
  --no_test


data=NTU_body
model_id_name=NTU02-08${model_name}leg-lr-${i}-bs${batch_size}
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
  --label_len ${pred_len} \
  --pred_len ${pred_len} \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 15 \
  --dec_in 15 \
  --c_out 15 \
  --des 'Exp' \
  --itr 1      \
  --dropout 0.1 \
  --embed timeNTU \
  --get_cat_value 0 \
  --get_time_value 1 \
  --use_gpu 1 \
  --train_epochs ${epoch}} \
  --batch_size ${batch_size}\
  --learning_rate $(echo "scale=10; 10^-$i" | bc)\
  --split_train_test action \
  --lradj sem_constant \
  --patience 4  \
  --no_test


data=NTU_arm
model_id_name=NTU02-08${model_name}leg-lr-${i}-bs${batch_size}
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
  --train_epochs ${epoch}} \
  --batch_size ${batch_size}\
  --learning_rate $(echo "scale=10; 10^-$i" | bc)\
  --split_train_test action \
  --lradj sem_constant \
  --patience 4 \
  --no_test

