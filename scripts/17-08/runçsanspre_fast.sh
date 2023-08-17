

model_name=LightTS
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/NTU_RGB+D/ \
  --data_path numpyed/ \
  --model_id ${model_name}-pre \
  --model $model_name \
  --data NTU \
  --features M \
  --embed timeNTU \
  --seq_len 16 \
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
  --train_epochs 14 \
  --use_gpu 1 \
  --batch_size 64 \
  --learning_rate $(echo "scale=10; 10^-3" | bc) \
  --preprocess 1 \
  --no_test
wait 1800
model_name=Reformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/NTU_RGB+D/ \
  --data_path numpyed/ \
  --model_id ${model_name}-pre \
  --model $model_name \
  --data NTU \
  --features M \
  --embed timeNTU \
  --seq_len 16 \
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
  --train_epochs 14 \
  --use_gpu 1 \
  --batch_size 64 \
  --learning_rate $(echo "scale=10; 10^-3" | bc) \
  --preprocess 1 \
  --no_test

wait 1800
model_name=ETSformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/NTU_RGB+D/ \
  --data_path numpyed/ \
  --model_id ${model_name}-pre \
  --model $model_name \
 --data NTU \
  --features M \
  --embed timeNTU \
  --seq_len 16 \
  --label_len 32 \
  --pred_len 32 \
  --e_layers 2 \
  --d_layers 2 \
  --factor 3 \
  --enc_in 75 \
  --dec_in 75 \
  --c_out 75 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 14 \
  --use_gpu 1 \
  --batch_size 64 \
  --learning_rate $(echo "scale=10; 10^-3" | bc) \
  --preprocess 1 \
  --no_test


wait 1800

model_name=PatchTST

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/NTU_RGB+D/ \
  --data_path numpyed/ \
  --model_id ${model_name}-pre \
  --model $model_name \
 --data NTU \
  --features M \
  --embed timeNTU \
  --seq_len 16 \
  --label_len 32 \
  --pred_len 32 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 75 \
  --dec_in 75 \
  --c_out 75 \
  --des 'Exp' \
  --batch_size 16 \
  --itr 1 \
  --train_epochs 14 \
  --use_gpu 1 \
  --batch_size 64 \
  --learning_rate $(echo "scale=10; 10^-3" | bc) \
  --preprocess 1 \
  --no_test

wait 1800
model_name=Pyraformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/NTU_RGB+D/ \
  --data_path numpyed/ \
  --model_id ${model_name}-pre \
  --model $model_name \
 --data NTU \
  --features M \
  --embed timeNTU \
  --seq_len 16 \
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
  --train_epochs 14 \
  --use_gpu 1 \
  --batch_size 64 \
  --learning_rate $(echo "scale=10; 10^-3" | bc) \
  --preprocess 1 \
  --no_test

wait 1800

model_name=MICN

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/NTU_RGB+D/ \
  --data_path numpyed/ \
  --model_id ${model_name}-pre \
  --model $model_name \
 --data NTU \
  --features M \
  --embed timeNTU \
  --seq_len 16 \
  --label_len 32 \
  --pred_len 32 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 75 \
  --dec_in 75 \
  --c_out 75 \
  --d_model 256 \
  --d_ff 512 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 14 \
  --use_gpu 1 \
  --batch_size 64 \
  --learning_rate $(echo "scale=10; 10^-3" | bc) \
  --preprocess 1 \
  --no_test
wait 1800
model_name=Crossformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/NTU_RGB+D/ \
  --data_path numpyed/ \
  --model_id ${model_name}-pre \
  --model $model_name \
 --data NTU \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 14 \
  --use_gpu 1 \
  --batch_size 64 \
  --learning_rate $(echo "scale=10; 10^-3" | bc) \
  --preprocess 1 \
  --no_test


wait 1800
model_name=FiLM

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/NTU_RGB+D/ \
  --data_path numpyed/ \
  --model_id ${model_name}-pre \
  --model $model_name \
 --data NTU \
  --features M \
  --embed timeNTU \
  --seq_len 16 \
  --label_len 32 \
  --pred_len 32 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 75 \
  --dec_in 75 \
  --c_out 75 \
  --des 'Exp' \
  --batch_size 4 \
  --itr 1 \
  --train_epochs 14 \
  --use_gpu 1 \
  --batch_size 64 \
  --learning_rate $(echo "scale=10; 10^-3" | bc) \
  --preprocess 1 \
  --no_test