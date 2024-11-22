export CUDA_VISIBLE_DEVICES=0

#cd ..

for model in FEDformer
do

for preLen in 96
do

# ETT m1
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/elect/ \
  --data_path predict_450kW.csv \
  --task_id elect \
  --model $model \
  --data custom \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 512 \
  --itr 3 \


done




done

