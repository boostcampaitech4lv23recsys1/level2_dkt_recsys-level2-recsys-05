# train
python train.py --model lstmattn --model_name model_lstmattn_new.pt 
python train.py --model bert --model_name model_bert_new.pt 
python train.py --model bert --model_name model_bert_stride_20.pt 
python train.py --model bert --model_name model_bert_stride_1.pt 
python train.py --model bert --model_name model_bert_decomposition.pt 
python train.py --model bert --model_name model_bert_decomposition_all.pt 

python train.py --model bert --model_name model_bert_best --batch_size 256 --drop_out 0.2 --embed_dim 16 --hidden_dim 256 --max_seq_len 7 --n_heads 4 --n_layers 10 --kfold 5
python inference.py --model bert --model_name model_bert_best --batch_size 256 --drop_out 0.2 --embed_dim 16 --hidden_dim 256 --max_seq_len 7 --n_heads 4 --n_layers 10 --kfold 5 --output_file_name model_bert_best_kfold.csv
# 0.8178731143235964
# 0.8418934825519276 0.8305820163094465 0.8225809365515248 0.8159330915828863 0.8280639842642677 -> 0.8278107022520107 0.00865444531842082

python train.py --model lstmattn --model_name model_lstmattn_best.pt --batch_size 256 --drop_out 0.4 --embed_dim 16 --hidden_dim 512 --max_seq_len 6 --n_heads 8 --n_layers 1 
# 0.8242673190087738

python train.py --model lstm --model_name model_lstm_best.pt --batch_size 512 --drop_out 0.2 --embed_dim 64 --hidden_dim 512 --max_seq_len 8 --n_layers 1
# 0.820780081735384

# inference
python inference.py --model lstmattn --model_name model.pt --output_file_name lstmattn.csv
python inference.py --model lstmattn --model_name model_lstmattn_lastvalid.pt --output_file_name lstmattn_lastvalid.csv
python inference.py --model bert --model_name model_bert_valid_test.pt --output_file_name bert_valid_test.csv
python inference.py --model bert --model_name model_bert_stride_20.pt --output_file_name bert_stride_20.csv
python inference.py --model bert --model_name model_bert_decomposition.pt --output_file_name model_bert_decomposition.csv

python inference.py --model lstm --model_name model_lstm_best.pt --batch_size 512 --drop_out 0.2 --embed_dim 64 --hidden_dim 512 --max_seq_len 8 --n_layers 1 --output_file_name model_lstm_best.csv
python inference.py --model lstmattn --model_name model_lstmattn_best.pt --batch_size 256 --drop_out 0.4 --embed_dim 16 --hidden_dim 512 --max_seq_len 6 --n_heads 8 --n_layers 1 --output_file_name model_lstmattn_best.csv
python inference.py --model bert --model_name model_bert_best.pt --batch_size 256 --drop_out 0.2 --embed_dim 16 --hidden_dim 256 --max_seq_len 7 --n_heads 4 --n_layers 10 --output_file_name model_bert_best.csv

# nohup
nohup python train.py --model lstmattn --model_name model_lstmattn_lastvalid.pt > ./nohup/lstmattn_lastvalid.out
nohup python train.py --model bert --model_name model_bert_valid_test.pt > ./nohup/bert_valid_test.out

# sweep 
wandb sweep sweep.yaml

# gpu
gpustat -i

# kill
ps -ef | grep nohup
kill -9 [PID1 PID2]
72314 72778