# train
python train.py --model lstmattn --model_name model_lstmattn_new.pt 
python train.py --model bert --model_name model_bert_new.pt 
python train.py --model bert --model_name model_bert_stride_20.pt 
python train.py --model bert --model_name model_bert_stride_1.pt 
python train.py --model lastquery --model_name model_lastquery.pt --max_seq_len 8 --hidden_dim 64 --batch_size 34 --lr 0.0006008111834559934 --optimizer adamW --patience 20 --n_epochs 300

# inference
python inference.py --model lstmattn --model_name model.pt --output_file_name lstmattn.csv
python inference.py --model lstmattn --model_name model_lstmattn_lastvalid.pt --output_file_name lstmattn_lastvalid.csv
python inference.py --model bert --model_name model_bert_valid_test.pt --output_file_name bert_valid_test.csv
python inference.py --model bert --model_name model_bert_stride_20.pt --output_file_name bert_stride_20.csv

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