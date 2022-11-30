# train
python train.py --model lstmattn --model_name model_lstmattn_new.pt 
python train.py --model bert --model_name model_bert_new.pt 
python train.py --model bert --model_name model_bert_stride_20.pt 
python train.py --model bert --model_name model_bert_stride_1.pt 
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