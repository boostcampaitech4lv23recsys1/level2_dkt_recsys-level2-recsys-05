# train
python train.py --model lstmattn --model_name model_lstmattn.pt 

python train.py --model lstm --model_name model_lstm.pt 
python train.py --model lstmattn --model_name model_lstmattn.pt 
python train.py --model bert --model_name model_bert.pt 

# inference
python inference.py --model lstmattn --model_name model.pt --output_file_name lstmattn.csv
python inference.py --model bert --model_name model_bert.pt --output_file_name bert.csv

# nohup
nohup python train.py --model lstmattn --model_name model_lstmattn.pt > ./nohup/lstmattn.out