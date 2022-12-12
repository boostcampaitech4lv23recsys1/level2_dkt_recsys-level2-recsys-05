import os

import numpy as np
import torch
import wandb
from sklearn.model_selection import KFold
from args import parse_args
from src import trainer
from src.dataloader import Preprocess
from src.utils import setSeeds


def main(args):
    # wandb.login()

    setSeeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.model == 'tabnet'
        tabnet_run(args)
    
    else:
        preprocess = Preprocess(args)
        preprocess.load_train_data(args.file_name)
        train_data = preprocess.get_train_data()

        # wandb.init(project="dkt-v3", config=vars(args))

        if args.kfold:
            kf = KFold(args.kfold, shuffle=True, random_state=args.seed)
            user_id = train_data['userID'].unique()
            auc = []
            acc = []
            k = 0
            for train_i, valid_i in kf.split(user_id):
                k +=1
                train_userid = user_id[train_i]
                valid_userid = user_id[valid_i]
                train_d = train_data[train_data['userID'].isin(train_userid)].reset_index(drop=True).sort_values(["userID", "Timestamp"])
                valid_d = train_data[train_data['userID'].isin(valid_userid)].reset_index(drop=True).sort_values(["userID", "Timestamp"])

                model = trainer.get_model(args).to(args.device)    
                auc_k, acc_k = trainer.run(args, train_d, valid_d, model, str(k))
                auc.append(auc_k)
                acc.append(acc_k)
            print('-'*50)
            print(f'FOLD AUC : {auc}')
            print(f'MEAN AUC : {np.mean(auc)}  STD AUC : {np.std(auc)}')
            print(f'FOLD ACC : {acc}')
            print(f'MEAN ACC : {np.mean(acc)}  STD ACC : {np.std(acc)}')
            print('-'*50)

        else:
            train_data, valid_data = preprocess.split_data(train_data)
            model = trainer.get_model(args).to(args.device)
            trainer.run(args, train_data, valid_data, model)

    
    


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
