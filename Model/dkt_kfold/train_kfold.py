import os

import torch
import wandb
from args import parse_args
from src import trainer_kfold
from src.dataloader import Preprocess
from src.utils import setSeeds


def main(args):
    # wandb.login()

    setSeeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()

    # train_data, valid_data = preprocess.split_data(train_data)

    # kfold
    # wandb.init(project="dkt", name=f"{args.model}", config=vars(args))
    # wandb.config.update(args)
    model = trainer_kfold.get_model(args).to(args.device)
    trainer_kfold.run_kfold(args, train_data, model)

    # train full
    # model = trainer_kfold.get_model(args).to(args.device)
    # trainer_kfold.run_full(args, train_data, model)
    

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
