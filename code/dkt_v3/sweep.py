import math
import os

import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from args_sweep import parse_args
from src.dataloader import Preprocess
from src.utils import setSeeds
from src.criterion import get_criterion
from src.dataloader import get_loaders
from src.metric import get_metric
from src.model import LSTM, LSTMATTN, Bert
from src.optimizer import get_optimizer
from src.scheduler import get_scheduler
from src import trainer

def run(config=None):
    
    wandb.init(config=config)
    args = parse_args()
    wandb.config.update(args)
    args = wandb.config

    setSeeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # preprocessing
    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()

    train_data, valid_data = preprocess.split_data(train_data)

    model = trainer.get_model(args).to(args.device)

    # training
    train_loader, valid_loader = get_loaders(args, train_data, valid_data)
    
    # only when using warmup scheduler
    args.total_steps = int(math.ceil(len(train_loader.dataset) / args.batch_size)) * (
        args.n_epochs
    )
    args.warmup_steps = args.total_steps // 10

    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    best_auc = -1
    early_stopping_counter = 0
    for epoch in range(args.n_epochs):

        print(f"Start Training: Epoch {epoch + 1}")

        ### TRAIN
        train_auc, train_acc, train_loss = trainer.train(
            train_loader, model, optimizer, scheduler, args
        )

        ### VALID
        auc, acc = trainer.validate(valid_loader, model, args)

        wandb.log(
            {
                "epoch": epoch+1,
                "train_loss_epoch": train_loss,
                "train_auc_epoch": train_auc,
                "train_acc_epoch": train_acc,
                "valid_auc_epoch": auc,
                "valid_acc_epoch": acc,
            }
        )

        if auc > best_auc:
            best_epoch = epoch+1
            best_auc = auc
            best_acc = acc
            best_train_auc = train_auc
            best_train_acc = train_acc
            best_train_loss = train_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                print(
                    f"EarlyStopping counter: {early_stopping_counter} out of {args.patience}"
                )
                break

        # scheduler
        if args.scheduler == "plateau":
            scheduler.step(best_auc)
    
    wandb.log(
        {
            "best_epoch": best_epoch,
            "best_train_loss": best_train_loss,
            "best_train_auc": best_train_auc,
            "best_train_acc": best_train_acc,
            "best_valid_auc": best_auc,
            "best_valid_acc": best_acc,
        }
    )



sweep_config = {
    'name' : 'bayes-test',
    'method': 'bayes',
    'metric' : {
        'name': 'best_valid_auc',
        'goal': 'maximize'   
        },
    'parameters' : {
        'max_seq_len': {
            'min': 5,
            'max': 50
        },
        'embed_dim': {
            'values': [16, 32, 64, 128, 256]
        },
        'hidden_dim': {
            'values': [16, 32, 64, 128, 256]
        },
        'n_layers': {
            'values':[2]
        },
        'n_heads': {
            'values':[2]
        },
        'optimizer': {
            'values': ['adam']
            },
        'dropout': {
            'values': [0.3]
            },
        'lr': {
            'values': [0.1, 0.01, 0.001, 0.0001, 0.00001]
            },
        'n_epochs': {
            'values': [100]
            },
        'patience': {
            'values': [5]
        },
        'batch_size': {
            'values': [16, 32, 64, 128, 256]
            },
        'scheduler': {
            'values': ['plateau']
        }
        }
    }

sweep_id = wandb.sweep(sweep_config, project="sweep_tutorial")
wandb.agent(sweep_id, run, count=1000)