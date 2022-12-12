import pandas as pd
import torch
import optuna
import wandb

from optuna.integration.wandb import WeightsAndBiasesCallback
from config import CFG, logging_conf
from lightgcn.datasets import prepare_dataset
from lightgcn.models_optuna import build, train
from lightgcn.utils import class2dict, get_logger

logger = get_logger(logging_conf)
use_cuda = torch.cuda.is_available() and CFG.use_cuda_if_available
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

# wandb_kwargs = {"project": "lightgcn_optuna_2"}
wandb_kwargs = {"project": "lightgcn_context_optuna"}
wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)

@wandbc.track_in_wandb()
def objective(trial):#, n_node, train_data, valid_data, device):
    params = {
              'embedding_dim' : trial.suggest_int("embedding_dim", 10, 128),
              'num_layers' : trial.suggest_int("num_layers", 1, 10),
              'alpha' : trial.suggest_loguniform("alpha", 1e-3, 9e-1),
              'n_epoch' : trial.suggest_int("n_epoch", 300, 3000),
              'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 9e-1),
            #   'batch_size' : trial.suggest_int("batch_size", 64, 512, 16),
              'batch_size' : 512
              }
    # params = {'embedding_dim': 123, 'num_layers': 9, 'alpha': 0.0797271712810023, 'n_epoch': 2711, 'learning_rate': 0.00217008368273351}
    build_kwargs = {
    }
    print(params)
    model = build(batch_size=params['batch_size'],
                  n_node=n_node,
                  embedding_dim=params['embedding_dim'],
                  num_layers=params['num_layers'],
                  alpha=params['alpha'],
                  logger=logger.getChild("build"),
                  **build_kwargs)
    model.to(device)
    auc = train(model,
                     train_data,
                     valid_data,
                     n_epoch=params['n_epoch'],
                     learning_rate=params['learning_rate'],
                     use_wandb=CFG.user_wandb,
                     weight=CFG.weight_basepath,
                     logger=logger.getChild("train"),
                     batch_size=params['batch_size'],
                    device=device,
                    #  params=params,
    )
    return auc

def main():
    global train_data, valid_data, n_node, device
    logger.info("Task Started")

    logger.info("[1/1] Data Preparing - Start")
    train_data, test_data, valid_data, n_node = prepare_dataset(
        device, CFG.basepath, verbose=CFG.loader_verbose, logger=logger.getChild("data")
    )
    logger.info("[1/1] Data Preparing - Done")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    # study.optimize(lambda trial: objective(trial, n_node, train_data, valid_data, device), n_trials=100, callbacks=[wandbc])
    study.optimize(objective, n_trials=100, callbacks=[wandbc])
    trial = study.best_trial
    trial_params = trial.params
    print('Best Trial: score {},\nparams {}'.format(trial.value, trial_params))

if __name__ == "__main__":
    main()
