from args import parse_args
import pandas as pd
import numpy as np
import torch
import random
import os
from optuna.samplers import TPESampler
from ML_optuna_utils import setSeeds, load_data
from run_optuna import RunOptuna

def main(args):
    setSeeds(args.seed)
    args.sampler = TPESampler(args.seed)
    h_train_X, h_valid_X, h_train_y, h_valid_y, test_X = load_data(args)
    Optuna = RunOptuna(args, h_train_X, h_valid_X, h_train_y, h_valid_y)
    Optuna.run_optuna()

if __name__ == "__main__":
    args = parse_args()
    main(args)