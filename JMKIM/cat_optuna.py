import pandas as pd
import numpy as np
import datetime as dt
import optuna
import statsmodels.api as sm
import datetime
import re
import ray
import torch
import gc

from collections import defaultdict
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.samplers import TPESampler
from statsmodels.tsa.seasonal import STL
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, ElasticNet, ridge_regression, SGDRegressor, RANSACRegressor, SGDOneClassSVM
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.svm import SVR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, CatBoostRegressor
import lightgbm as lgb

from tqdm import tqdm
import warnings
import random
import os
seed=777
os.environ['PYTHONHASHSEED']=str(seed)
random.seed(seed)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
pd.set_option('mode.chained_assignment',  None)
sampler = TPESampler(seed=seed)

# train_df = pd.read_csv("train_feature_engineering.csv")
# test_df = pd.read_csv("test_feature_engineering.csv")
# total = pd.read_csv("total.csv")

total = pd.read_csv("sequential_total.csv")
test_df = pd.read_csv("sequential_test.csv")

total = total.loc[total["answerCode"] != -1]
test_df = test_df.loc[test_df['answerCode'] == -1]

total = pd.concat([total.iloc[:, 0:9], total.iloc[:, 27:]], axis=1)
train_X = total.drop(['userID', 'answerCode', 'assessmentItemID', 'testId', 'Timestamp'], axis=1)
train_y = total[['userID', 'answerCode']]
test_X = test_df.drop(['userID', 'answerCode', 'assessmentItemID', 'testId', 'Timestamp'], axis=1)

h_train_X, h_valid_X, h_train_y, h_valid_y = train_test_split(train_X, train_y['answerCode'], test_size=0.3, stratify=train_y['answerCode'], random_state=seed)

print(h_train_X.shape, h_train_y.shape, h_valid_X.shape, h_valid_y.shape)

def objective(trial):
    param = {
        "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        # "objective" : "RMSE",
        # "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        "depth": trial.suggest_int("depth", 3, 16),
        # "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "boosting_type": "Plain",
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "learning_rate" : trial.suggest_loguniform("learning_rate", 0.0001, 1.0),
        "n_estimators":trial.suggest_int("n_estimators", 1000, 5000),
        "l2_leaf_reg":trial.suggest_float("l2_leaf_reg",1e-8,3e-5),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "max_bin": trial.suggest_int("max_bin", 200, 500),
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.01, 100.00)

    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)
    
    cat = CatBoostClassifier(**param, cat_features=[0, 1, 2, 3, ], random_seed=seed, task_type="GPU")
    cat_model = cat.fit(h_train_X, h_train_y, eval_set=[(h_valid_X, h_valid_y)], verbose=0, early_stopping_rounds=25)    
    preds = cat_model.predict_proba(h_valid_X)
    # pred_labels = np.rint(preds)
    # accuracy = roc_auc_score(h_valid_y, pred_labels)
    accuracy = roc_auc_score(h_valid_y, preds[:, 1])
    torch.cuda.empty_cache()
    gc.collect()
    return accuracy

study_cat = optuna.create_study(direction="maximize",)
study_cat.optimize(objective, n_trials=100)