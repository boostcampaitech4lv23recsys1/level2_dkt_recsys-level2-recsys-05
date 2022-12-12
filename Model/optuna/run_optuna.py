import torch
import gc
import numpy as np
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, classification_report

import lightgbm as lgb
from sklearn.linear_model import LinearRegression, ElasticNet, SGDOneClassSVM
from sklearn.naive_bayes import BernoulliNB
from catboost import CatBoostClassifier, CatBoostRegressor

class RunOptuna():
    def __init__(self, args, h_train_X, h_valid_X, h_train_y, h_valid_y):
        self.args = args
        self.h_train_X = h_train_X
        self.h_valid_X = h_valid_X
        self.h_train_y = h_train_y
        self.h_valid_y = h_valid_y

    def run_optuna(self):
        if self.args.model == "LinearReg":
            self.linearReg_optuna()
        if self.args.model == "ElasticNet":
            self.elastic_optuna()
        if self.args.model == "SGDOneClassSVM":
            self.sgboneclass_svm()
        if self.args.model == "BernoulliNB":
            self.bernoullinb()
        if self.args.model == "LGBMReg":
            self.lgbmreg()
        if self.args.model == "LGBMClf":
            self.lgbmclf()
        if self.args.model == "CatBoostReg":
            self.catreg()
        if self.args.model == "CatBoostClf":
            self.catclf()

    def linearReg_optuna(self):
        m1 = LinearRegression()
        m1.fit(self.h_train_X, self.h_train_y)
        p1 = m1.predict(self.h_valid_X)
        
        print(accuracy_score(self.h_valid_y, np.where(p1 > 0.5, 1, 0)))
        print(roc_auc_score(self.h_valid_y, p1))
        print(classification_report(self.h_valid_y, np.where(p1 > 0.5, 1, 0)))
    
    def elastic_optuna(self):
        def objective(trial):
            param = {
                'tol' : trial.suggest_uniform('tol' , 1e-6 , 1.0),
                'max_iter' : trial.suggest_int('max_iter', 1000, 10000),
                'selection' : trial.suggest_categorical('selection' , ['cyclic','random']),
                'l1_ratio' : trial.suggest_uniform('l1_ratio' , 1e-6 , 1.0),
                'alpha' : trial.suggest_uniform('alpha' , 1e-6 , 2.0),
            }
            model = ElasticNet(**param, random_state=self.args.seed)
            ElasticNet_model = model.fit(self.h_train_X, self.h_train_y)
            preds = ElasticNet_model.predict(self.h_valid_X)
            loss = roc_auc_score(self.h_valid_y, preds)
            return loss
        study = optuna.create_study(direction='maximize', sampler=self.args.sampler)
        study.optimize(objective, n_trials=1000)
        trial = study.best_trial
        trial_params = trial.params
        print('Best Trial: score {},\nparams {}'.format(trial.value, trial_params))

    def sgboneclass_svm(self):
        def objective(trial):
            param = {
                'nu' : trial.suggest_uniform('nu' , 0.0 , 1.0),
                'max_iter' : trial.suggest_int('max_iter', 1000, 10000),
                'tol' : trial.suggest_uniform('tol' , 1e-6 , 1.0),
                'learning_rate' : trial.suggest_categorical('learning_rate' , ['constant','optimal', 'invscaling', 'adaptive']),
                'eta0' : trial.suggest_uniform('eta0' , 1e-6 , 1.0),
                'power_t' : trial.suggest_uniform('power_t' , 1e-6 , 1.0)
            }
            model = SGDOneClassSVM(**param, random_state=self.args.seed)
            SGDOneClass_model = model.fit(self.h_train_X, self.h_train_y)
            preds = SGDOneClass_model.predict(self.h_valid_X)
            loss = roc_auc_score(self.h_valid_y, preds)
            return loss
        study = optuna.create_study(direction='maximize', sampler=self.args.sampler)
        study.optimize(objective, n_trials=1000)
        trial = study.best_trial
        trial_params = trial.params
        print('Best Trial: score {},\nparams {}'.format(trial.value, trial_params))

    def bernoullinb(self):
        def objective(trial):
            param = {
                'alpha' : trial.suggest_uniform('alpha' , 0.0 , 1.0),
            }
            model = BernoulliNB(**param)
            Bernoulli_model = model.fit(self.h_train_X, self.h_train_y)
            preds = Bernoulli_model.predict(self.h_valid_X)
            pred_labels = np.rint(preds)
            loss = roc_auc_score(self.h_valid_y, pred_labels)
            return loss
        study = optuna.create_study(direction='maximize', sampler=self.args.sampler)
        study.optimize(objective, n_trials=1000)
        trial = study.best_trial
        trial_params = trial.params
        print('Best Trial: score {},\nparams {}'.format(trial.value, trial_params))
    
    def lgbmreg(self):
        def objective(trial):
            param = {
                # 'objective': 'binary', # 이진 분류
                "objective": trial.suggest_categorical("objective", ["binary", "cross_entropy"]),
                'verbose': -1,
                'metric': 'AUC',
                'max_depth': trial.suggest_int('max_depth',3, 15),
                'learning_rate': trial.suggest_loguniform("learning_rate", 0.001, 1.0),
                'n_estimators': trial.suggest_int('n_estimators', 100, 10000),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                # 'subsample': trial.suggest_loguniform('subsample', 0.4, 1),
                'lambda_l1' : trial.suggest_loguniform('lambda_l1', 1e-8, 1e-4),
                'lambda_l2' : trial.suggest_loguniform('lambda_l2', 1e-8, 1e-4),
                'path_smooth' : trial.suggest_loguniform('path_smooth', 1e-8, 1e-3),
                'num_leaves' : trial.suggest_int('num_leaves', 30, 200),
                'min_data_in_leaf' : trial.suggest_int('min_data_in_leaf', 10, 100),
                'max_bin' : trial.suggest_int('max_bin', 100, 255),
                'feature_fraction' : trial.suggest_uniform('feature_fraction', 0.5, 0.9),
                'bagging_fraction' : trial.suggest_uniform('bagging_fraction', 0.5, 0.9),
                # 'device' : 'gpu',
                # 'reg_alpha' : None,
            }
            categorical = [0, 1, 2, 3, ]
            model = lgb.LGBMRegressor(**param, categorical_feature=categorical, random_state=self.args.seed)
            lgb_model = model.fit(self.h_train_X, self.h_train_y, eval_set=[(self.h_valid_X, self.h_valid_y)], verbose=0, early_stopping_rounds=25)
            loss = roc_auc_score(self.h_valid_y, lgb_model.predict(self.h_valid_X))
            return loss
                
        study = optuna.create_study(direction='maximize', sampler=self.args.sampler)
        study.optimize(objective, n_trials=100)
        trial = study.best_trial
        trial_params = trial.params
        print('Best Trial: score {},\nparams {}'.format(trial.value, trial_params))

    def lgbmclf(self):
        def objective(trial):
            param = {
                "objective": trial.suggest_categorical("objective", ["binary", "cross_entropy"]),
                'verbose': -1,
                'metric': 'AUC',
                'max_depth': trial.suggest_int('max_depth',3, 15),
                'learning_rate': trial.suggest_loguniform("learning_rate", 0.001, 1.0),
                'n_estimators': trial.suggest_int('n_estimators', 100, 10000),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'lambda_l1' : trial.suggest_loguniform('lambda_l1', 1e-8, 1e-4),
                'lambda_l2' : trial.suggest_loguniform('lambda_l2', 1e-8, 1e-4),
                'path_smooth' : trial.suggest_loguniform('path_smooth', 1e-8, 1e-3),
                'num_leaves' : trial.suggest_int('num_leaves', 30, 200),
                'min_data_in_leaf' : trial.suggest_int('min_data_in_leaf', 10, 100),
                'max_bin' : trial.suggest_int('max_bin', 100, 255),
                'feature_fraction' : trial.suggest_uniform('feature_fraction', 0.5, 0.9),
                'bagging_fraction' : trial.suggest_uniform('bagging_fraction', 0.5, 0.9),
            }
            categorical = [0, 1, 2, 3, ]
            model = lgb.LGBMClassifier(**param, categorical_feature=categorical, random_state=self.args.seed)
            lgb_model = model.fit(self.h_train_X, self.h_train_y, eval_set=[(self.h_valid_X, self.h_valid_y)], verbose=0, early_stopping_rounds=25)
            loss = roc_auc_score(self.h_valid_y, lgb_model.predict_proba(self.h_valid_X)[:, 1])
            return loss
                
        study = optuna.create_study(direction='maximize', sampler=self.args.sampler)
        study.optimize(objective, n_trials=100)
        trial = study.best_trial
        trial_params = trial.params
        print('Best Trial: score {},\nparams {}'.format(trial.value, trial_params))

    def catreg(self):
        def objective(trial):
            param = {
                "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
                # "objective" : "RMSE",
                "depth": trial.suggest_int("depth", 3, 16),
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
            
            cat = CatBoostClassifier(**param, cat_features=[0, 1, 2, 3, 4], random_seed=self.args.seed, task_type=self.args.device)
            cat_model = cat.fit(self.h_train_X, self.h_train_y, eval_set=[(self.h_valid_X, self.h_valid_y)], verbose=0, early_stopping_rounds=25)    
            preds = cat_model.predict_proba(self.h_valid_X)
            accuracy = roc_auc_score(self.h_valid_y, preds[:, 1])
            torch.cuda.empty_cache()
            gc.collect()
            return accuracy

        study = optuna.create_study(direction="maximize",)
        study.optimize(objective, n_trials=100)
        trial = study.best_trial
        trial_params = trial.params
        print('Best Trial: score {},\nparams {}'.format(trial.value, trial_params))

    def catclf(self):
        def objective(trial):
            param = {
                "objective" : "RMSE",
                "depth": trial.suggest_int("depth", 3, 16),
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
            
            cat = CatBoostRegressor(**param, cat_features=[0, 1, 2, 3, 4], random_seed=self.args.seed, task_type=self.args.device)
            cat_model = cat.fit(self.h_train_X, self.h_train_y, eval_set=[(self.h_valid_X, self.h_valid_y)], verbose=0, early_stopping_rounds=25)    
            preds = cat_model.predict(self.h_valid_X)
            accuracy = roc_auc_score(self.h_valid_y, preds)
            torch.cuda.empty_cache()
            gc.collect()
            return accuracy

        study = optuna.create_study(direction="maximize",)
        study.optimize(objective, n_trials=100)
        trial = study.best_trial
        trial_params = trial.params
        print('Best Trial: score {},\nparams {}'.format(trial.value, trial_params))
