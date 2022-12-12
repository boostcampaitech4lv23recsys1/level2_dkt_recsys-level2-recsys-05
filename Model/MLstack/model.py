import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def train_LGBM(data_train, data_y_train, data_test, data_y_test, columns) :
    lgbm_best_auc = 0
    lgbm_best_col = []
    item = list(columns)
    tolerance = 0
    for i in range(len(columns)-1) :
        lgbm = LGBMClassifier(boosting_type='gbdt', metric='auc')
        lgbm.fit(data_train[item], data_y_train)
        auc = roc_auc_score(data_y_test, lgbm.predict_proba(data_test[item])[:, 1])
        if auc >= lgbm_best_auc :
            lgbm_best_auc = auc
            lgbm_best_col = item.copy()
            tolerance = 0
        else :
            tolerance += 1
            if tolerance == 10 :
                break
        
        item.pop(np.argmin(lgbm.feature_importances_))
                
    model = LGBMClassifier(boosting_type='gbdt', metric='auc')
    return model, lgbm_best_col

def train_XGB(data_train, data_y_train, data_test, data_y_test, columns) :
    xgb_best_auc = 0
    xgb_best_col = []
    item = list(columns)
    tolerance = 0
    for i in range(len(columns)-1) :
        xgb = XGBClassifier(tree_method='gpu_hist', gpu_id=0)
        xgb.fit(data_train[item], data_y_train, verbose=False)
        auc = roc_auc_score(data_y_test, xgb.predict_proba(data_test[item])[:, 1])
        if auc >= xgb_best_auc :
            xgb_best_auc = auc
            xgb_best_col = item[:]
            tolerance = 0
        else :
            tolerance += 1
            if tolerance == 10 :
                break
                
        print(f"count:{i}, auc:{auc}, cols:{item}")
        
        item.pop(np.argmin(xgb.feature_importances_))
                       
    model = XGBClassifier(tree_method='gpu_hist', gpu_id=0)
    return model, xgb_best_col

def train_cat(data_train, data_y_train, data_test, data_y_test, columns) :
    cat_best_auc = 0
    cat_best_col = []
    item = list(columns)
    tolerance = 0
    for i in range(len(columns)-1) :
        cat = CatBoostClassifier(task_type="GPU", eval_metric='AUC')
        cat.fit(data_train[item], data_y_train, verbose=False)
        auc = roc_auc_score(data_y_test, cat.predict_proba(data_test[item])[:, 1])
        if auc >= cat_best_auc :
            cat_best_auc = auc
            cat_best_col = item[:]
            tolerance = 0
        else :
            tolerance += 1
            if tolerance == 10 :
                break
                
        print(f"count:{i}, auc:{auc}, cols:{item}")
        
        item.pop(np.argmin(cat.feature_importances_))
            
    model = CatBoostClassifier(boosting_type='gbdt', metric='auc')
    return model, cat_best_col

def train_ridge(data_train, data_y_train, data_test, data_y_test, columns) :
    rg_best_auc = 0
    rg_best_col = []
    temp_col = columns
    tolerance = 0
    for i in range(len(columns), 1, -1) :
        cols = temp_col
        best_temp_auc = 0
        for item in itertools.combinations(cols, i) :
            item = list(item)
            rg = Ridge()
            rg.fit(data_train[item], data_y_train)
            auc = roc_auc_score(data_y_test, rg.predict(data_test[item]))
            if best_temp_auc < auc :
                best_temp_auc = auc
                temp_col = item
                
        if best_temp_auc >= rg_best_auc :
            rg_best_auc = best_temp_auc
            rg_best_col = temp_col[:]
            tolerance = 0
        else :
            tolerance += 1
            if tolerance == 10 :
                break

        print(f"len:{i}, best_temp_auc:{best_temp_auc}, cols:{temp_col}")
                
    model = Ridge()
    return model, rg_best_col