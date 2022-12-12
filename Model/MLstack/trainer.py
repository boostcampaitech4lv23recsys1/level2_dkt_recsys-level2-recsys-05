from model import train_cat, train_LGBM, train_ridge, train_XGB

import pandas as pd

from lightgbm import LGBMClassifier

def inference(X_train, X_valid, y_train, y_valid, test, columns) :
    model = LGBMClassifier(boosting_type='gbdt', metric='auc')

    lgbm, lgbm_col = train_LGBM(X_train, X_valid, y_train, y_valid, columns)
    cat, cat_col = train_cat(X_train, X_valid, y_train, y_valid, columns)
    xgb, xgb_col = train_XGB(X_train, X_valid, y_train, y_valid, columns)
    ridge, ridge_col = train_ridge(X_train, X_valid, y_train, y_valid, columns)

    answer_train = {}
    answer_train['lgbm'] = lgbm.predict_proba(X_train[lgbm_col])[:, 1]
    answer_train['cat'] = cat.predict_proba(X_train[cat_col])[:, 1]
    answer_train['xgb'] = xgb.predict_proba(X_train[xgb_col])[:, 1]
    answer_train['ridge'] = ridge.predict(X_train[ridge_col])
    answer_train_df = pd.DataFrame(answer_train)

    model.fit(answer_train_df, y_train)

    answer_test = {}
    answer_test['lgbm'] = lgbm.predict_proba(test[lgbm_col])[:, 1]
    answer_test['cat'] = cat.predict_proba(test[cat_col])[:, 1]
    answer_test['xgb'] = xgb.predict_proba(test[xgb_col])[:, 1]
    answer_test['ridge'] = ridge.predict(test[ridge_col])
    answer_test_df = pd.DataFrame(answer_test)

    prediction = model.predict_proba(answer_test_df)[:, 1]

    return prediction