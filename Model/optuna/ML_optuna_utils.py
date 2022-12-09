import os
import random
import numpy as np
import pandas as pd
import torch
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, classification_report

def setSeeds(seed=42):
    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_data(args):
    pd.read_csv(os.path.join(args.data_dir, args.file_name))
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    pd.set_option('mode.chained_assignment',  None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

    train_df = pd.read_csv("../data/train_feature_engineering.csv")
    test_df = pd.read_csv("../data/test_feature_engineering.csv")
    total = pd.read_csv("../data/total.csv")

    total = total.loc[total["answerCode"] != -1]
    test_df = test_df.loc[test_df['answerCode'] == -1]

    train_X = total.drop(['answerCode', 'assessmentItemID', 'testId', 'Timestamp', 'relative_answered_correctly'], axis=1)
    train_y = total[['userID', 'answerCode']]
    test_X = test_df.drop(['answerCode', 'assessmentItemID', 'testId', 'Timestamp', 'relative_answered_correctly'], axis=1)
    h_train_X, h_valid_X, h_train_y, h_valid_y = train_test_split(train_X, train_y['answerCode'], test_size=0.1, stratify=train_y['answerCode'], random_state=seed)

    return h_train_X, h_valid_X, h_train_y, h_valid_y, test_X
    