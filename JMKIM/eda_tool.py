import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from collections import defaultdict
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm

def null_check(train_df, test_df, columns=['userID', 'assessmentItemID', 'testId', 'answerCode', 'Timestamp', 'KnowledgeTag']):
    if sum((train_df==train_df).sum().values * len(train_df.columns) == len(train_df) * len(train_df.columns)) == len(train_df.columns):
        for column in columns:
            print(f"[TOTAL] {column}'s null : {len(train_df.loc[train_df[column].isnull()])}")
        return

    for column in columns:
        print(f"[TRAIN] {column}'s null : {len(train_df.loc[train_df[column].isnull()])}")
        print(f"[TEST] {column}'s null : {len(test_df.loc[test_df[column].isnull()])}")
    return

def nunique(train_df, test_df, column):
    if sum((train_df==train_df).sum().values * len(train_df.columns) == len(train_df) * len(train_df.columns)) == len(train_df.columns):
        print(f"[TOTAL] {column}'s number of unique : {train_df[column].nunique()}")    
        return

    print(f"[TRAIN] {column}'s number of unique : {train_df[column].nunique()}")
    print(f"[TEST] {column}'s number of unique : {test_df[column].nunique()}")
    print(f"TRAIN {column}'s unique values are equal to TEST ? : {sorted(train_df[column].unique()) == sorted(test_df[column].unique())}")
    return

def count_plot(train_df, test_df, column, train_palette=None, test_palette=None):
    if sum((train_df==train_df).sum().values * len(train_df.columns) == len(train_df) * len(train_df.columns)) == len(train_df.columns):
        print(f"[TOTAL] {column}'s number of unique : {train_df[column].nunique()}")    
        plt.title("TRAIN Timestamp")
        sns.countplot(x=column, data=train_df, palette=train_palette)
        plt.show()
        return

    plt.figure(figsize=(13, 5))
    plt.subplot(1, 2, 1)
    plt.title("TRAIN Timestamp")
    sns.countplot(x=column, data=train_df, palette=train_palette)
    
    plt.subplot(1, 2, 2)
    plt.title("TEST Timestamp")
    sns.countplot(x=column, data=test_df, palette=test_palette)
    plt.show()
    return

def kde_hist_plot(train_df, x, hue='answerCode'):
    plt.figure(figsize=(20, 6))
    plt.subplot(1, 3, 1)

    sns.kdeplot(x=x, data=train_df.loc[train_df['answerCode'] == 1], label="Correct")
    sns.kdeplot(x=x, data=train_df.loc[train_df['answerCode'] == 0], label="Wrong")
    plt.legend()

    plt.subplot(1, 3, 2)
    sns.histplot(x=x, data=train_df, hue=hue)

    plt.subplot(1, 3, 3)
    sns.kdeplot(x=x, data=train_df, hue=hue, multiple='fill')
    
    plt.show()
    return

def extract_datetime(df):
    df['month'] = pd.to_datetime(df['Timestamp']).apply(lambda x : x.month)
    df['day'] = pd.to_datetime(df['Timestamp']).apply(lambda x : x.day)
    df['hour'] = pd.to_datetime(df['Timestamp']).apply(lambda x : x.hour)
    df['minute'] = pd.to_datetime(df['Timestamp']).apply(lambda x : x.minute)
    df['second'] = pd.to_datetime(df['Timestamp']).apply(lambda x : x.second)
    return df

def extract_testId(df):
    df['testClass'] = df['testId'].apply(lambda x : int(x[2]))
    df['testCode'] = df['testId'].apply(lambda x : int(x[7:]))
    return df

def extract_assessmentItemID(df):
    df['assessmentItemCode'] = df['assessmentItemID'].apply(lambda x : int(x[7:]))
    return df

def decomposition(total):
    total_decompose_col = defaultdict(lambda: np.zeros(len(total)))
    ignore = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'Timestamp', 'KnowledgeTag', 'assessmentItemID_last', 'testId_first', 'testId_last', 'relative_answered_correctly']
    for key, group in tqdm(total.groupby(by=["userID"]), total=total['userID'].nunique()):
        indices = group.index
        for idx, col in enumerate(group.columns):
            if not (col in ignore or re.findall('trend|seasonal|resid', col)):
                res = STL(group[col], period=60).fit()
                total_decompose_col[f'trend_{col}'][indices] = res.trend.values
                total_decompose_col[f'seasonal_{col}'][indices] = res.seasonal.values
                total_decompose_col[f'resid_{col}'][indices] = res.resid.values
                
    new_total = pd.concat([total, pd.DataFrame(total_decompose_col)], axis=1)
    new_total.to_csv("sequential_total.csv", index=False)
    
    return new_total

def forward_stepwise_regression(x_train, y_train):

    # 변수목록, 선택된 변수 목록, 단계별 모델과 AIC 저장소 정의
    features = list(x_train)
    selected = []
    step_df = pd.DataFrame({ 'step':[], 'feature':[],'aic':[]})

    # 
    for s in tqdm(range(0, len(features))) :
        result =  { 'step':[], 'feature':[],'aic':[]}

        # 변수 목록에서 변수 한개씩 뽑아서 모델에 추가
        for f in features :
            vars = selected + [f]
            x_tr = x_train[vars]
            model = sm.OLS(y_train, x_tr).fit()
            result['step'].append(s+1)
            result['feature'].append(vars)
            result['aic'].append(model.aic)
        
        # 모델별 aic 집계
        temp = pd.DataFrame(result).sort_values('aic').reset_index(drop = True)
        
        # 만약 이전 aic보다 새로운 aic 가 크다면 멈추기
        if step_df['aic'].min() < temp['aic'].min() :
            break
        step_df = pd.concat([step_df, temp], axis = 0).reset_index(drop = True)

        # 선택된 변수 제거
        v = temp.loc[0,'feature'][s]
        features.remove(v)

        selected.append(v)
    print(f"[Forward Selection] Remove : {set(x_train.columns) - set(features)}")
    
    # 선택된 변수와 step_df 결과 반환
    return selected, step_df