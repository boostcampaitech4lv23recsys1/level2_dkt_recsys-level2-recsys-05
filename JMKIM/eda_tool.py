import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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