import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

import torch
import torch.nn as nn
from torch_geometric.nn.models import LightGCN

import math
import scipy.sparse as sp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')

concat = pd.concat([train, test]).reset_index(drop=True)

user2vec = {v:k+1 for k, v in enumerate(sorted(concat.userID.unique()))}
n_user = concat.userID.nunique()

item2vec = {v:k+n_user for k, v in enumerate(sorted(concat.assessmentItemID.unique()))}
tag2vec = {v:k for k, v in enumerate(sorted(concat.KnowledgeTag.unique()))}

train['userID'] = train['userID'].apply(lambda x : user2vec[x])
test['userID'] = test['userID'].apply(lambda x : user2vec[x])
concat['userID'] = concat['userID'].apply(lambda x : user2vec[x])
train['assessmentItemID'] = train['assessmentItemID'].apply(lambda x : item2vec[x])
test['assessmentItemID'] = test['assessmentItemID'].apply(lambda x : item2vec[x])
concat['assessmentItemID'] = concat['assessmentItemID'].apply(lambda x : item2vec[x])
train['KnowledgeTag'] = train['KnowledgeTag'].apply(lambda x : tag2vec[x])
test['KnowledgeTag'] = test['KnowledgeTag'].apply(lambda x : tag2vec[x])
concat['KnowledgeTag'] = concat['KnowledgeTag'].apply(lambda x : tag2vec[x])

data = concat[concat.answerCode >= 0]

train_index, valid_index = train_test_split(data.index, random_state=0, test_size=0.2)

train = data.loc[train_index, ['userID', 'assessmentItemID', 'KnowledgeTag', 'answerCode']]
userid, itemid, tags, answer = train.userID, train.assessmentItemID, train.KnowledgeTag, train.answerCode
train_edge, train_feature, train_label = [], [], []
for user, item, tag, acode in zip(userid, itemid, tags, answer):
    train_edge.append([user, item])
    train_feature.append(tag)
    train_label.append(acode)

edge_train = torch.LongTensor(train_edge).T
feature_train = torch.LongTensor(train_feature)
y_train = torch.LongTensor(train_label)

valid = data.loc[valid_index, ['userID', 'assessmentItemID', 'KnowledgeTag', 'answerCode']]
userid, itemid, tags, answer = valid.userID, valid.assessmentItemID, valid.KnowledgeTag, valid.answerCode
valid_edge, valid_feature, valid_label = [], [], []
for user, item, tag, acode in zip(userid, itemid, tags, answer):
    valid_edge.append([user, item])
    valid_feature.append(tag)
    valid_label.append(acode)
    
edge_valid = torch.LongTensor(valid_edge).T
feature_valid = torch.LongTensor(valid_feature)
y_valid = torch.LongTensor(valid_label)

train_data = dict(edge=edge_train.to(device), feature=feature_train.to(device), label=y_train.to(device))
valid_data = dict(edge=edge_valid.to(device), feature=feature_valid.to(device), label=y_valid.to(device))

test_model = LightGCN(concat.userID.nunique()+concat.assessmentItemID.nunique(), embedding, hidden_dim, feature, layers)
test_model = test_model.to(device)
test_model.load_state_dict(torch.load('model_2.pt')['state_dict'])

test_model.eval()
test_edge = test_edge.to(device)
features = features.to(device)
prediction = test_model(test_edge, features).sigmoid()

sub = pd.read_csv('sample_submission.csv')
sub['prediction'] = prediction.detach().cpu()
sub.to_csv(f'{embedding}_{hidden_dim}_{layers}_{best_auc:.03f}_tag.csv', index=False)