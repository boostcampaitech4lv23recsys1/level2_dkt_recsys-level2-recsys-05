from torch.utils.data import Dataset, DataLoader

import pandas as pd
import torch

from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, idx):
        edge = torch.stack([self.data['edge'][0][idx], self.data['edge'][1][idx]])
        feature = self.data['feature'][idx]
        label = self.data['label'][idx]
        return edge, feature, label

def load_data(train_path, test_path, device) :
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    concat = pd.concat([train, test]).reset_index(drop=True)

    data = preprocessing(concat)
    train_data, valid_data, test_data = split(data, device)
    
    train_set = CustomDataset(train_data)
    train_dataloader = DataLoader(train_set, batch_size=2048, shuffle=True)

    nodes = concat.userID.nunique()+concat.assessmentItemID.nunique()

    return train_dataloader, valid_data, test_data, nodes

def preprocessing(data) :
    user2vec = {v:k+1 for k, v in enumerate(sorted(data.userID.unique()))}
    n_user = data.userID.nunique()

    item2vec = {v:k+n_user for k, v in enumerate(sorted(data.assessmentItemID.unique()))}
    tag2vec = {v:k for k, v in enumerate(sorted(data.KnowledgeTag.unique()))}

    data['userID'] = data['userID'].apply(lambda x : user2vec[x])
    data['assessmentItemID'] = data['assessmentItemID'].apply(lambda x : item2vec[x])
    data['KnowledgeTag'] = data['KnowledgeTag'].apply(lambda x : tag2vec[x])

    return data

def split(concat, device) :
    data = [concat.answerCode >= 0]

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

    test = concat[concat.answerCode == -1]
    users, features, items = [], [], []
    for _, (user, item, tag) in test[['userID', 'assessmentItemID', 'KnowledgeTag']].iterrows() :
        users.append(user)
        items.append(item)
        features.append(tag)
    users = torch.LongTensor(users)
    items = torch.LongTensor(items)
    features = torch.LongTensor(features)
    test_edge = torch.stack([users, items])

    train_data = dict(edge=edge_train.to(device), feature=feature_train.to(device), label=y_train.to(device))
    valid_data = dict(edge=edge_valid.to(device), feature=feature_valid.to(device), label=y_valid.to(device))
    test_data = dict(edge=test_edge.to(device), feature=features.to(device))

    return train_data, valid_data, test_data

def prediction(model, test_data) :
    sub = pd.read_csv('sample_submission.csv')
    prediction = model.prediction(test_data)
    sub['prediction'] = prediction.detach().cpu()
    sub.to_csv(f'output.csv', index=False)
    print('success')