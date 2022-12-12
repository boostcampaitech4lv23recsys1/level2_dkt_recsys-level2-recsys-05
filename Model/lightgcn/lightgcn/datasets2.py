import os

import numpy as np
import pandas as pd
import torch


def prepare_dataset(device, basepath, cate_cols, cont_cols, verbose=True, logger=None, ratio=0.3):
    data = load_data(basepath)
    train_data, test_data = separate_data(data)
    id2index, offset = indexing_data(data, cate_cols)
    train_data_proc = process_data(train_data, id2index, cont_cols, device)
    test_data_proc = process_data(test_data, id2index, cont_cols, device)
    train_data_proc, valid_data_proc = train_valid_split(train_data_proc, ratio)

    if verbose:
        print_data_stat(train_data, "Train", logger=logger)
        print_data_stat(test_data, "Test", logger=logger)

    return train_data_proc, valid_data_proc, test_data_proc, len(id2index['user_item']), offset


def load_data(basepath):
    path1 = os.path.join(basepath, "train_feature_engineering.csv")
    path2 = os.path.join(basepath, "test_feature_engineering.csv")
    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)

    data = pd.concat([data1, data2])
    data.drop_duplicates(
        subset=["userID", "assessmentItemID"], keep="last", inplace=True
    )

    return data


def separate_data(data):
    train_data = data[data.answerCode >= 0]
    test_data = data[data.answerCode < 0]

    return train_data, test_data


def indexing_data(data, cate_cols):
    id2index = {}
    userid, itemid = (
        sorted(list(set(data.userID))),
        sorted(list(set(data.assessmentItemID))),
    )
    n_user, n_item = len(userid), len(itemid)

    userid_2_index = {v: i for i, v in enumerate(userid)}
    itemid_2_index = {v: i + n_user for i, v in enumerate(itemid)}
    id2index['user_item'] = dict(userid_2_index, **itemid_2_index)
    
    offset = 1
    for col in cate_cols:    
        cate2id = dict([(v, i+offset) for i, v in enumerate(data[col].unique())])
        data[col] = data[col].map(cate2id)
        id2index[col] = cate2id
        offset += len(cate2id)    
    
    return id2index, offset


def process_data(data, id2index, cont_cols, device):
    edge, label, cate_features, cont_features = [], [], [], []
    for user, item, acode in zip(data.userID, data.assessmentItemID, data.answerCode):
        uid, iid = id2index['user_item'][user], id2index['user_item'][item]
        edge.append([uid, iid])
        label.append(acode)

    for k, v in id2index.items():
        if k != 'user_item':
            feat = data[k].map(v).to_list()
            cate_features.append(feat)
    
    for col in cont_cols:
        feat = data[col].to_list()
        cont_features.append(feat)

    edge = torch.LongTensor(edge).T
    label = torch.LongTensor(label)
    cate_features = torch.LongTensor(cate_features)
    cont_features = torch.FloatTensor(cont_features)

    return dict(edge=edge.to(device), label=label.to(device), cate_features=cate_features.to(device), cont_features=cont_features.to(device))

def train_valid_split(data, ratio):
    n = len(data["label"])
    valid_idx = np.random.choice(n, int(n*ratio), replace=False)
    train_idx = np.setdiff1d(range(n), valid_idx)
    edge, label, cate_features, cont_features = data["edge"], data["label"], data["cate_features"], data["cont_features"]
    train_data = dict(edge=edge[:, train_idx], label=label[train_idx], cate_features=cate_features[:,train_idx], cont_features=cont_features[:,train_idx])
    valid_data = dict(edge=edge[:, valid_idx], label=label[valid_idx], cate_features=cate_features[:,valid_idx], cont_features=cont_features[:,valid_idx])

    return train_data, valid_data

def print_data_stat(data, name, logger):
    userid, itemid = list(set(data.userID)), list(set(data.assessmentItemID))
    n_user, n_item = len(userid), len(itemid)

    logger.info(f"{name} Dataset Info")
    logger.info(f" * Num. Users    : {n_user}")
    logger.info(f" * Max. UserID   : {max(userid)}")
    logger.info(f" * Num. Items    : {n_item}")
    logger.info(f" * Num. Records  : {len(data)}")
