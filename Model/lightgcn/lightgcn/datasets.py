import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def prepare_dataset(device, basepath, verbose=True, logger=None):
    data = load_data(basepath)
    train_data, test_data = separate_data(data)
    id2index = indexing_data(data)
    train_data_proc, valid_data_proc = process_data(train_data, id2index, device, True)
    test_data_proc = process_data(test_data, id2index, device)

    if verbose:
        print_data_stat(train_data, "Train", logger=logger)
        print_data_stat(test_data, "Test", logger=logger)

    return train_data_proc, test_data_proc, valid_data_proc, len(id2index)


def load_data(basepath):
    # path1 = os.path.join(basepath, "train_data.csv")
    # path2 = os.path.join(basepath, "test_data.csv")
    path1 = os.path.join(basepath, "train_feature_engineering.csv")
    path2 = os.path.join(basepath, "test_feature_engineering.csv")
    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)

    data = pd.concat([data1, data2])
    data.drop_duplicates(
        subset=["userID", "assessmentItemID"], keep="last", inplace=True
    )
    scaler = RobustScaler()
    data.iloc[:, 9:] = scaler.fit_transform(data.iloc[:, 9:])

    return data


def separate_data(data):
    train_data = data[data.answerCode >= 0]
    test_data = data[data.answerCode < 0]

    return train_data, test_data


def indexing_data(data):
    userid, itemid = (
        sorted(list(set(data.userID))),
        sorted(list(set(data.assessmentItemID))),
    )
    n_user, n_item = len(userid), len(itemid)
    userid_2_index = {v: i for i, v in enumerate(userid)}
    itemid_2_index = {str(v): i + n_user for i, v in enumerate(itemid)}
    id_2_index = dict(userid_2_index, **itemid_2_index)

    return id_2_index


def process_data(data, id_2_index, device, istrain=False):
    data = data.drop(['relative_answered_correctly', 'row_id'], axis=1);

    edge, label = [], []
    edge_context = []
    context = np.array(data.iloc[:, 5:]);
    for user, item, acode, ct in zip(data.userID, data.assessmentItemID, data.answerCode, context):
        # uid, iid = id_2_index[user], id_2_index[item]
        # edge.append([uid, iid])
        uid, iid = id_2_index[user], id_2_index[str(item)];
        edge_context.append([[uid, iid], list(ct)])
        label.append(acode)
    
    # import pdb;pdb.set_trace();
    if istrain:
        # h_train_X, h_valid_X, h_train_y, h_valid_y = train_test_split(edge, label, test_size=0.2, stratify=label, random_state=777)
        h_train_X, h_valid_X, h_train_y, h_valid_y = train_test_split(edge_context, label, test_size=0.2, stratify=label, random_state=777);
        train_edge = [];
        train_context = [];
        valid_edge = [];
        valid_context = [];

        for e, c in h_train_X:
            train_edge.append(e);
            train_context.append(c);
        for e, c in h_valid_X:
            valid_edge.append(e);
            valid_context.append(c);

        train_edge = torch.LongTensor(train_edge)
        train_label = torch.LongTensor(h_train_y)
        valid_edge = torch.LongTensor(valid_edge)
        valid_label = torch.LongTensor(h_valid_y)
        train_context = torch.FloatTensor(train_context)
        valid_context = torch.FloatTensor(valid_context)

        # train_edge = torch.LongTensor(h_train_X).T
        # train_label = torch.LongTensor(h_train_y)
        # valid_edge = torch.LongTensor(h_valid_X).T
        # valid_label = torch.LongTensor(h_valid_y)
        # return dict(train_edge=train_edge.to(device), train_label=train_label.to(device)), dict(valid_edge=valid_edge.to(device), valid_label=valid_label)
        return dict(train_edge=train_edge, train_context=train_context, train_label=train_label), dict(valid_edge=valid_edge, valid_context=valid_context, valid_label=valid_label)
    else:
        test_edge, test_context = [], []
        for e, c in edge_context:
            test_edge.append(e);
            test_context.append(c);

        test_edge = torch.LongTensor(test_edge).T;
        test_context = torch.FloatTensor(test_context);
        label = torch.LongTensor(label);
        return dict(test_edge=test_edge, test_context=test_context, label=label);

        # edge = torch.LongTensor(edge).T
        # label = torch.LongTensor(label)
        # return dict(edge=edge.to(device), label=label.to(device))


    


def print_data_stat(data, name, logger):
    userid, itemid = list(set(data.userID)), list(set(data.assessmentItemID))
    n_user, n_item = len(userid), len(itemid)

    logger.info(f"{name} Dataset Info")
    logger.info(f" * Num. Users    : {n_user}")
    logger.info(f" * Max. UserID   : {max(userid)}")
    logger.info(f" * Num. Items    : {n_item}")
    logger.info(f" * Num. Records  : {len(data)}")



class LightgcnDataset(torch.utils.data.Dataset):
    def __init__(self, edge, context, label=None):
        self.edge = edge
        self.context = context
        self.label = label

    def __getitem__(self, index):
        edge, context = self.edge[index], self.context[index]
        if self.label != None:
            label = self.label[index]
            return [edge, context, label]
        return [edge, context]

    def __len__(self):
        assert len(self.edge) == len(self.context)
        return len(self.edge)