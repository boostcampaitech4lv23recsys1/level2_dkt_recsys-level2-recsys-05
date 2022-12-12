import os
import random
import time
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.preprocessing import LabelEncoder


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None
        self.cate_cols = args.cate_cols
        self.cont_cols = args.cont_cols

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        # self.args.test_start_row_id_by_user_id, self.args.test__user_id_row_id_list = self.save_row_ids(self.test_data)
        return self.test_data

    def split_data(self, df, ratio=0.7, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        # train, valid 분리
        n = len(df['userID'].unique())
        user_permute_list = np.random.permutation(df['userID'].unique())
        train_userid = user_permute_list[:(int(n*ratio))]
        valid_userid = user_permute_list[(int(n*ratio)):]
        train = df[df['userID'].isin(train_userid)].reset_index(drop=True).sort_values(["userID", "Timestamp"])
        valid = df[df['userID'].isin(valid_userid)].reset_index(drop=True).sort_values(["userID", "Timestamp"])

        return train, valid


    def __preprocessing(self, df, is_train=True):

        # cate to index
        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)
        
        offset = 1
        if is_train:
            cate2id_dict = {}
            for col in self.cate_cols:    
                cate2id = dict([(v, i+offset) for i, v in enumerate(df[col].unique())])
                df[col] = df[col].map(cate2id)
                cate2id_dict[col] = cate2id
                offset += len(cate2id)          
            cate2id_dict_path = os.path.join(self.args.asset_dir, "cate2id_dict.pickle")
            with open(cate2id_dict_path,'wb') as fw:
                pickle.dump(cate2id_dict, fw)  

        else:
            cate2id_dict_path = os.path.join(self.args.asset_dir, "cate2id_dict.pickle")
            with open(cate2id_dict_path,'rb') as fr:
                cate2id_dict = pickle.load(fr)  
            for col in self.cate_cols:    
                df[col] = df[col].map(cate2id_dict[col])
                offset += len(cate2id_dict[col])

        self.args.offset = offset

        
        return df

    def __feature_engineering(self, df):
        # TODO
        return df

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)  # , nrows=100000)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

        return df

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train=False)

def get_row_ids(df, max_seq_len, stride):
    # 데이터 증강을 위한 row_id 저장 -> dataset 생성시 활용
    df = df.reset_index().rename({'index':'row_id'}, axis=1)
    # 학습 과정에서 학습 샘플을 생성하기 위해서 필요한 유저별 row_ids를 저장
    question_row_ids_by_user_id = df.groupby('userID').apply(lambda x: x['row_id'].values)
    user_id_len = question_row_ids_by_user_id.apply(len)
    user_id_row_id_list = []
    for user_id, row_ids in question_row_ids_by_user_id.items():
        if len(row_ids) <= max_seq_len:
            user_id_row_id_list.append((user_id, row_ids[0], row_ids[-1]+1))
        else:
            for row_id in row_ids[:(max_seq_len-2):(-stride)]:
                user_id_row_id_list.append((user_id, row_id-max_seq_len+1, row_id+1))

    return user_id_len, user_id_row_id_list


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):

        self.max_seq_len = args.max_seq_len
        self.stride = args.max_seq_len
        
        self.user_id_len, self.user_id_row_id_list = get_row_ids(data, self.max_seq_len, self.stride)

        self.cate_cols = args.cate_cols
        self.cont_cols = args.cont_cols
        
        self.cate_features = data[self.cate_cols].values
        self.cont_features = data[self.cont_cols].values
        self.target = data['answerCode'].values

    def __getitem__(self, index):
        user_id, start_row_id, end_row_id = self.user_id_row_id_list[index]
        seq_len = end_row_id - start_row_id

        # 0으로 채워진 output tensor 제작
        cate_feature = torch.zeros(self.max_seq_len, len(self.cate_cols), dtype=torch.long)
        cont_feature = torch.zeros(self.max_seq_len, len(self.cont_cols), dtype=torch.float)
        target = torch.zeros(self.max_seq_len, dtype=torch.float)
        mask = torch.zeros(self.max_seq_len, dtype=torch.int16)
        
        # tensor에 값 채워넣기
        cate_feature[-seq_len:] = torch.ShortTensor(self.cate_features[start_row_id:end_row_id]) # 이미 preprocessing에서 1부터 인덱싱하기 때문에 padding 값과 구분되어 있음
        cont_feature[-seq_len:] = torch.HalfTensor(self.cont_features[start_row_id:end_row_id])
        target[-seq_len:] = torch.HalfTensor(self.target[start_row_id:end_row_id]) 
        mask[-seq_len:] = 1     
        
        return cate_feature, cont_feature, mask, target


    def __len__(self):
        return len(self.user_id_row_id_list)


def get_loaders(args, train, valid):

    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            drop_last=False,
            pin_memory=pin_memory,
        )
    if valid is not None:
        valset = DKTDataset_test(valid, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            drop_last=False,
            pin_memory=pin_memory,
        )

    return train_loader, valid_loader


# inference를 위한 dataloader 생성

def get_row_ids_test(df, max_seq_len):
    # 데이터 증강을 위한 row_id 저장 -> dataset 생성시 활용
    df = df.reset_index().rename({'index':'row_id'}, axis=1)
    # 학습 과정에서 학습 샘플을 생성하기 위해서 필요한 유저별 row_ids를 저장
    question_row_ids_by_user_id = df.groupby('userID').apply(lambda x: x['row_id'].values)
    user_id_row_id_start_end = []
    for user_id, row_ids in question_row_ids_by_user_id.items():
        target_row_ids = row_ids[-max_seq_len:]
        user_id_row_id_start_end.append([user_id, target_row_ids[0], target_row_ids[-1]+1])

    return user_id_row_id_start_end


class DKTDataset_test(torch.utils.data.Dataset):
    def __init__(self, data, args):

        self.max_seq_len = args.max_seq_len
        
        self.user_id_row_id_list = get_row_ids_test(data, self.max_seq_len)

        self.cate_cols = args.cate_cols
        self.cont_cols = args.cont_cols
        
        self.cate_features = data[self.cate_cols].values
        self.cont_features = data[self.cont_cols].values
        self.target = data['answerCode'].values

    def __getitem__(self, index):
        user_id, start_row_id, end_row_id = self.user_id_row_id_list[index]
        seq_len = end_row_id - start_row_id

        # 0으로 채워진 output tensor 제작
        cate_feature = torch.zeros(self.max_seq_len, len(self.cate_cols), dtype=torch.long)
        cont_feature = torch.zeros(self.max_seq_len, len(self.cont_cols), dtype=torch.float)
        target = torch.zeros(self.max_seq_len, dtype=torch.float)
        mask = torch.zeros(self.max_seq_len, dtype=torch.int16)
        
        # tensor에 값 채워넣기
        cate_feature[-seq_len:] = torch.ShortTensor(self.cate_features[start_row_id:end_row_id])
        cont_feature[-seq_len:] = torch.HalfTensor(self.cont_features[start_row_id:end_row_id])
        target[-seq_len:] = torch.HalfTensor(self.target[start_row_id:end_row_id])
        mask[-seq_len:] = 1     
        
        return cate_feature, cont_feature, mask, target

    def __len__(self):
        return len(self.user_id_row_id_list)
