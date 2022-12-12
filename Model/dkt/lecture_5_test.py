import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.utils import setSeeds

setSeeds()

# RiiiD 데이터셋 path 설정
PATH = '../../data'

# 데이터셋 불러오기
train_df = pd.read_csv(os.path.join(PATH, 'train_data.csv'))
test_df = pd.read_csv(os.path.join(PATH, 'test_data.csv'))

test_user_id = test_df['userID'].unique()

train_df = pd.concat([pd.read_csv(os.path.join(PATH, 'train_data.csv')), test_df]).reset_index(drop=True)
train_df = train_df.sort_values(['userID', 'Timestamp'])

##############################################################################
# Preprocessing
##############################################################################

####################################################
##### `assessmentItemID`, `testId` 분리 feature 생성
####################################################
train_df['assessmentItemID_last'] = train_df['assessmentItemID'].apply(lambda x: x[-3:]).astype(str)
train_df['testId_first'] = train_df['testId'].apply(lambda x: x[2]).astype(str)
train_df['testId_last'] = train_df['testId'].apply(lambda x: x[-3:]).astype(str)

### category to index
cate2id_dict = {}

# 0은 nan이 사용한다
offset = 0

# assessmentItemID
assessment2id = dict([(v, i+offset) for i, v in enumerate(train_df['assessmentItemID'].unique())])
cate2id_dict['assessment2id'] = assessment2id
offset += len(assessment2id)

# testId
test2id = dict([(v, i+offset) for i, v in enumerate(train_df['testId'].unique())])
cate2id_dict['test2id'] = test2id
offset += len(test2id)

# KnowledgeTag
tag2id = dict([(v, i+offset) for i, v in enumerate(train_df['KnowledgeTag'].unique())])
cate2id_dict['tag2id'] = tag2id
offset += len(tag2id)

# assessmentItemID_last
assessment_last2id = dict([(v, i+offset) for i, v in enumerate(train_df['assessmentItemID_last'].unique())])
cate2id_dict['assessment_last2id'] = assessment_last2id
offset += len(assessment_last2id)

# testId_first
test_first2id = dict([(v, i+offset) for i, v in enumerate(train_df['testId_first'].unique())])
cate2id_dict['test_first2id'] = test_first2id
offset += len(test_first2id)

# testId_last
test_last2id = dict([(v, i+offset) for i, v in enumerate(train_df['testId_last'].unique())])
cate2id_dict['test_last2id'] = test_last2id
offset += len(test_last2id)

# mapping
train_df['assessmentItemID'] = train_df['assessmentItemID'].map(assessment2id)
train_df['testId'] = train_df['testId'].map(test2id)
train_df['KnowledgeTag'] = train_df['KnowledgeTag'].map(tag2id)
train_df['assessmentItemID_last'] = train_df['assessmentItemID_last'].map(assessment_last2id)
train_df['testId_first'] = train_df['testId_first'].map(test_first2id)
train_df['testId_last'] = train_df['testId_last'].map(test_last2id)

### feature engineering

#### `Timestamp` -> `elapsed`
train_df['Timestamp'] = pd.to_datetime(train_df['Timestamp'])
time_diff = train_df.groupby(['userID', 'testId'], group_keys=True)['Timestamp'].diff().fillna(pd.Timedelta(seconds=0))
train_df['elapsed'] = time_diff.apply(lambda x: x.total_seconds())

#### 정답률 feature 생성

accuracy_by_assessment = train_df[train_df['answerCode']!=-1].groupby('assessmentItemID')['answerCode'].mean()
accuracy_by_test = train_df[train_df['answerCode']!=-1].groupby('testId')['answerCode'].mean()
accuracy_by_tag = train_df[train_df['answerCode']!=-1].groupby('KnowledgeTag')['answerCode'].mean()
accuracy_by_assessment_last = train_df[train_df['answerCode']!=-1].groupby('assessmentItemID_last')['answerCode'].mean()
accuracy_by_test_first = train_df[train_df['answerCode']!=-1].groupby('testId_first')['answerCode'].mean()
accuracy_by_test_last = train_df[train_df['answerCode']!=-1].groupby('testId_last')['answerCode'].mean()

train_df['accuracy_by_assessment'] = train_df['assessmentItemID'].map(accuracy_by_assessment)
train_df['accuracy_by_test'] = train_df['testId'].map(accuracy_by_test)
train_df['accuracy_by_tag'] = train_df['KnowledgeTag'].map(accuracy_by_tag)
train_df['accuracy_by_assessment_last'] = train_df['assessmentItemID_last'].map(accuracy_by_assessment_last)
train_df['accuracy_by_test_first'] = train_df['testId_first'].map(accuracy_by_test_first)
train_df['accuracy_by_test_last'] = train_df['testId_last'].map(accuracy_by_test_last)

#### 상대적 (relative) feature 생성

train_df['relative_answered_correctly'] = train_df['answerCode'] - train_df['accuracy_by_assessment']

#### `과거 (prior)` feature 생성

# 이전 문제 정답 횟수
train_df['prior_ac_count'] = train_df.groupby('userID')['answerCode'].cumsum().shift(fill_value=0)

# 이전에 푼 문제 수
train_df['prior_quest_count'] = train_df.groupby('userID')['answerCode'].cumcount()

# 이전 문제 정답률
train_df['prior_ac_accuracy'] = (train_df['prior_ac_count'] / train_df['prior_quest_count']).fillna(0)

# 이전 문제 상대적인(relative) 정답률
train_df['prior_relative_ac_sum'] = train_df.groupby('userID')['relative_answered_correctly'].cumsum().shift(fill_value=0)
train_df['prior_relative_accuracy'] = (train_df['prior_relative_ac_sum'] / train_df['prior_quest_count']).fillna(0)

# 각 문제 종류별로 이전에 몇번 풀었는지 (중복으로 푼 문제수)\
train_df['prior_assessment_frequency'] = train_df.groupby(['userID', 'assessmentItemID']).cumcount().clip(0, 255)

# 각 파트별로 이전에 몇번 풀었는지
train_df['prior_test_frequency'] = train_df.groupby(['userID', 'testId']).cumcount()

# 각 태그별로 이전에 몇번 풀었는지
train_df['prior_tags_frequency'] = train_df.groupby(['userID', 'KnowledgeTag']).cumcount()

features = ['prior_ac_accuracy',
            'prior_ac_count',
            'prior_quest_count',
            'prior_relative_ac_sum',
            'prior_relative_accuracy']
# 각 학생의 첫 row는 prior과 lagtime feature의 값을 0으로 초기화한다
train_df.loc[train_df['userID'].diff().fillna(1)>0, features] = 0

#### `바로전 (previous)` feature 생성

# 각 문제 종류별 마지막으로 푼 시간
prev_tags_timestamp = train_df.groupby(['userID', 'KnowledgeTag'])[['Timestamp']].shift()        

# 각 문제 종류별 마지막으로 푼 시점으로부터 지난 시간
# 해당 문제 종류를 마지막으로 푼 시점으로부터 시간이 오래 지날수록 문제를 맞추기 힘들 것이다
train_df['diff_time_btw_tags'] = (train_df['Timestamp'] - prev_tags_timestamp['Timestamp']).apply(lambda x: x.total_seconds())

# nan값은 [ diff_time_btw_content_ids ] 데이터 중 최대값으로 imputation을 한다
max_diff_time_btw_tags = train_df['diff_time_btw_tags'].max()
train_df['diff_time_btw_tags'] = train_df['diff_time_btw_tags'].fillna(max_diff_time_btw_tags)          

# 각 문제 종류별 마지막으로 풀었을때 정답 여부
prev_correct_ac = train_df.groupby(['userID', 'KnowledgeTag'])[['answerCode']].shift()        
train_df['prev_tag_answer'] = prev_correct_ac['answerCode'].fillna(0)

### Transform

log_trans_features = ['elapsed', 'prior_ac_count', 'prior_quest_count', 'prior_assessment_frequency', 'prior_test_frequency', 'prior_tags_frequency', 'diff_time_btw_tags']
train_df[log_trans_features] = np.log1p(train_df[log_trans_features])

### 범주형/수치형 feature 데이터 타입 변환

train_df.drop('Timestamp', axis=1, inplace=True)

cont_features = [
    'elapsed',
    'accuracy_by_assessment', 'accuracy_by_test', 'accuracy_by_tag',
    'accuracy_by_assessment_last', 'accuracy_by_test_first',
    'accuracy_by_test_last', 
    'prior_ac_count', 'prior_quest_count', 'prior_ac_accuracy',
    'prior_relative_ac_sum', 'prior_relative_accuracy',
    'prior_assessment_frequency', 'prior_test_frequency',
    'prior_tags_frequency', 'diff_time_btw_tags', 'prev_tag_answer',
    'relative_answered_correctly', 'answerCode'
]
cate_features = [
    'assessmentItemID', 'testId', 'KnowledgeTag',
       'assessmentItemID_last', 'testId_first', 'testId_last'
]

train_df[cate_features] = train_df[cate_features].astype(np.int16) # -32768 ~ 32767
train_df[cont_features] = train_df[cont_features].astype(np.float32)

### train, valid, test 분리

# train test 다시 분리
train = train_df[~train_df['userID'].isin(test_user_id)].reset_index(drop=True)
test = train_df[train_df['userID'].isin(test_user_id)].reset_index(drop=True).reset_index().rename({'index':'row_id'}, axis=1)

# train, valid 분리
n = len(train['userID'].unique())
n_ratio = 0.7
user_permute_list = np.random.permutation(train['userID'].unique())
train_userid = user_permute_list[:(int(n*n_ratio))]
valid_userid = user_permute_list[(int(n*n_ratio)):]
valid = train[train['userID'].isin(valid_userid)].reset_index(drop=True).reset_index().rename({'index':'row_id'}, axis=1)
train = train[train['userID'].isin(train_userid)].reset_index(drop=True).reset_index().rename({'index':'row_id'}, axis=1)

## 📗 CFG (Configuration) class
class CFG:
    seed=7
    device='gpu'

    batch_size=128

    dropout=0.2
    emb_size=100
    hidden_size=128
    nlayers=2
    nheads=8
  
    seq_len=32
    target_size=1

    # 학습
    n_epochs = 20
    lr = 0.0001
    clip_grad = 10
    patience = 5
    log_steps = 50

    optimizer = 'adam'
    scheduler = 'plateau'

    model_dir = 'models/'
    output_dir = 'output/'

CFG.total_cate_size = offset
CFG.cate_cols = cate_features
CFG.cont_cols = cont_features

CFG.cate_col_size = len(cate_features)
CFG.cont_col_size = len(cont_features)

CFG.n_rows_per_step = 2

## 📗 데이터셋 및 데이터 로더 (Dataset and DataLoader)

# train

# 학습 과정에서 학습 샘플을 생성하기 위해서 필요한 유저별 row_ids를 저장
train_question_row_ids_by_user_id = train.groupby('userID').apply(lambda x: x['row_id'].values)
train_question_row_ids_by_user_id.reset_index().head()
# 학습 과정에서 학습 샘플을 생성하기 위해서 필요한 유저별 시작 row_id를 저장
train_start_row_id_by_user_id = train.groupby('userID').apply(lambda x: x['row_id'].values[0])
train_start_row_id_by_user_id.reset_index().head()

train_user_id_row_id_list = [(user_id, row_id)
                             for user_id, row_ids in train_question_row_ids_by_user_id.items()
                             for row_id in row_ids]
train_user_id_row_id_list[:10]

# valid

# 학습 과정에서 학습 샘플을 생성하기 위해서 필요한 유저별 row_ids를 저장
valid_question_row_ids_by_user_id = valid.groupby('userID').apply(lambda x: x['row_id'].values)
valid_question_row_ids_by_user_id.reset_index().head()
# 학습 과정에서 학습 샘플을 생성하기 위해서 필요한 유저별 시작 row_id를 저장
valid_start_row_id_by_user_id = valid.groupby('userID').apply(lambda x: x['row_id'].values[0])
valid_start_row_id_by_user_id.reset_index().head()

valid_user_id_row_id_list = [(user_id, row_id)
                             for user_id, row_ids in valid_question_row_ids_by_user_id.items()
                             for row_id in row_ids]
valid_user_id_row_id_list[:10]

# test

# 학습 과정에서 학습 샘플을 생성하기 위해서 필요한 유저별 row_ids를 저장
test_question_row_ids_by_user_id = test.groupby('userID').apply(lambda x: x['row_id'].values)
test_question_row_ids_by_user_id.reset_index().head()
# 학습 과정에서 학습 샘플을 생성하기 위해서 필요한 유저별 시작 row_id를 저장
test_start_row_id_by_user_id = test.groupby('userID').apply(lambda x: x['row_id'].values[0])
test_start_row_id_by_user_id.reset_index().head()

test_user_id_row_id_list = [(user_id, row_id)
                             for user_id, row_ids in test_question_row_ids_by_user_id.items()
                             for row_id in row_ids]
test_user_id_row_id_list[:10]

# configuration에 등록!
CFG.train_start_row_id_by_user_id = train_start_row_id_by_user_id
CFG.train_user_id_row_id_list = train_user_id_row_id_list

### 🟡 IscreamDataset / DataLoader

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class IscreamDataset(Dataset):
    def __init__(self, df, cfg, user_id_row_id_list, start_row_id_by_user_id, max_seq_len=100, max_content_len=1000):        
        
        self.max_seq_len = max_seq_len
        self.max_content_len = max_content_len
        
        self.user_id_row_id_list = user_id_row_id_list
        self.start_row_id_by_user_id = start_row_id_by_user_id

        self.cate_cols = cfg.cate_cols
        self.cont_cols = cfg.cont_cols
        
        self.cate_features = df[self.cate_cols].values
        self.cont_features = df[self.cont_cols].values

    def __getitem__(self, idx):
        
        user_id, end_row_id = self.user_id_row_id_list[idx]
        end_row_id += 1
        
        start_row_id = self.start_row_id_by_user_id[user_id]
        start_row_id = max(end_row_id - self.max_seq_len, start_row_id) # lower bound
        seq_len = end_row_id - start_row_id

        # 0으로 채워진 output tensor 제작                  
        cate_feature = torch.zeros(self.max_seq_len, len(self.cate_cols), dtype=torch.long)
        cont_feature = torch.zeros(self.max_seq_len, len(self.cont_cols), dtype=torch.float)
        mask = torch.zeros(self.max_seq_len, dtype=torch.int16)
       
        # tensor에 값 채워넣기
        cate_feature[-seq_len:] = torch.ShortTensor(self.cate_features[start_row_id:end_row_id])
        cont_feature[-seq_len:] = torch.HalfTensor(self.cont_features[start_row_id:end_row_id])
        mask[-seq_len:] = 1        
            
        # answered_correctly가 cont_feature[-1]에 위치한다
        target = torch.FloatTensor([cont_feature[-1, -1]])

        # answered_correctly 및 relative_answered_correctly는
        # data leakage가 발생할 수 있으므로 0으로 모두 채운다
        cont_feature[-1, -1] = 0
        cont_feature[-1, -2] = 0
        
        return cate_feature, cont_feature, mask, target
        
    def __len__(self):
        return len(self.user_id_row_id_list)

train_db = IscreamDataset(train, CFG, train_user_id_row_id_list, train_start_row_id_by_user_id, max_seq_len=CFG.seq_len)
train_loader = DataLoader(train_db, batch_size=CFG.batch_size, shuffle=True,
                          drop_last=False, pin_memory=True)    
valid_db = IscreamDataset(valid, CFG, valid_user_id_row_id_list, valid_start_row_id_by_user_id, max_seq_len=CFG.seq_len)
valid_loader = DataLoader(valid_db, batch_size=CFG.batch_size, shuffle=True,
                          drop_last=False, pin_memory=True) 
test_db = IscreamDataset(test, CFG, test_user_id_row_id_list, test_start_row_id_by_user_id, max_seq_len=CFG.seq_len)
test_loader = DataLoader(test_db, batch_size=CFG.batch_size, shuffle=True,
                          drop_last=False, pin_memory=True)

## 📗 Transformer Input / Output 구현

import torch.nn as nn

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel    
except:
    from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel   

class TransformerModel(nn.Module):
    def __init__(self, cfg):
        super(TransformerModel, self).__init__()
        self.cfg = cfg

        cate_col_size = len(cfg.cate_cols)
        cont_col_size = len(cfg.cont_cols)

        # category
        self.cate_emb = nn.Embedding(cfg.total_cate_size, cfg.emb_size, padding_idx=0)
        self.cate_proj = nn.Sequential(
            nn.Linear(cfg.emb_size * cfg.cate_col_size * cfg.n_rows_per_step, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
        )

        # continuous
        self.cont_bn = nn.BatchNorm1d(cfg.cont_col_size)
        self.cont_emb = nn.Sequential(
            nn.Linear(cfg.cont_col_size*cfg.n_rows_per_step, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
        )

        # combination
        self.comb_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(cfg.hidden_size*2, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
        )
        
        self.config = BertConfig( 
            3, # not used
            hidden_size=cfg.hidden_size,
            num_hidden_layers=cfg.nlayers,
            num_attention_heads=cfg.nheads,
            intermediate_size=cfg.hidden_size,
            hidden_dropout_prob=cfg.dropout,
            attention_probs_dropout_prob=cfg.dropout,
        )
        self.encoder = BertEncoder(self.config)        
        
        def get_reg():
            return nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
            nn.Dropout(cfg.dropout),
            nn.ReLU(),            
            nn.Linear(cfg.hidden_size, cfg.target_size),
        )     
        self.reg_layer = get_reg()
        
    def forward(self, cate_x, cont_x, mask):        
        batch_size = cate_x.size(0)
        half_seq_len = cate_x.size(1) // self.cfg.n_rows_per_step
        
        # category
        cate_emb = self.cate_emb(cate_x).view(batch_size, half_seq_len, -1)
        cate_emb = self.cate_proj(cate_emb)

        # continuous
        cont_x = self.cont_bn(cont_x.view(-1, cont_x.size(-1))).view(batch_size, -1, cont_x.size(-1))
        cont_emb = self.cont_emb(cont_x.view(batch_size, half_seq_len, -1))        
        
        # combination
        seq_emb = torch.cat([cate_emb, cont_emb], 2)        
        seq_emb = self.comb_proj(seq_emb)   
        
        mask, _ = mask.view(batch_size, half_seq_len, -1).max(2)
        
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2) # (batch size) x 1 x 1 x (max_seq_length)
        encoded_layers = self.encoder(seq_emb, attention_mask=extended_attention_mask)
        sequence_output = encoded_layers[0]
        sequence_output = sequence_output[:, -1]        
        
        pred_y = self.reg_layer(sequence_output)

        return pred_y

import math
import os

import torch
import wandb

from src.criterion import get_criterion
from src.metric import get_metric
from src.optimizer import get_optimizer
from src.scheduler import get_scheduler


def run(CFG, train_loader, valid_loader, model):

    # only when using warmup scheduler
    CFG.total_steps = int(math.ceil(len(train_loader.dataset) / CFG.batch_size)) * (
        CFG.n_epochs
    )
    CFG.warmup_steps = CFG.total_steps // 10

    optimizer = get_optimizer(model, CFG)
    scheduler = get_scheduler(optimizer, CFG)

    best_auc = -1
    early_stopping_counter = 0
    for epoch in range(CFG.n_epochs):

        print(f"Start Training: Epoch {epoch + 1}")

        ### TRAIN
        train_auc, train_acc, train_loss = train(
            train_loader, model, optimizer, scheduler, CFG
        )

        ### VALID
        auc, acc = validate(valid_loader, model, CFG)

        ### TODO: model save or early stopping
        # wandb.log(
        #     {
        #         "epoch": epoch,
        #         "train_loss_epoch": train_loss,
        #         "train_auc_epoch": train_auc,
        #         "train_acc_epoch": train_acc,
        #         "valid_auc_epoch": auc,
        #         "valid_acc_epoch": acc,
        #     }
        # )
        if auc > best_auc:
            best_auc = auc
            # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, "module") else model
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model_to_save.state_dict(),
                },
                CFG.model_dir,
                "model.pt",
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= CFG.patience:
                print(
                    f"EarlyStopping counter: {early_stopping_counter} out of {CFG.patience}"
                )
                break

        # scheduler
        if CFG.scheduler == "plateau":
            scheduler.step(best_auc)

def train(train_loader, model, optimizer, scheduler, CFG):
    model.train()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(train_loader):
        cate_x, cont_x, mask, target = batch
        preds = model(cate_x, cont_x, mask)
        targets = target  # correct

        loss = compute_loss(preds, targets)
        update_params(loss, model, optimizer, scheduler, CFG)

        # if step % CFG.log_steps == 0:
            # print(f"Training steps: {step} Loss: {str(loss.item())}")

        # predictions
        preds = preds[:, -1]
        targets = targets[:, -1]

        total_preds.append(preds.detach())
        total_targets.append(targets.detach())
        losses.append(loss)

    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    loss_avg = sum(losses) / len(losses)
    print(f"TRAIN AUC : {auc} ACC : {acc}")
    return auc, acc, loss_avg

def validate(valid_loader, model, CFG):
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(valid_loader):
        cate_x, cont_x, mask, target = batch
        preds = model(cate_x, cont_x, mask)
        targets = target  # correct

        # predictions
        preds = preds[:, -1]
        targets = targets[:, -1]

        total_preds.append(preds.detach())
        total_targets.append(targets.detach())

    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)

    print(f"VALID AUC : {auc} ACC : {acc}\n")

    return auc, acc


def inference(CFG, test_loader, model):

    model.eval()

    total_preds = []

    for step, batch in enumerate(test_loader):
        cate_x, cont_x, mask, target = batch
        preds = model(cate_x, cont_x, mask)

        # predictions
        preds = preds[:, -1]
        preds = preds.cpu().detach().numpy()
        total_preds += list(preds)

    write_path = os.path.join(CFG.output_dir, "submission.csv")
    if not os.path.exists(CFG.output_dir):
        os.makedirs(CFG.output_dir)
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write("{},{}\n".format(id, p))

# loss계산하고 parameter update!
def compute_loss(preds, targets):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)

    """
    loss = get_criterion(preds, targets)

    # 마지막 시퀀드에 대한 값만 loss 계산
    loss = loss[:, -1]
    loss = torch.mean(loss)
    return loss


def update_params(loss, model, optimizer, scheduler, CFG):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.clip_grad)
    if CFG.scheduler == "linear_warmup":
        scheduler.step()
    optimizer.step()
    optimizer.zero_grad()

def save_checkpoint(state, model_dir, model_filename):
    print("saving model ...")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(state, os.path.join(model_dir, model_filename))

# wandb.login()
# wandb.init(project="dkt")
model = TransformerModel(CFG)
run(CFG, train_loader, valid_loader, model)