import math
import os

import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from .criterion import get_criterion
from .dataloader import get_loaders
from .metric import get_metric
from .model import LSTM, LSTMATTN, Bert, LastQuery, Saint
from .optimizer import get_optimizer
from .scheduler import get_scheduler

from pytorch_tabnet.multitask import TabNetMultiTaskClassifier


def run(args, train_data, valid_data, model, kfold=''):
    print('-'*50)
    print(f'FOLD {kfold}')
    print('-'*50)
    train_loader, valid_loader = get_loaders(args, train_data, valid_data)
    
    # only when using warmup scheduler
    args.total_steps = int(math.ceil(len(train_loader.dataset) / args.batch_size)) * (
        args.n_epochs
    )
    args.warmup_steps = args.total_steps // 10

    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    best_auc = -1
    early_stopping_counter = 0
    for epoch in range(args.n_epochs):

        print(f"Start Training: Epoch {epoch + 1}")

        ### TRAIN
        train_auc, train_acc, train_loss = train(
            train_loader, model, optimizer, scheduler, args
        )

        ### VALID
        auc, acc = validate(valid_loader, model, args)

        ### TODO: model save or early stopping
        # wandb.log(
        #     {
        #         "epoch": epoch+1,
        #         "train_loss_epoch": train_loss,
        #         "train_auc_epoch": train_auc,
        #         "train_acc_epoch": train_acc,
        #         "valid_auc_epoch": auc,
        #         "valid_acc_epoch": acc,
        #     }
        # )
        if auc > best_auc:
            best_epoch = epoch+1
            best_auc = auc
            best_acc = acc
            best_train_auc = train_auc
            best_train_acc = train_acc
            best_train_loss = train_loss
            early_stopping_counter = 0
            # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, "module") else model
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model_to_save.state_dict(),
                },
                args.model_dir,
                args.model_name + kfold + '.pt',
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                print(
                    f"EarlyStopping counter: {early_stopping_counter} out of {args.patience}"
                )
                break

        # scheduler
        if args.scheduler == "plateau":
            scheduler.step(best_auc)
    print('-'*50)
    print(f"BEST EPOCHS : {best_epoch} TRAIN LOSS : {best_train_loss}")
    print(f"TRAIN AUC : {best_train_auc} ACC : {best_train_acc}")
    print(f"VALID AUC : {best_auc} ACC : {best_acc}\n")
    print('-'*50)

    # wandb.log(
    #     {
    #         "best_epoch": best_epoch,
    #         "best_train_loss": best_train_loss,
    #         "best_train_auc": best_train_auc,
    #         "best_train_acc": best_train_acc,
    #         "best_valid_auc": best_auc,
    #         "best_valid_acc": best_acc,
    #     }
    # )

    return best_auc, best_acc

def tabnet_run(args):
    df = pd.read_csv('/opt/ml/input/data/train_feature_engineering.csv')
    categorical_columns = ['userID', 'assessmentItemID', 'testId', 'KnowledgeTag', 'assessmentItemID_last', 'testId_first', 'testId_last']
    categorical_dims =  {}
    for col in tqdm(df.columns):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].values)
        categorical_dims[col] = len(le.classes_)

    n = len(df['userID'].unique())
    user_permute_list = np.random.permutation(df['userID'].unique())
    train_userid = user_permute_list[:(int(n*ratio))]
    valid_userid = user_permute_list[(int(n*ratio)):]
    train = df[df['userID'].isin(train_userid)].reset_index(drop=True).sort_values(["userID", "Timestamp"])
    valid = df[df['userID'].isin(valid_userid)].reset_index(drop=True).sort_values(["userID", "Timestamp"])

    train_x = train.drop(['answerCode', 'Timestamp','relative_answered_correctly'], axis = 1)
    train_y = train[['answerCode']]
    valid_x = valid.drop(['answerCode', 'Timestamp', 'relative_answered_correctly'], axis = 1)
    valid_y = valid[['answerCode']]

    features = [ col for col in train_x.columns] 

    cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

    cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

    X_train = train_x[features].values
    y_train = train_y.values
    X_valid = valid_x[features].values
    y_valid = valid_y.values

    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=['train', 'valid'],
        eval_metric=['auc'],
        max_epochs=max_epochs,
        patience=30,
        batch_size=14598,
        virtual_batch_size=4430,
        drop_last=False,
    )

    max_epochs = 53
    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=['train', 'valid'],
        eval_metric=['auc'],
        max_epochs=max_epochs,
        patience=30,
        batch_size=14598,
        virtual_batch_size=4430,
        drop_last=False,
    )

    clf.save_model('./tabnet_model')


def train(train_loader, model, optimizer, scheduler, args):
    model.train()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
    # for step, batch in enumerate(train_loader):
        input = list(map(lambda t: t.to(args.device), batch)) # cate_x, cont_x, mask, target
        targets = input[-1] 
        preds = model(input[:-1])
        
        loss = compute_loss(preds, targets)
        update_params(loss, model, optimizer, scheduler, args)

        # if step % args.log_steps == 0:
        #     print(f"Training steps: {step} Loss: {str(loss.item())}")

        # predictions
        sigmoid = nn.Sigmoid()
        preds = sigmoid(preds)[:, -1]
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


def validate(valid_loader, model, args):
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
    # for step, batch in enumerate(valid_loader):
        input = list(map(lambda t: t.to(args.device), batch))
        targets = input[-1]
        preds = model(input[:-1])

        # predictions
        sigmoid = nn.Sigmoid()
        preds = sigmoid(preds)[:, -1]
        targets = targets[:, -1]

        total_preds.append(preds.detach())
        total_targets.append(targets.detach())

    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)

    print(f"VALID AUC : {auc} ACC : {acc}\n")

    return auc, acc


def inference(args, test_data):
    kfold = int(args.kfold) if args.kfold else 1

    kfold_preds = np.array([0] * test_data.userID.nunique(), dtype=float)
    for k in range(kfold):
        if args.kfold: 
            model_name = args.model_name + f"{k+1}.pt"
        else:
            model_name = args.model_name + ".pt"
        model = load_model(args, model_name).to(args.device)

        model.eval()
        _, test_loader = get_loaders(args, train=None, valid=test_data)

        total_preds = []
        
        for step, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        # for step, batch in enumerate(test_loader):
            input = list(map(lambda t: t.to(args.device), batch))
            preds = model(input[:-1])
            
            # predictions
            sigmoid = nn.Sigmoid()
            preds = sigmoid(preds)[:, -1]
            preds = preds.cpu().detach().numpy()
            total_preds += list(preds)
        
        kfold_preds += total_preds
    
    breakpoint()
    kfold_preds /= kfold

    write_path = os.path.join(args.output_dir, args.output_file_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(kfold_preds):
            w.write("{},{}\n".format(id, p))

def tabnet_inference(args):
    test = pd.read_csv('/opt/ml/input/data/test_feature_engineering.csv')
    md = TabNetMultiTaskClassifier(device_name='cpu')
    md.load_model('./tabnet_model.zip')

    test.pop('Timestamp')
    test.pop('relative_answered_correctly')
    test.pop('answerCode')

    for col in tqdm(test.columns):
        le = LabelEncoder()
        test[col] = le.fit_transform(test[col].values)

    test_data = test[test['userID'] != test['userID'].shift(-1)].to_numpy()
    loaded_preds = md.predict_proba(test_data)
    preds = loaded_preds[0][:, 1]

    output_dir = './output'
    write_path = os.path.join(output_dir, "tabnetmulti.csv")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(preds):
            w.write('{},{}\n'.format(id,p))




def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    if args.model == "lstm":
        model = LSTM(args)
    if args.model == "lstmattn":
        model = LSTMATTN(args)
    if args.model == "bert":
        model = Bert(args)
    if args.model == "lastquery":
        model = LastQuery(args)
    if args.model == "saint":
        model = Saint(args)


    return model


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


def update_params(loss, model, optimizer, scheduler, args):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    if args.scheduler == "linear_warmup":
        scheduler.step()
    optimizer.step()
    optimizer.zero_grad()


def save_checkpoint(state, model_dir, model_filename):
    print("saving model ...")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(state, os.path.join(model_dir, model_filename))


def load_model(args, model_name):

    model_path = os.path.join(args.model_dir, model_name)
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # load model state
    model.load_state_dict(load_state["state_dict"], strict=True)

    print("Loading Model from:", model_path, "...Finished.")
    return model
