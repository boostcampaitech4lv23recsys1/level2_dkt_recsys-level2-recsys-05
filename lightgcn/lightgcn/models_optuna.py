from typing import Optional, Union
import os
import numpy as np
import torch
from torch import Tensor
from sklearn.metrics import accuracy_score, roc_auc_score
from torch_geometric.nn.models import LightGCN
from lightgcn.utils import CosineAnnealingWarmUpRestarts, get_metric
from lightgcn.datasets import LightgcnDataset
from tqdm import tqdm

class LightGCN_context(torch.nn.Module):
    def __init__(self, batch_size, n_node, embedding_dim: int, num_layers: int, alpha: Optional[Union[float, Tensor]] = None, **kwargs,):
        super().__init__()
        self.num_nodes = n_node
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.alpha = alpha
        self.kwargs = kwargs
        self.batch_size = batch_size

        self.a = torch.nn.Parameter(torch.ones(1), requires_grad=True)

        self.linear1 = torch.nn.Linear(21, 128)
        self.linear2 = torch.nn.Linear(128, 256)
        self.linear3 = torch.nn.Linear(256, 512)
        self.linear4 = torch.nn.Linear(512, 1024)
        self.linear5 = torch.nn.Linear(1024, 2048)
        self.linear6 = torch.nn.Linear(2048, 1024)
        self.linear7 = torch.nn.Linear(1024, 512)
        self.linear8 = torch.nn.Linear(512, 256)
        self.linear9 = torch.nn.Linear(256, 128)
        self.linear10 = torch.nn.Linear(128, 21)
        self.linear11 = torch.nn.Linear(21, 1)

        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128 * 2)
        self.bn3 = torch.nn.BatchNorm1d(128 * 4)
        self.bn4 = torch.nn.BatchNorm1d(128 * 8)
        self.bn5 = torch.nn.BatchNorm1d(128 * 16)
        self.bn6 = torch.nn.BatchNorm1d(128 * 8)
        self.bn7 = torch.nn.BatchNorm1d(128 * 4)
        self.bn8 = torch.nn.BatchNorm1d(128 * 2)
        self.bn9 = torch.nn.BatchNorm1d(128)
        self.bn10 = torch.nn.BatchNorm1d(21)
        
        self.relu1 = torch.nn.ReLU(128)
        self.relu2 = torch.nn.ReLU(128 * 2)
        self.relu3 = torch.nn.ReLU(128 * 4)
        self.relu4 = torch.nn.ReLU(128 * 8)
        self.relu5 = torch.nn.ReLU(128 * 16)
        self.relu6 = torch.nn.ReLU(128 * 8)
        self.relu7 = torch.nn.ReLU(128 * 4)
        self.relu8 = torch.nn.ReLU(128 * 2)
        self.relu9 = torch.nn.ReLU(128)
        self.relu10 = torch.nn.ReLU(21)
        self.context_model = torch.nn.Sequential(self.linear1, self.bn1, self.relu1,
                                                 self.linear2, self.bn2, self.relu2,
                                                 self.linear3, self.bn3, self.relu3,
                                                 self.linear4, self.bn4, self.relu4,
                                                 self.linear5, self.bn5, self.relu5,
                                                 self.linear6, self.bn6, self.relu6,
                                                 self.linear7, self.bn7, self.relu7,
                                                 self.linear8, self.bn8, self.relu8,
                                                 self.linear9, self.bn9, self.relu9,
                                                 self.linear10, self.bn10, self.relu10,
                                                 )
        self.lightgcn = LightGCN(self.num_nodes, self.embedding_dim, self.num_layers, self.alpha, **self.kwargs)

    def forward(self, edge, context):
        x_ = context
        # import pdb;pdb.set_trace();
        x = self.context_model(context)
        x = x + x_
        x = self.linear11(x)
        graph = self.lightgcn(edge.T)
        return (x * graph.view(-1,1)).view(-1)

# def build(n_node, weight=None, logger=None, **kwargs):
#     model = LightGCN(n_node, **kwargs)
#     if weight:
#         if not os.path.isfile(weight):
#             logger.fatal("Model Weight File Not Exist")
#         logger.info("Load model")
#         state = torch.load(weight)["model"]
#         model.load_state_dict(state)
#         return model
#     else:
#         logger.info("No load model")
#         return model

def build(batch_size, n_node, weight=None, logger=None, **kwargs):
    model = LightGCN_context(batch_size, n_node, **kwargs)
    if weight:
        if not os.path.isfile(weight):
            logger.fatal("Model Weight File Not Exist")
        logger.info("Load model")
        state = torch.load(weight)["model"]
        model.load_state_dict(state)
        return model
    else:
        logger.info("No load model")
        return model

def train(
    model,
    train_data,
    valid_data=None,
    n_epoch=100,
    learning_rate=0.01,
    use_wandb=False,
    weight=None,
    logger=None,
    batch_size=64,
    device='cpu',
    # params=None,
):
    # wandb.config.n_epoch = n_epoch
    # wandb.config.learning_rate = learning_rate
    # wandb.config.embedding_dim = params['embedding_dim']
    # wandb.config.num_layers = params['num_layers']
    # wandb.config.alpha = params['alpha']

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=30, T_mult=1, eta_max=0.1,  T_up=10, gamma=0.5)
    # scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=45, T_mult=1, eta_max=0.1,  T_up=10, gamma=0.5)
    # scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=50, T_mult=1, eta_max=0.1,  T_up=10, gamma=0.7)

    train_dataset = LightgcnDataset(train_data["train_edge"], train_data["train_context"], train_data["train_label"])
    valid_dataset = LightgcnDataset(valid_data["valid_edge"], valid_data["valid_context"], valid_data["valid_label"])
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            num_workers=8,
            shuffle=True,
            batch_size=batch_size
        )
    valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            num_workers=8,
            shuffle=False,
            batch_size=batch_size
        )
    if not os.path.exists(weight):
        os.makedirs(weight)

    # if valid_data is None:
    #     eids = np.arange(len(train_data["label"]))
    #     eids = np.random.permutation(eids)[:1000]
    #     edge, label = train_data["edge"], train_data["label"]
    #     label = label.to("cpu").detach().numpy()
    #     valid_data = dict(edge=edge[:, eids], label=label[eids])

    # import pdb;pdb.set_trace();
    logger.info(f"Training Started : n_epoch={n_epoch} learning_rate={learning_rate}")
    best_auc, best_epoch = 0, -1
    early_stopping_counter = 0

    for e in range(n_epoch):
        total_preds = []
        total_targets = []
        losses = []
        model.train()
        model.lightgcn.train();
        for step, batch in enumerate(train_loader):
        # forward
        # pred = model(train_data["train_edge"])
        # loss = model.link_pred_loss(pred, train_data["train_label"])
            edge, context, label = batch;
            edge, context, label = edge.to(device), context.to(device), label.to(device);
            pred = model(edge, context);
            loss = model.lightgcn.link_pred_loss(pred, label)

            # backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            
            total_preds.append(pred.detach())
            total_targets.append(label.detach())
            losses.append(loss)
        total_preds = torch.concat(total_preds).cpu().numpy()
        total_targets = torch.concat(total_targets).cpu().numpy()
        
        auc, acc = get_metric(total_targets, total_preds)
        loss_avg = sum(losses) / len(losses)
        
        print(f"[TRAIN] Epoch : {e+1} AUC : {auc} ACC : {acc} LOSS : {loss_avg}")
        # scheduler.step(e)
        
        
        with torch.no_grad():
            model.eval()
            model.lightgcn.eval()
            total_preds = []
            total_targets = []
            losses = []
            for step, batch in enumerate(valid_loader):
            # prob = model.predict_link(valid_data["valid_edge"], prob=True)
                edge, context, label = batch;
                edge, context, label = edge.to(device), context.to(device), label.to(device);
                prob = model(edge, context)
                loss = model.lightgcn.link_pred_loss(prob, label)
                
                total_preds.append(prob.detach())
                total_targets.append(label.detach())
                losses.append(loss.item())
            total_preds = torch.concat(total_preds).cpu().numpy()
            total_targets = torch.concat(total_targets).cpu().numpy()
            # losses = torch.concat(losses).cpu().numpy()
            # Train AUC / ACC
            auc, acc = get_metric(total_targets, total_preds)
            loss_avg = sum(losses) / len(losses)
            print(f"VALID AUC : {auc} ACC : {acc} LOSS : {loss_avg}\n")
                # acc = accuracy_score(valid_data["valid_label"], prob > 0.5)
                # auc = roc_auc_score(valid_data["valid_label"], prob)
            # logger.info(
                    # f" * In epoch {(e+1):04}, loss={loss:.05f}, acc={acc:.05f}, AUC={auc:.05f}"
                # )
            if use_wandb:
                import wandb
                wandb.log(dict(loss=loss, acc=acc, auc=auc))

        if weight:
            if auc > best_auc:
                logger.info(
                    f" * In epoch {(e+1):04}, loss={loss_avg:.05f}, acc={acc:.05f}, AUC={auc:.05f}, Best AUC"
                )
                best_auc, best_epoch = auc, e
                if best_auc >= 0.8335:
                    torch.save(
                        {"model": model.state_dict(), "epoch": e + 1},
                        os.path.join(weight, f"model_{best_auc}_{best_epoch}.pt"),
                    )
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter > 20:
                    print(
                        f"EarlyStopping counter: {early_stopping_counter} out of {20}"
                    )
                    break
    torch.save(
        {"model": model.state_dict(), "epoch": e + 1},
        os.path.join(weight, f"last_model.pt"),
    )
    logger.info(f"Best Weight Confirmed : {best_epoch+1}'th epoch")
    return best_auc

def inference(model, data, logger=None):
    model.eval()
    with torch.no_grad():
        pred = model.predict_link(data["edge"], prob=True)
        return pred