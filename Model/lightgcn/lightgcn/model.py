from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
from torch.nn import Embedding, ModuleList, Linear
from torch.nn.modules.loss import _Loss
from torch_sparse import SparseTensor

from torch_geometric.nn.conv import LGConv
from torch_geometric.typing import Adj, OptTensor


class LightGCN_context(torch.nn.Module):

    def __init__(
        self,
        num_nodes: int,
        offset: int,
        embedding_dim: int,
        num_layers: int,
        cate_col_size: int,
        cont_col_size: int,
        alpha: Optional[Union[float, Tensor]] = None,
        **kwargs,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        if alpha is None:
            alpha = 1. / (num_layers + 1)

        if isinstance(alpha, Tensor):
            assert alpha.size(0) == num_layers + 1
        else:
            alpha = torch.tensor([alpha] * (num_layers + 1))
        self.register_buffer('alpha', alpha)

        self.embedding = Embedding(num_nodes, embedding_dim)
        self.convs = ModuleList([LGConv(**kwargs) for _ in range(num_layers)])

        self.reset_parameters()
        
        # category
        self.cate_embed = nn.Embedding(offset, embedding_dim, padding_idx=0)
        self.cate_proj = nn.Sequential(
            nn.Linear(embedding_dim * cate_col_size, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # continuous
        self.cont_bn = nn.BatchNorm1d(cont_col_size)
        self.cont_embed = nn.Sequential(
            nn.Linear(cont_col_size, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # combination
        self.comb_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embedding_dim*2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # Fully connected layer
        self.fc = nn.Linear(embedding_dim*3, 1)


    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        for conv in self.convs:
            conv.reset_parameters()


    def get_embedding(self, edge_index: Adj) -> Tensor:
        
        x = self.embedding.weight
        out = x * self.alpha[0]

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            out = out + x * self.alpha[i + 1]

        return out


    def forward(self, edge_index: Adj,
                cate_features, cont_features) -> Tensor:
       
        if isinstance(edge_index, SparseTensor):
            edge_label_index = torch.stack(edge_index.coo()[:2], dim=0)
        else:
            edge_label_index = edge_index
        out = self.get_embedding(edge_index)
        
        out_src = out[edge_label_index[0]] # n x embedding_dim
        out_dst = out[edge_label_index[1]] # n x embedding_dim
        
        cate_features = torch.transpose(cate_features, 0, 1) # n x cate_col_size
        cont_features = torch.transpose(cont_features, 0, 1) # n x cont_col_size
        batch_size = cate_features.shape[0]

        # category
        cate_emb = self.cate_embed(cate_features).view(batch_size, -1) # batch_size x cate_col_size*embedding_dim
        cate_emb = self.cate_proj(cate_emb) # batch_size x embedding_dim

        # continuous
        cont_x = self.cont_bn(cont_features)
        cont_emb = self.cont_embed(cont_x)   
        
        # combination
        seq_emb = torch.cat([cate_emb, cont_emb], 1)        
        seq_emb = self.comb_proj(seq_emb)   

        return self.fc(torch.cat([out_src, out_dst, seq_emb],1)).view(-1)

    def predict_link(self, edge_index: Adj, cate_features, cont_features,
                     prob: bool = False) -> Tensor:

        pred = self(edge_index, cate_features, cont_features).sigmoid()

        return pred if prob else pred.round()

    
    def link_pred_loss(self, pred: Tensor, edge_label: Tensor,
                       **kwargs) -> Tensor:
        loss_fn = torch.nn.BCEWithLogitsLoss(**kwargs)
        return loss_fn(pred, edge_label.to(pred.dtype))