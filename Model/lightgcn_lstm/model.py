from typing import Optional, Union

from sklearn.metrics import roc_auc_score, accuracy_score

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, ModuleList
from torch.nn.modules.loss import _Loss
from torch_sparse import SparseTensor

from torch_geometric.nn.conv import LGConv
from torch_geometric.typing import Adj, OptTensor

class LightGCN_LSTM(torch.nn.Module):
    r"""The LightGCN model from the `"LightGCN: Simplifying and Powering
    Graph Convolution Network for Recommendation"
    <https://arxiv.org/abs/2002.02126>`_ paper.

    :class:`~torch_geometric.nn.models.LightGCN` learns embeddings by linearly
    propagating them on the underlying graph, and uses the weighted sum of the
    embeddings learned at all layers as the final embedding

    .. math::
        \textbf{x}_i = \sum_{l=0}^{L} \alpha_l \textbf{x}^{(l)}_i,

    where each layer's embedding is computed as

    .. math::
        \mathbf{x}^{(l+1)}_i = \sum_{j \in \mathcal{N}(i)}
        \frac{1}{\sqrt{\deg(i)\deg(j)}}\mathbf{x}^{(l)}_j.

    Two prediction heads and trainign objectives are provided:
    **link prediction** (via
    :meth:`~torch_geometric.nn.models.LightGCN.link_pred_loss` and
    :meth:`~torch_geometric.nn.models.LightGCN.predict_link`) and
    **recommendation** (via
    :meth:`~torch_geometric.nn.models.LightGCN.recommendation_loss` and
    :meth:`~torch_geometric.nn.models.LightGCN.recommend`).

    .. note::

        Embeddings are propagated according to the graph connectivity specified
        by :obj:`edge_index` while rankings or link probabilities are computed
        according to the edges specified by :obj:`edge_label_index`.

    Args:
        num_nodes (int): The number of nodes in the graph.
        embedding_dim (int): The dimensionality of node embeddings.
        num_layers (int): The number of
            :class:`~torch_geometric.nn.conv.LGConv` layers.
        alpha (float or Tensor, optional): The scalar or vector specifying the
            re-weighting coefficients for aggregating the final embedding.
            If set to :obj:`None`, the uniform initialization of
            :obj:`1 / (num_layers + 1)` is used. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of the underlying
            :class:`~torch_geometric.nn.conv.LGConv` layers.
    """
    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        hidden_dim: int,
        feature_len: int,
        num_layers: int,
        train_loader,
        valid_data,
        patience: int,
        alpha: Optional[Union[float, Tensor]] = None,
        **kwargs,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.feature_len = feature_len

        self.train_dataloader = train_loader

        self.valid_edge = torch.stack([i for i in valid_data['edge']])
        self.valid_feature = valid_data['feature']
        self.valid_label = valid_data['label']

        self.patience = patience

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
        
        if alpha is None:
            alpha = 1. / (num_layers + 1)

        if isinstance(alpha, Tensor):
            assert alpha.size(0) == num_layers + 1
        else:
            alpha = torch.tensor([alpha] * (num_layers + 1))
        self.register_buffer('alpha', alpha)

        self.embedding = Embedding(self.num_nodes, self.embedding_dim)
        self.convs = ModuleList([LGConv(**kwargs) for _ in range(self.num_layers)])
        self.embedding_feature = Embedding(self.feature_len, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim*2, self.hidden_dim)
        self.ln = nn.Linear(self.hidden_dim, 1)
        self.reset_parameters()

        self.model = self

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


    def forward(self, edge_index: Adj, feature,
                edge_label_index: OptTensor = None) -> Tensor:
        r"""Computes rankings for pairs of nodes.

        Args:
            edge_index (Tensor or SparseTensor): Edge tensor specifying the
                connectivity of the graph.
            edge_label_index (Tensor, optional): Edge tensor specifying the
                node pairs for which to compute rankings or probabilities.
                If :obj:`edge_label_index` is set to :obj:`None`, all edges in
                :obj:`edge_index` will be used instead. (default: :obj:`None`)
        """
        if edge_label_index is None:
            if isinstance(edge_index, SparseTensor):
                edge_label_index = torch.stack(edge_index.coo()[:2], dim=0)
            else:
                edge_label_index = edge_index

        out = self.get_embedding(edge_index)
        out_src = out[edge_label_index[0]]
        out_dst = out[edge_label_index[1]]
        
        feature = self.embedding_feature(feature)
        out = torch.concat([(out_src * out_dst), feature], 1)
        
        out, _ = self.lstm(out)
        out = self.ln(out)
        return out.view(-1)

    def predict_link(self, edge_index: Adj, edge_label_index: OptTensor = None,
                     prob: bool = False) -> Tensor:
        r"""Predict links between nodes specified in :obj:`edge_label_index`.

        Args:
            prob (bool): Whether probabilities should be returned. (default:
                :obj:`False`)
        """
        pred = self(edge_index, edge_label_index).sigmoid()
        return pred if prob else pred.round()


    def recommend(self, edge_index: Adj, src_index: OptTensor = None,
                  dst_index: OptTensor = None, k: int = 1) -> Tensor:
        r"""Get top-:math:`k` recommendations for nodes in :obj:`src_index`.

        Args:
            src_index (Tensor, optional): Node indices for which
                recommendations should be generated.
                If set to :obj:`None`, all nodes will be used.
                (default: :obj:`None`)
            dst_index (Tensor, optional): Node indices which represent the
                possible recommendation choices.
                If set to :obj:`None`, all nodes will be used.
                (default: :obj:`None`)
            k (int, optional): Number of recommendations. (default: :obj:`1`)
        """
        out_src = out_dst = self.get_embedding(edge_index)

        if src_index is not None:
            out_src = out_src[src_index]

        if dst_index is not None:
            out_dst = out_dst[dst_index]

        pred = out_src @ out_dst.t()
        top_index = pred.topk(k, dim=-1).indices

        if dst_index is not None:  # Map local top-indices to original indices.
            top_index = dst_index[top_index.view(-1)].view(*top_index.size())

        return top_index


    def link_pred_loss(self, pred: Tensor, edge_label: Tensor,
                       **kwargs) -> Tensor:
        r"""Computes the model loss for a link prediction objective via the
        :class:`torch.nn.BCEWithLogitsLoss`.

        Args:
            pred (Tensor): The predictions.
            edge_label (Tensor): The ground-truth edge labels.
            **kwargs (optional): Additional arguments of the underlying
                :class:`torch.nn.BCEWithLogitsLoss` loss function.
        """
        loss_fn = torch.nn.BCEWithLogitsLoss(**kwargs)
        return loss_fn(pred, edge_label.to(pred.dtype))


    def recommendation_loss(self, pos_edge_rank: Tensor, neg_edge_rank: Tensor,
                            lambda_reg: float = 1e-4, **kwargs) -> Tensor:
        r"""Computes the model loss for a ranking objective via the Bayesian
        Personalized Ranking (BPR) loss.

        .. note::

            The i-th entry in the :obj:`pos_edge_rank` vector and i-th entry
            in the :obj:`neg_edge_rank` entry must correspond to ranks of
            positive and negative edges of the same entity (*e.g.*, user).

        Args:
            pos_edge_rank (Tensor): Positive edge rankings.
            neg_edge_rank (Tensor): Negative edge rankings.
            lambda_reg (int, optional): The :math:`L_2` regularization strength
                of the Bayesian Personalized Ranking (BPR) loss.
                (default: 1e-4)
            **kwargs (optional): Additional arguments of the underlying
                :class:`torch_geometric.nn.models.lightgcn.BPRLoss` loss
                function.
        """
        loss_fn = BPRLoss(lambda_reg, **kwargs)
        return loss_fn(pos_edge_rank, neg_edge_rank, self.embedding.weight)


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_nodes}, '
                f'{self.embedding_dim}, num_layers={self.num_layers})')

    def run(self) :
        for epoch in range(1001):
            total_loss, total_acc, total_auc = self.train()
            
            if not epoch % 1:
                print(f" * In epoch {(epoch+1):04}, loss={total_loss:.03f}, acc={total_acc:.03f}, AUC={total_auc:.03f}")    

            loss, acc, auc = self.validate()

            if auc > best_auc :
                best_auc = auc
                best_acc = acc
                torch.save(
                    {
                        "state_dict": self.model.state_dict()
                    },
                    "model_2.pt",
                )
                count = 0
            else :
                count += 1
                
            if count == self.patience :
                break

        if not epoch % 1:
            print(f" * In epoch {(epoch+1):04}, val_loss={loss:.03f}, val_acc={acc:.03f}, val_AUC={auc:.03f}")
                
        print(f"best_acc={best_acc:.03f}, best_AUC={best_auc:.03f}")

    
    def train(self) :
        total_loss, total_acc, total_auc = 0.0, 0.0, 0.0

        self.model.train()
        for edge, feature, label in tqdm(self.train_dataloader, total=len(self.train_dataloader)):
            preds = self.model(edge.T, feature)
            loss = self.model.link_pred_loss(preds, label)
            
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            prob = preds.sigmoid()
            prob = prob.detach().cpu().numpy()
            acc = accuracy_score(label.detach().cpu().numpy(), prob > 0.5)
            auc = roc_auc_score(label.detach().cpu().numpy(), prob)
            total_loss += loss.item()
            total_acc += acc
            total_auc += auc
        
        return total_loss/len(self.train_dataloader), total_acc/len(self.train_dataloader), total_auc/len(self.train_dataloader)
                
            
    def validate(self) :
        with torch.no_grad() :
            self.model.eval()
            preds = self.model(self.valid_edge, self.valid_feature)
            loss = self.model.link_pred_loss(preds, self.valid_label)

            prob = preds.sigmoid()
            prob = prob.detach().cpu().numpy()
            acc = accuracy_score(self.valid_label.detach().cpu().numpy(), prob > 0.5)
            auc = roc_auc_score(self.valid_label.detach().cpu().numpy(), prob)

        return loss, acc, auc

    def prediction(self, test_edge, features) :
        self.model.load_state_dict(torch.load('model_2.pt')['state_dict'])

        self.model.eval()
        prediction = self.model(test_edge, features).sigmoid()

        return prediction


class BPRLoss(_Loss):
    r"""The Bayesian Personalized Ranking (BPR) loss.

    The BPR loss is a pairwise loss that encourages the prediction of an
    observed entry to be higher than its unobserved counterparts
    (see `here <https://arxiv.org/abs/2002.02126>`__).

    .. math::
        L_{\text{BPR}} = - \sum_{u=1}^{M} \sum_{i \in \mathcal{N}_u}
        \sum_{j \not\in \mathcal{N}_u} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj})
        + \lambda \vert\vert \textbf{x}^{(0)} \vert\vert^2

    where :math:`lambda` controls the :math:`L_2` regularization strength.
    We compute the mean BPR loss for simplicity.

    Args:
        lambda_reg (float, optional): The :math:`L_2` regularization strength
            (default: 0).
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch.nn.modules.loss._Loss` class.
    """
    __constants__ = ['lambda_reg']
    lambda_reg: float

    def __init__(self, lambda_reg: float = 0, **kwargs) -> None:
        super().__init__(None, None, "sum", **kwargs)
        self.lambda_reg = lambda_reg

    def forward(self, positives: Tensor, negatives: Tensor,
                parameters: Tensor = None) -> Tensor:
        r"""Compute the mean Bayesian Personalized Ranking (BPR) loss.

        .. note::

            The i-th entry in the :obj:`positives` vector and i-th entry
            in the :obj:`negatives` entry should correspond to the same
            entity (*.e.g*, user), as the BPR is a personalized ranking loss.

        Args:
            positives (Tensor): The vector of positive-pair rankings.
            negatives (Tensor): The vector of negative-pair rankings.
            parameters (Tensor, optional): The tensor of parameters which
                should be used for :math:`L_2` regularization
                (default: :obj:`None`).
        """
        n_pairs = positives.size(0)
        log_prob = F.logsigmoid(positives - negatives).mean()
        regularization = 0

        if self.lambda_reg != 0:
            regularization = self.lambda_reg * parameters.norm(p=2).pow(2)

        return (-log_prob + regularization) / n_pairs