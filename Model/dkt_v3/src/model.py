import torch
import torch.nn as nn

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel
except:
    from transformers.models.bert.modeling_bert import (
        BertConfig,
        BertEncoder,
        BertModel,
    )


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args

        self.embed_dim = self.args.embed_dim
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.cate_col_size = len(args.cate_cols)
        self.cont_col_size = len(args.cont_cols)

        # category
        self.cate_embed = nn.Embedding(args.offset, args.embed_dim, padding_idx=0)
        self.cate_proj = nn.Sequential(
            nn.Linear(self.embed_dim * self.cate_col_size, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        # continuous
        self.cont_bn = nn.BatchNorm1d(self.cont_col_size)
        self.cont_embed = nn.Sequential(
            nn.Linear(self.cont_col_size, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        # combination
        self.comb_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

    def forward(self, input):
        # (cate_x, cont_x, mask)
        # cate_x, cont_x : (batch_size) x (max_seq_len) x (n_cols) 
        # mask : (batch_size) x (max_seq_len)
        cate_x, cont_x, mask = input
        batch_size = cate_x.size(0)
        max_seq_len = cate_x.size(1)

        # category
        cate_emb = self.cate_embed(cate_x).view(batch_size, max_seq_len, -1)
        cate_emb = self.cate_proj(cate_emb)

        # continuous
        cont_x = self.cont_bn(cont_x.view(-1, cont_x.size(-1))).view(batch_size, -1, cont_x.size(-1))
        cont_emb = self.cont_embed(cont_x)   
        
        # combination
        seq_emb = torch.cat([cate_emb, cont_emb], 2)        
        seq_emb = self.comb_proj(seq_emb)   

        out, _ = self.lstm(seq_emb) # (batch size) x (max_seq_length) x (hidden_dim)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out


class LSTMATTN(nn.Module):
    def __init__(self, args):
        super(LSTMATTN, self).__init__()
        self.args = args

        self.embed_dim = self.args.embed_dim
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out
        self.cate_col_size = len(args.cate_cols)
        self.cont_col_size = len(args.cont_cols)

        # category
        self.cate_embed = nn.Embedding(args.offset, args.embed_dim, padding_idx=0)
        self.cate_proj = nn.Sequential(
            nn.Linear(self.embed_dim * self.cate_col_size, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        # continuous
        self.cont_bn = nn.BatchNorm1d(self.cont_col_size)
        self.cont_embed = nn.Sequential(
            nn.Linear(self.cont_col_size, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        # combination
        self.comb_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def forward(self, input):
        # (cate_x, cont_x, mask)
        # cate_x, cont_x : (batch_size) x (max_seq_len) x (n_cols)
        # mask : (batch_size) x (max_seq_len)
        cate_x, cont_x, mask = input
        batch_size = cate_x.size(0)
        max_seq_len = cate_x.size(1)

        # category
        cate_emb = self.cate_embed(cate_x).view(batch_size, max_seq_len, -1)
        cate_emb = self.cate_proj(cate_emb)

        # continuous
        cont_x = self.cont_bn(cont_x.view(-1, cont_x.size(-1))).view(batch_size, -1, cont_x.size(-1))
        cont_emb = self.cont_embed(cont_x)   
        
        # combination
        seq_emb = torch.cat([cate_emb, cont_emb], 2)        
        seq_emb = self.comb_proj(seq_emb)   

        out, _ = self.lstm(seq_emb) # (batch size) x (max_seq_length) x (hidden_dim)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        ###################### 여기까지 LSTM과 동일 ######################

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2) # (batch size) x 1 x 1 x (max_seq_length)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output).view(batch_size, -1)
        return out


class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args

        self.embed_dim = self.args.embed_dim
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.cate_col_size = len(args.cate_cols)
        self.cont_col_size = len(args.cont_cols)

        # category
        self.cate_embed = nn.Embedding(args.offset, args.embed_dim, padding_idx=0)
        self.cate_proj = nn.Sequential(
            nn.Linear(self.embed_dim * self.cate_col_size, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        # continuous
        self.cont_bn = nn.BatchNorm1d(self.cont_col_size)
        self.cont_embed = nn.Sequential(
            nn.Linear(self.cont_col_size, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        # combination
        self.comb_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        
        # Bert config
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len,
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def forward(self, input):
        # (cate_x, cont_x, mask)
        # cate_x, cont_x : (batch_size) x (max_seq_len) x (n_cols)
        # mask : (batch_size) x (max_seq_len)
        cate_x, cont_x, mask = input
        batch_size = cate_x.size(0)
        max_seq_len = cate_x.size(1)

        # category
        cate_emb = self.cate_embed(cate_x).view(batch_size, max_seq_len, -1)
        cate_emb = self.cate_proj(cate_emb)

        # continuous
        cont_x = self.cont_bn(cont_x.view(-1, cont_x.size(-1))).view(batch_size, -1, cont_x.size(-1))
        cont_emb = self.cont_embed(cont_x)   
        
        # combination
        seq_emb = torch.cat([cate_emb, cont_emb], 2)        
        seq_emb = self.comb_proj(seq_emb)   

        # Bert
        encoded_layers = self.encoder(inputs_embeds=seq_emb, attention_mask=mask)
        out = encoded_layers[0]

        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out).view(batch_size, -1)
        return out

