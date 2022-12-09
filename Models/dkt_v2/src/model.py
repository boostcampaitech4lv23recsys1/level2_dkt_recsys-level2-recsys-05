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


class Feed_Forward_block(nn.Module):
    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff, out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff, out_features=dim_ff)

    def forward(self,ffn_in):
        return self.layer2(torch.nn.functional.relu(self.layer1(ffn_in)))

class LastQuery(nn.Module):
    def __init__(self, args):
        super(LastQuery, self).__init__()
        self.args = args
        self.device = args.device
        self.embed_dim = self.args.embed_dim

        self.hidden_dim = self.args.hidden_dim
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        
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

        # Encoder
        self.query = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.key = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.value = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)

        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.args.n_heads)
        self.ffn = Feed_Forward_block(self.hidden_dim)      

        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)
       
        self.activation = nn.Sigmoid()
        
    # lastquery는 2D mask가 필요함
    def get_mask(self, seq_len, mask, batch_size):
        new_mask = torch.zeros_like(mask)
        new_mask[mask == 0] = 1
        new_mask[mask != 0] = 0
        tmask = new_mask
    
        # batchsize * n_head 수만큼 각 mask를 반복하여 증가시킨다
        tmask = tmask.repeat(1, self.args.n_heads).view(batch_size*self.args.n_heads, -1, seq_len)
        return tmask.masked_fill(tmask==1, float('-inf'))

    def get_pos(self, seq_len):
        # use sine positional embeddinds
        return torch.arange(seq_len).unsqueeze(0)
 
    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.args.n_layers,
            batch_size,
            self.args.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.args.n_layers,
            batch_size,
            self.args.hidden_dim)
        c = c.to(self.device)

        return (h, c)


    def forward(self, input):
        # test, question, tag, _, mask, interaction = input
        # batch_size = interaction.size(0)
        # seq_len = interaction.size(1)

        cate_x, cont_x, mask = input
        batch_size = cate_x.size(0)
        seq_len = cate_x.size(1)
        mask = mask.float()

         # category
        cate_emb = self.cate_embed(cate_x).view(batch_size, seq_len, -1)
        # cate_emb = self.cate_embed(cate_x)
        cate_emb = self.cate_proj(cate_emb)

        # continuous
        cont_x = self.cont_bn(cont_x.view(-1, cont_x.size(-1))).view(batch_size, -1, cont_x.size(-1))
        cont_emb = self.cont_embed(cont_x)   
        
        # combination
        embed = torch.cat([cate_emb, cont_emb], 2)        
        embed = self.comb_proj(embed)   

        # Encoder 
        q = self.query(embed)[:, -1:, :].permute(1, 0, 2)
        k = self.key(embed).permute(1, 0, 2)
        v = self.value(embed).permute(1, 0, 2)

        # attention
        self.mask = self.get_mask(seq_len, mask, batch_size).to(self.device)
        out, _ = self.attn(q, k, v, attn_mask=self.mask)
        
        # residual + layer norm
        out = out.permute(1, 0, 2)
        out = embed + out
        out = self.ln1(out)

        # feed forward network
        out = self.ffn(out)

        # residual + layer norm
        out = embed + out
        out = self.ln2(out)

        # LSTM
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(out, hidden)

        # DNN 
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        return preds