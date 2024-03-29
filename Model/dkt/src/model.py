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
        return self.layer2(F.relu(self.layer1(ffn_in)))

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


        # 신나는 embedding

        # embed_interaction = self.embedding_interaction(interaction)
        # embed_test = self.embedding_test(test)
        # embed_question = self.embedding_question(question)
        # embed_tag = self.embedding_tag(tag)

        # embed = torch.cat(
        #     [
        #         embed_interaction,
        #         embed_test,
        #         embed_question,
        #         embed_tag,
        #     ],
        #     2,
        # )

        # embed = self.comb_proj(embed)

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
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)


class Saint(nn.Module):
    def __init__(self, args):
        super(Saint, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        # self.dropout = self.args.dropout
        self.dropout = 0.
        
        ### Embedding 
        # ENCODER embedding
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim//3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim//3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim//3)
        
        # encoder combination projection
        self.enc_comb_proj = nn.Linear((self.hidden_dim//3)*3, self.hidden_dim)

        # DECODER embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//3)
        
        # decoder combination projection
        self.dec_comb_proj = nn.Linear((self.hidden_dim//3)*4, self.hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.hidden_dim, self.dropout, self.args.max_seq_len)
        self.pos_decoder = PositionalEncoding(self.hidden_dim, self.dropout, self.args.max_seq_len)
        

        self.transformer = nn.Transformer(
            d_model=self.hidden_dim, 
            nhead=self.args.n_heads,
            num_encoder_layers=self.args.n_layers, 
            num_decoder_layers=self.args.n_layers, 
            dim_feedforward=self.hidden_dim, 
            dropout=self.dropout, 
            activation='relu')

        self.fc = nn.Linear(self.hidden_dim, 1)
        self.activation = nn.Sigmoid()

        self.enc_mask = None
        self.dec_mask = None
        self.enc_dec_mask = None
    
    def get_mask(self, seq_len):
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1))

        return mask.masked_fill(mask==1, float('-inf'))

    def forward(self, input):
        test, question, tag, _, mask, interaction = input

        batch_size = interaction.size(0)
        seq_len = interaction.size(1)

        # 신나는 embedding
        # Encoder
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed_enc = torch.cat([embed_test,
                               embed_question,
                               embed_tag,], 2)

        embed_enc = self.enc_comb_proj(embed_enc)
        
        # Decoder     
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed_interaction = self.embedding_interaction(interaction)
        embed_dec = torch.cat([embed_test,
                               embed_question,
                               embed_tag,
                               embed_interaction], 2)

        embed_dec = self.dec_comb_proj(embed_dec)

        # ATTENTION MASK
        if self.enc_mask is None or self.enc_mask.size(0) != seq_len:
            self.enc_mask = self.get_mask(seq_len).to(self.device)
            
        if self.dec_mask is None or self.dec_mask.size(0) != seq_len:
            self.dec_mask = self.get_mask(seq_len).to(self.device)
            
        if self.enc_dec_mask is None or self.enc_dec_mask.size(0) != seq_len:
            self.enc_dec_mask = self.get_mask(seq_len).to(self.device)
            
  
        embed_enc = embed_enc.permute(1, 0, 2)
        embed_dec = embed_dec.permute(1, 0, 2)
        
        # Positional encoding
        embed_enc = self.pos_encoder(embed_enc)
        embed_dec = self.pos_decoder(embed_dec)
        
        out = self.transformer(embed_enc, embed_dec,
                               src_mask=self.enc_mask,
                               tgt_mask=self.dec_mask,
                               memory_mask=self.enc_dec_mask)

        out = out.permute(1, 0, 2)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        return preds