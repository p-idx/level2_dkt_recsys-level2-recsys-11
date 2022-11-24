import numpy as np

import torch
import torch.nn as nn

from transformers.models.bert.modeling_bert import (
        BertConfig,
        BertEncoder,
        BertModel,
)

class LSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # cate_x embedding
        self.cate_emb_layer = nn.Embedding(args.offset, args.cate_emb_dim, padding_idx=0)
        self.cate_proj_layer = nn.Sequential(
            nn.Linear(args.cate_emb_dim * args.cate_num, args.cate_proj_dim),
            nn.LayerNorm(args.cate_proj_dim)
        )

        # cont_x embedding
        self.cont_proj_layer = nn.Sequential(
            nn.Linear(args.cont_num, args.cont_proj_dim),
            nn.LayerNorm(args.cont_proj_dim)
        )

        # cate_x + cont_x projection
        self.comb_proj_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(args.cate_proj_dim + args.cont_proj_dim, args.hidden_dim),
            nn.LayerNorm(args.hidden_dim),
        )

        # lstm 
        self.lstm_layer = \
            nn.LSTM(args.hidden_dim, args.hidden_dim, args.n_layers, batch_first=True)

        # final layer
        self.final_layer = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim), # 원래는 (args.hidden_dim, args.hidden_dim)
            nn.LayerNorm(args.hidden_dim),
            nn.Dropout(args.drop_out),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        )
        
    def forward(self, cate_x: torch.Tensor, cont_x: torch.Tensor, mask: torch.Tensor):
        # cate forward
        cate_emb_x = self.cate_emb_layer(cate_x)
        cate_emb_x = cate_emb_x.view(cate_emb_x.size(0), self.args.max_seq_len, -1)
        cate_proj_x = self.cate_proj_layer(cate_emb_x)

        # cont forawrd
        cont_proj_x = self.cont_proj_layer(cont_x)

        # comb forward
        comb_x = torch.cat([cate_proj_x, cont_proj_x], dim=2)
        comb_proj_x = self.comb_proj_layer(comb_x)
        
        # lstm forward
        hs, _ = self.lstm_layer(comb_proj_x)

        # batch_first = True 이기 때문에.
        hs = hs.contiguous().view(hs.size(0), -1, self.args.hidden_dim)

        # final forward
        out = self.final_layer(hs)
        return out.squeeze()
    


class GRU(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # cate_x embedding
        self.cate_emb_layer = nn.Embedding(args.offset, args.cate_emb_dim, padding_idx=0)
        self.cate_proj_layer = nn.Sequential(
            nn.Linear(args.cate_emb_dim * args.cate_num, args.cate_proj_dim),
            nn.LayerNorm(args.cate_proj_dim)
        )

        # cont_x embedding
        self.cont_proj_layer = nn.Sequential(
            nn.Linear(args.cont_num, args.cont_proj_dim),
            nn.LayerNorm(args.cont_proj_dim)
        )

        # cate_x + cont_x projection
        self.comb_proj_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(args.cate_proj_dim + args.cont_proj_dim, args.hidden_dim),
            nn.LayerNorm(args.hidden_dim),
        )

        # gru 
        self.gru_layer = \
            nn.GRU(args.hidden_dim, args.hidden_dim, args.n_layers, batch_first=True)

        # final layer
        self.final_layer = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim), # 원래는 (args.hidden_dim, args.hidden_dim)
            nn.LayerNorm(args.hidden_dim),
            nn.Dropout(args.drop_out),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        )
        
    def forward(self, cate_x: torch.Tensor, cont_x: torch.Tensor, mask: torch.Tensor):
        # cate forward
        cate_emb_x = self.cate_emb_layer(cate_x)
        cate_emb_x = cate_emb_x.view(cate_emb_x.size(0), self.args.max_seq_len, -1)
        cate_proj_x = self.cate_proj_layer(cate_emb_x)

        # cont forawrd
        cont_proj_x = self.cont_proj_layer(cont_x)

        # comb forward
        comb_x = torch.cat([cate_proj_x, cont_proj_x], dim=2)
        comb_proj_x = self.comb_proj_layer(comb_x)
        
        # lstm forward
        hs, _ = self.gru_layer(comb_proj_x)

        # batch_first = True 이기 때문에.
        hs = hs.contiguous().view(hs.size(0), -1, self.args.hidden_dim)

        # final forward
        out = self.final_layer(hs)
        return out.squeeze()


class BERT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # cate_x embedding
        self.cate_emb_layer = nn.Embedding(args.offset, args.cate_emb_dim, padding_idx=0)
        self.cate_proj_layer = nn.Sequential(
            nn.Linear(args.cate_emb_dim * args.cate_num, args.cate_proj_dim),
            nn.LayerNorm(args.cate_proj_dim)
        )

        # cont_x embedding
        self.cont_proj_layer = nn.Sequential(
            nn.Linear(args.cont_num, args.cont_proj_dim),
            nn.LayerNorm(args.cont_proj_dim)
        )

        # cate_x + cont_x projection
        self.comb_proj_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(args.cate_proj_dim + args.cont_proj_dim, args.hidden_dim),
            nn.LayerNorm(args.hidden_dim),
        )

        # gru 
        self.config = BertConfig( 
            3, # not used
            hidden_size=self.args.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len,
            intermediate_size=self.args.hidden_dim,
            hidden_dropout_prob=self.args.drop_out,
            attention_probs_dropout_prob=self.args.drop_out,
        )
        self.encoder = BertModel(self.config)

        # final layer
        self.final_layer = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim), # 원래는 (args.hidden_dim, args.hidden_dim)
            nn.LayerNorm(args.hidden_dim),
            nn.Dropout(args.drop_out),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        )
        
    def forward(self, cate_x: torch.Tensor, cont_x: torch.Tensor, mask: torch.Tensor):
        # cate forward
        cate_emb_x = self.cate_emb_layer(cate_x)
        cate_emb_x = cate_emb_x.view(cate_emb_x.size(0), self.args.max_seq_len, -1)
        cate_proj_x = self.cate_proj_layer(cate_emb_x)

        # cont forawrd
        cont_proj_x = self.cont_proj_layer(cont_x)

        # comb forward
        comb_x = torch.cat([cate_proj_x, cont_proj_x], dim=2)
        comb_proj_x = self.comb_proj_layer(comb_x)

        # mask, _ = mask.view(self.args.batch_size, self.args.max_seq_len, -1).max(2)

        encoded_layers = self.encoder(inputs_embeds=comb_proj_x, attention_mask=mask)
        hs = encoded_layers[0]
        # lstm forward

        # batch_first = True 이기 때문에.
        hs = hs.contiguous().view(hs.size(0), -1, self.args.hidden_dim)

        # final forward
        out = self.final_layer(hs)
        return out.squeeze()