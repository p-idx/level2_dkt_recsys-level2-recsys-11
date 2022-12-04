import numpy as np

import torch
import torch.nn as nn

from .modules import EntireEmbedding, FinalConnecting

from transformers.models.bert.modeling_bert import (
        BertConfig,
        BertEncoder,
        BertModel,
)


class LSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # embedding layer
        self.embedding_layer = EntireEmbedding(args)

        # lstm 
        self.lstm_layer = \
            nn.LSTM(args.hidden_dim, args.hidden_dim, args.n_layers, batch_first=True)

        # final layer
        self.final_layer = FinalConnecting(args)
        
    def forward(self, cate_x: torch.Tensor, cont_x: torch.Tensor, mask: torch.Tensor, targets):
        # embedding
        comb_proj_x = self.embedding_layer(cate_x, cont_x)
        
        # lstm forward
        hs, _ = self.lstm_layer(comb_proj_x)

        # batch_first = True 이기 때문에.
        hs = hs.contiguous().view(hs.size(0), -1, self.args.hidden_dim)

        # final forward
        out = self.final_layer(hs)
        if out.dim() == 2:
            return out
        return out.squeeze()
    

class GRU(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # entire embedding
        self.embedding_layer = EntireEmbedding(args)

        # gru 
        self.gru_layer = \
            nn.GRU(args.hidden_dim, args.hidden_dim, args.n_layers, batch_first=True)

        # final layer
        self.final_layer = FinalConnecting(args)
        
    def forward(self, cate_x: torch.Tensor, cont_x: torch.Tensor, mask: torch.Tensor, targets):
        # embedding
        comb_proj_x = self.embedding_layer(cate_x, cont_x)

        # gru forward
        hs, _ = self.gru_layer(comb_proj_x)

        # batch_first = True 이기 때문에.
        hs = hs.contiguous().view(hs.size(0), -1, self.args.hidden_dim)

        # final forward
        out = self.final_layer(hs)
        return out.squeeze(-1)


class SelfAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # entire embedding
        self.embedding_layer = EntireEmbedding(args)


    def forward(self, cate_x: torch.Tensor, cont_x: torch.Tensor, mask: torch.Tensor, targets):
        # embedding
        comb_proj_x = self.embedding_layer(cate_x, cont_x)


    



# class GRUEncoder(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args

#         self.gru = \
#             nn.GRU(args.hidden_dim, args.hidden_dim, args.n_layers, bidirectional=True)
#         self.fc = nn.Sequential(
#             nn.Linear(args.hidden_dim * 2, args.hidden_dim),
#             nn.LayerNorm(args.hidden_dim)
#         )
        
#     def forward(self, comb_proj_x_t):
#         outputs, hidden = self.gru(comb_proj_x_t)
#         hidden = self.fc(torch.cat([hidden[0:self.args.n_layers], hidden[self.args.n_layers: self.args.n_layers * 2]], dim=2))
#         return outputs, hidden
    

# class GRUDecoder(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args

#         self.gru = \
#             nn.GRU(args.hidden_dim * 2 + args.hidden_dim, args.hidden_dim, args.n_layers)

#         self.energy = nn.Sequential(
#             nn.Linear(args.hidden_dim * 3, 1),
#             nn.ReLU(),
#             nn.Softmax(dim=0)
#         )
        
#     def forward(self, encoder_states, comb_proj_x_t, hidden):
        
#         h_reshape = hidden.repeat(encoder_states.size(0) // self.args.n_layers, 1, 1)
#         # outputs, hidden = self.gru(comb_proj_x_t, hidden)

#         attention = self.energy(torch.cat([h_reshape, encoder_states], dim=2))
#         # (S, B, 1)
#         attention = attention.permute(1, 2, 0)
#         # (B, 1, S)
#         encoder_states = encoder_states.permute(1, 0, 2)
#         # (B, S, hidden_dim * 2)

#         context_vector = torch.bmm(attention, encoder_states).permute(1, 0, 2)
#         # (B, 1, hidden_dim * 2) --> (1, B, hidden_dim * 2)
#         context_vector = context_vector.repeat(comb_proj_x_t.size(0), 1, 1)
#         gru_input = torch.cat([context_vector, comb_proj_x_t], dim=2)

#         outputs, hidden = self.gru(gru_input, hidden)
        
#         return outputs, hidden
    

# class S2SGRU(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
        
#         self.cate_emb_proj_layer = CateEmbeddingProjector(args)
#         self.cont_emb_proj_layer = ContEmbeddingProjector(args)

#         # cate_x + cont_x projection
#         self.comb_proj_layer = nn.Sequential(
#             nn.ReLU(),
#             nn.Linear(args.cate_proj_dim + args.cont_proj_dim, args.hidden_dim),
#             nn.LayerNorm(args.hidden_dim),
#         )

#         # gru encoder
#         self.encoder = GRUEncoder(args)
#         self.decoder = GRUDecoder(args)

#         # final layer
#         self.final_layer = nn.Sequential(
#             nn.Linear(args.hidden_dim, args.hidden_dim), # 원래는 (args.hidden_dim, args.hidden_dim)
#             nn.LayerNorm(args.hidden_dim),
#             nn.Dropout(args.drop_out),
#             nn.ReLU(),
#             nn.Linear(args.hidden_dim, 1)
#         )
        
#     def forward(self, cate_x: torch.Tensor, cont_x: torch.Tensor, mask: torch.Tensor, targets):
#         cate_proj_x = self.cate_emb_proj_layer(cate_x)
#         cont_proj_x = self.cont_emb_proj_layer(cont_x)

#         # comb forward
#         comb_x = torch.cat([cate_proj_x, cont_proj_x], dim=2)
#         comb_proj_x = self.comb_proj_layer(comb_x)

#         comb_proj_x_t = comb_proj_x.transpose(0, 1) # (S, B, F)

#         batch_size = targets.size(0)
#         target_len = targets.size(1) # (B, S) 임으로

#         encoder_states, hidden = self.encoder(comb_proj_x_t)
#         outputs, _ = self.decoder(encoder_states, comb_proj_x_t, hidden)

#         outputs = outputs.transpose(0, 1)
#         # final forward
#         out = self.final_layer(outputs)
#         return out.squeeze(-1)
    
    
# class GRUBI(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
        
#         self.cate_emb_proj_layer = CateEmbeddingProjector(args)
#         self.cont_emb_proj_layer = ContEmbeddingProjector(args)

#         # cate_x + cont_x projection
#         self.comb_proj_layer = nn.Sequential(
#             nn.ReLU(),
#             nn.Linear(args.cate_proj_dim + args.cont_proj_dim, args.hidden_dim),
#             nn.LayerNorm(args.hidden_dim),
#         )

#         # gru 
#         self.gru_layer = \
#             nn.GRU(args.hidden_dim, args.hidden_dim, args.n_layers, batch_first=True, bidirectional=True)

#         # final layer
#         self.final_layer = nn.Sequential(
#             nn.Linear(args.hidden_dim  * 2, args.hidden_dim), # 원래는 (args.hidden_dim, args.hidden_dim)
#             nn.LayerNorm(args.hidden_dim),
#             nn.Dropout(args.drop_out),
#             nn.ReLU(),
#             nn.Linear(args.hidden_dim, 1)
#         )
        
#     def forward(self, cate_x: torch.Tensor, cont_x: torch.Tensor, mask: torch.Tensor):
#         cate_proj_x = self.cate_emb_proj_layer(cate_x)
#         cont_proj_x = self.cont_emb_proj_layer(cont_x)

#         # comb forward
#         comb_x = torch.cat([cate_proj_x, cont_proj_x], dim=2)
#         comb_proj_x = self.comb_proj_layer(comb_x)
        
#         # gru forward
#         hs, _ = self.gru_layer(comb_proj_x)

#         # batch_first = True 이기 때문에.
#         hs = hs.contiguous().view(hs.size(0), -1, self.args.hidden_dim  * 2)

#         # final forward
#         out = self.final_layer(hs)
#         return out.squeeze(-1)


# class GRUATT(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
        
#         self.cate_emb_proj_layer = CateEmbeddingProjector(args)
#         self.cont_emb_proj_layer = ContEmbeddingProjector(args)

#         # cate_x + cont_x projection
#         self.comb_proj_layer = nn.Sequential(
#             nn.ReLU(),
#             nn.Linear(args.cate_proj_dim + args.cont_proj_dim, args.hidden_dim),
#             nn.LayerNorm(args.hidden_dim),
#         )

#         # gru 
#         self.gru_layer = \
#             nn.GRU(args.hidden_dim, args.hidden_dim, args.n_layers, batch_first=True)
        
#         # attention
#         self.config = BertConfig(
#             3,  # not used
#             hidden_size=self.args.hidden_dim,
#             num_hidden_layers=1,
#             num_attention_heads=self.args.n_heads,
#             intermediate_size=self.args.hidden_dim,
#             hidden_dropout_prob=self.args.drop_out,
#             attention_probs_dropout_prob=self.args.drop_out,
#         )
#         self.attn = BertEncoder(self.config)

#         # final layer
#         self.final_layer = nn.Sequential(
#             nn.Linear(args.hidden_dim, args.hidden_dim), # 원래는 (args.hidden_dim, args.hidden_dim)
#             nn.LayerNorm(args.hidden_dim),
#             nn.Dropout(args.drop_out),
#             nn.ReLU(),
#             nn.Linear(args.hidden_dim, 1)
#         )
        
#     def forward(self, cate_x: torch.Tensor, cont_x: torch.Tensor, mask: torch.Tensor):
#         cate_proj_x = self.cate_emb_proj_layer(cate_x)
#         cont_proj_x = self.cont_emb_proj_layer(cont_x)

#         # comb forward
#         comb_x = torch.cat([cate_proj_x, cont_proj_x], dim=2)
#         comb_proj_x = self.comb_proj_layer(comb_x)
        
#         # gru forward
#         hs, _ = self.gru_layer(comb_proj_x)

#         # batch_first = True 이기 때문에.
#         hs = hs.contiguous().view(hs.size(0), -1, self.args.hidden_dim)

#         extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
#         extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
#         extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
#         head_mask = [None] * self.args.n_layers

#         encoded_layers = self.attn(hs, extended_attention_mask, head_mask=head_mask)
#         sequence_output = encoded_layers[-1]

#         # final forward
#         out = self.final_layer(sequence_output)
#         return out.squeeze(-1)


# class BERT(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
        
#         # cate_x embedding
#         self.cate_emb_layer = nn.Embedding(args.offset, args.cate_emb_dim, padding_idx=0)
#         self.cate_proj_layer = nn.Sequential(
#             nn.Linear(args.cate_emb_dim * args.cate_num, args.cate_proj_dim),
#             nn.LayerNorm(args.cate_proj_dim)
#         )

#         # cont_x embedding
#         self.cont_proj_layer = nn.Sequential(
#             nn.Linear(args.cont_num, args.cont_proj_dim),
#             nn.LayerNorm(args.cont_proj_dim)
#         )

#         # cate_x + cont_x projection
#         self.comb_proj_layer = nn.Sequential(
#             nn.ReLU(),
#             nn.Linear(args.cate_proj_dim + args.cont_proj_dim, args.hidden_dim),
#             nn.LayerNorm(args.hidden_dim),
#         )

#         # bert
#         self.config = BertConfig( 
#             3, # not used
#             hidden_size=self.args.hidden_dim,
#             num_hidden_layers=self.args.n_layers,
#             num_attention_heads=self.args.n_heads,
#             max_position_embeddings=self.args.max_seq_len,
#             intermediate_size=self.args.hidden_dim,
#             hidden_dropout_prob=self.args.drop_out,
#             attention_probs_dropout_prob=self.args.drop_out,
#         )
#         self.encoder = BertModel(self.config)

#         # final layer
#         self.final_layer = nn.Sequential(
#             nn.Linear(args.hidden_dim, args.hidden_dim), # 원래는 (args.hidden_dim, args.hidden_dim)
#             nn.LayerNorm(args.hidden_dim),
#             nn.Dropout(args.drop_out),
#             nn.ReLU(),
#             nn.Linear(args.hidden_dim, 1)
#         )
        
#     def forward(self, cate_x: torch.Tensor, cont_x: torch.Tensor, mask: torch.Tensor):
#         # cate forward
#         cate_emb_x = self.cate_emb_layer(cate_x)
#         cate_emb_x = cate_emb_x.view(cate_emb_x.size(0), self.args.max_seq_len, -1)
#         cate_proj_x = self.cate_proj_layer(cate_emb_x)

#         # cont forawrd
#         cont_proj_x = self.cont_proj_layer(cont_x)

#         # comb forward
#         comb_x = torch.cat([cate_proj_x, cont_proj_x], dim=2)
#         comb_proj_x = self.comb_proj_layer(comb_x)

#         # mask, _ = mask.view(self.args.batch_size, self.args.max_seq_len, -1).max(2)

#         encoded_layers = self.encoder(inputs_embeds=comb_proj_x, attention_mask=mask)
#         hs = encoded_layers[0]
#         # lstm forward

#         # batch_first = True 이기 때문에.
#         hs = hs.contiguous().view(hs.size(0), -1, self.args.hidden_dim)

#         # final forward
#         out = self.final_layer(hs)
#         return out.squeeze(-1)