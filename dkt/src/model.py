import numpy as np

import torch
import torch.nn as nn

from .modules import EntireEmbedding, FinalConnecting, PositionalEncoding

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
            nn.LSTM(args.hidden_dim, args.hidden_dim, args.n_layers, batch_first=False)

        # final layer
        self.final_layer = FinalConnecting(args, args.hidden_dim)
        
    def forward(self, cate_x: torch.Tensor, cont_x: torch.Tensor, mask: torch.Tensor, targets):
        # embedding
        comb_proj_x = self.embedding_layer(cate_x, cont_x)
        comb_proj_x = comb_proj_x.transpose(0, 1)
        
        # lstm forward
        hs, hn = self.lstm_layer(comb_proj_x)

        # batch_first = True 이기 때문에.
        # hs = hs.contiguous().view(hs.size(0), -1, self.args.hidden_dim)
    
        # final forward
        out = self.final_layer(hn[0][-1])
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
        self.final_layer = FinalConnecting(args, args.hidden_dim)
        
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

        self.Q_layer = nn.Linear(args.hidden_dim, args.attention_dim, bias=False)
        self.K_layer = nn.Linear(args.hidden_dim, args.attention_dim, bias=False)
        self.V_layer = nn.Linear(args.hidden_dim, args.attention_dim, bias=False)
        self.softmax_layer = nn.Softmax(dim=1)

        self.position_layer = PositionalEncoding(args.max_seq_len, args.hidden_dim, 10000)

        # final layer
        self.final_layer = FinalConnecting(args, args.attention_dim)


    def forward(self, cate_x: torch.Tensor, cont_x: torch.Tensor, mask: torch.Tensor, targets):
        # embedding
        comb_proj_x = self.embedding_layer(cate_x, cont_x)
        positions = self.position_layer(comb_proj_x).to(comb_proj_x.get_device())

        comb_proj_x = comb_proj_x + positions
        # B, S, F 상태
        Q = self.Q_layer(comb_proj_x)
        K = self.K_layer(comb_proj_x)
        V = self.V_layer(comb_proj_x)

        scores = torch.bmm(Q, K.transpose(1, 2))
        scores = torch.div(scores, K.size(2) ** 0.5)
        scores = self.softmax_layer(scores)
        z = torch.bmm(scores, V)

        out = self.final_layer(z)
        return out.squeeze(-1)



class SelfAttention2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # entire embedding
        self.embedding_layer = EntireEmbedding(args)

        self.Q_layer = nn.Linear(args.hidden_dim, args.attention_dim, bias=False)
        self.K_layer = nn.Linear(args.hidden_dim, args.attention_dim, bias=False)
        self.V_layer = nn.Linear(args.hidden_dim, args.attention_dim, bias=False)
        self.softmax_layer = nn.Softmax(dim=1)

        # gru 
        self.gru_layer = \
            nn.GRU(args.attention_dim, args.attention_dim, args.n_layers, batch_first=True)

        # final layer
        self.final_layer = FinalConnecting(args, args.attention_dim)


    def forward(self, cate_x: torch.Tensor, cont_x: torch.Tensor, mask: torch.Tensor, targets):
        # embedding
        comb_proj_x = self.embedding_layer(cate_x, cont_x)

        comb_proj_x = comb_proj_x
        # B, S, F 상태
        Q = self.Q_layer(comb_proj_x)
        K = self.K_layer(comb_proj_x)
        V = self.V_layer(comb_proj_x)

        scores = torch.bmm(Q, K.transpose(1, 2))
        scores = torch.div(scores, K.size(2) ** 0.5)
        scores = self.softmax_layer(scores)
        z = torch.bmm(scores, V)

        hs, hn = self.gru_layer(z)
        hs = hs.contiguous().view(hs.size(0), -1, self.args.attention_dim)

        out = self.final_layer(hs)
        return out.squeeze(-1)


class SelfAttention3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # entire embedding
        self.embedding_layer = EntireEmbedding(args)

        self.Q_layer = nn.Linear(args.hidden_dim, args.attention_dim, bias=False)
        self.K_layer = nn.Linear(args.hidden_dim, args.attention_dim, bias=False)
        self.V_layer = nn.Linear(args.hidden_dim, args.attention_dim, bias=False)
        self.softmax_layer = nn.Softmax(dim=1)

        self.position_layer = PositionalEncoding(args.max_seq_len, args.hidden_dim, 10000)

        # gru 
        self.gru_layer = \
            nn.GRU(args.attention_dim, args.attention_dim, args.n_layers, batch_first=True)

        # final layer
        self.final_layer = FinalConnecting(args, args.attention_dim)


    def forward(self, cate_x: torch.Tensor, cont_x: torch.Tensor, mask: torch.Tensor, targets):
        # embedding
        comb_proj_x = self.embedding_layer(cate_x, cont_x)
        positions = self.position_layer(comb_proj_x).to(comb_proj_x.get_device())

        comb_proj_x = comb_proj_x + positions

        comb_proj_x = comb_proj_x
        # B, S, F 상태
        Q = self.Q_layer(comb_proj_x)
        K = self.K_layer(comb_proj_x)
        V = self.V_layer(comb_proj_x)

        scores = torch.bmm(Q, K.transpose(1, 2))
        scores = torch.div(scores, K.size(2) ** 0.5)
        scores = self.softmax_layer(scores)
        z = torch.bmm(scores, V)

        hs, hn = self.gru_layer(z)
        hs = hs.contiguous().view(hs.size(0), -1, self.args.attention_dim)

        out = self.final_layer(hs)
        return out.squeeze(-1)


class SelfAttention4(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # entire embedding
        self.embedding_layer = EntireEmbedding(args)

        # head 1
        self.Q1_layer = nn.Linear(args.hidden_dim, args.attention_dim, bias=False)
        self.K1_layer = nn.Linear(args.hidden_dim, args.attention_dim, bias=False)
        self.V1_layer = nn.Linear(args.hidden_dim, args.attention_dim, bias=False)
        self.softmax_layer1 = nn.Softmax(dim=1)

        # head 2
        self.Q2_layer = nn.Linear(args.hidden_dim, args.attention_dim, bias=False)
        self.K2_layer = nn.Linear(args.hidden_dim, args.attention_dim, bias=False)
        self.V2_layer = nn.Linear(args.hidden_dim, args.attention_dim, bias=False)
        self.softmax_layer2 = nn.Softmax(dim=1)

        self.W0_layer = nn.Linear(args.attention_dim * 2, args.hidden_dim, bias=False)


        self.position_layer = PositionalEncoding(args.max_seq_len, args.hidden_dim, 10000)

        # final layer
        self.final_layer = FinalConnecting(args, args.hidden_dim)


    def forward(self, cate_x: torch.Tensor, cont_x: torch.Tensor, mask: torch.Tensor, targets):
        # embedding
        comb_proj_x = self.embedding_layer(cate_x, cont_x)
        positions = self.position_layer(comb_proj_x).to(comb_proj_x.get_device())

        comb_proj_x = comb_proj_x + positions
        # B, S, F 상태
        Q1 = self.Q1_layer(comb_proj_x)
        K1 = self.K1_layer(comb_proj_x)
        V1 = self.V1_layer(comb_proj_x)

        scores1 = torch.bmm(Q1, K1.transpose(1, 2))
        scores1 = torch.div(scores1, K1.size(2) ** 0.5)
        scores1 = self.softmax_layer1(scores1)
        z1 = torch.bmm(scores1, V1)

        Q2 = self.Q2_layer(comb_proj_x)
        K2 = self.K2_layer(comb_proj_x)
        V2 = self.V2_layer(comb_proj_x)

        scores2 = torch.bmm(Q2, K2.transpose(1, 2))
        scores2 = torch.div(scores2, K2.size(2) ** 0.5)
        scores2 = self.softmax_layer2(scores2)
        z2 = torch.bmm(scores2, V2)

        zs = torch.cat([z1, z2], dim=-1)

        z = self.W0_layer(zs)

        out = self.final_layer(z)
        return out.squeeze(-1)


class SelfAttention5(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # entire embedding
        self.embedding_layer = EntireEmbedding(args)

        # head 1
        self.Q1_layer = nn.Linear(args.hidden_dim, args.attention_dim, bias=False)
        self.K1_layer = nn.Linear(args.hidden_dim, args.attention_dim, bias=False)
        self.V1_layer = nn.Linear(args.hidden_dim, args.attention_dim, bias=False)
        self.softmax_layer1 = nn.Softmax(dim=1)

        # head 2
        self.Q2_layer = nn.Linear(args.hidden_dim, args.attention_dim, bias=False)
        self.K2_layer = nn.Linear(args.hidden_dim, args.attention_dim, bias=False)
        self.V2_layer = nn.Linear(args.hidden_dim, args.attention_dim, bias=False)
        self.softmax_layer2 = nn.Softmax(dim=1)

        self.W0_layer = nn.Linear(args.attention_dim * 2, args.hidden_dim, bias=False)
        self.Res_layer = nn.LayerNorm(args.hidden_dim)


        self.position_layer = PositionalEncoding(args.max_seq_len, args.hidden_dim, 10000)

        # final layer
        self.final_layer = FinalConnecting(args, args.hidden_dim)


    def forward(self, cate_x: torch.Tensor, cont_x: torch.Tensor, mask: torch.Tensor, targets):
        # embedding
        comb_proj_x = self.embedding_layer(cate_x, cont_x)
        positions = self.position_layer(comb_proj_x).to(comb_proj_x.get_device())

        comb_proj_x = comb_proj_x + positions
        # B, S, F 상태
        Q1 = self.Q1_layer(comb_proj_x)
        K1 = self.K1_layer(comb_proj_x)
        V1 = self.V1_layer(comb_proj_x)

        scores1 = torch.bmm(Q1, K1.transpose(1, 2))
        scores1 = torch.div(scores1, K1.size(2) ** 0.5)
        scores1 = self.softmax_layer1(scores1)
        z1 = torch.bmm(scores1, V1)

        Q2 = self.Q2_layer(comb_proj_x)
        K2 = self.K2_layer(comb_proj_x)
        V2 = self.V2_layer(comb_proj_x)

        scores2 = torch.bmm(Q2, K2.transpose(1, 2))
        scores2 = torch.div(scores2, K2.size(2) ** 0.5)
        scores2 = self.softmax_layer2(scores2)
        z2 = torch.bmm(scores2, V2)

        zs = torch.cat([z1, z2], dim=-1)

        z = self.W0_layer(zs)
        z = self.Res_layer(comb_proj_x + z)

        out = self.final_layer(z)
        return out.squeeze(-1)

class SelfAttention6(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # entire embedding
        self.embedding_layer = EntireEmbedding(args)

        # head 1
        self.Q1_layer = nn.Linear(args.hidden_dim, args.attention_dim, bias=False)
        self.K1_layer = nn.Linear(args.hidden_dim, args.attention_dim, bias=False)
        self.V1_layer = nn.Linear(args.hidden_dim, args.attention_dim, bias=False)
        self.softmax_layer1 = nn.Softmax(dim=-1)

        # head 2
        self.Q2_layer = nn.Linear(args.hidden_dim, args.attention_dim, bias=False)
        self.K2_layer = nn.Linear(args.hidden_dim, args.attention_dim, bias=False)
        self.V2_layer = nn.Linear(args.hidden_dim, args.attention_dim, bias=False)
        self.softmax_layer2 = nn.Softmax(dim=-1)

        self.W0_layer = nn.Linear(args.attention_dim * 2, args.hidden_dim, bias=False)
        self.res_layer1 = nn.LayerNorm(args.hidden_dim)

        self.ffnn_layer = nn.Sequential(
            nn.Linear(args.hidden_dim, args.ffnn_dim),
            nn.ReLU(),
            # nn.Dropout(args.drop_out),
            nn.Linear(args.ffnn_dim, args.hidden_dim)
        )

        self.res_layer2 = nn.LayerNorm(args.hidden_dim)

        self.position_layer = PositionalEncoding(args.max_seq_len, args.hidden_dim, 10000)

        # final layer
        self.final_layer = FinalConnecting(args, args.hidden_dim)


    def forward(self, cate_x: torch.Tensor, cont_x: torch.Tensor, mask: torch.Tensor, targets):
        # embedding
        comb_proj_x = self.embedding_layer(cate_x, cont_x)
        positions = self.position_layer(comb_proj_x).to(comb_proj_x.get_device())

        mask2 = (mask * 1_000_000) - 1_000_000

        comb_proj_x = comb_proj_x + positions
        # B, S, F 상태
        Q1 = self.Q1_layer(comb_proj_x)
        K1 = self.K1_layer(comb_proj_x)
        V1 = self.V1_layer(comb_proj_x)

        scores1 = torch.bmm(Q1, K1.transpose(1, 2))
        scores1 = torch.div(scores1, K1.size(2) ** 0.5)
        scores1 += mask2.unsqueeze(-1).view(mask2.size(0), 1, -1)
        scores1 = self.softmax_layer1(scores1)
        z1 = torch.bmm(scores1, V1)

        Q2 = self.Q2_layer(comb_proj_x)
        K2 = self.K2_layer(comb_proj_x)
        V2 = self.V2_layer(comb_proj_x)

        scores2 = torch.bmm(Q2, K2.transpose(1, 2))
        scores2 = torch.div(scores2, K2.size(2) ** 0.5)
        scores2 += mask2.unsqueeze(-1).view(mask2.size(0), 1, -1)
        scores2 = self.softmax_layer2(scores2)
        z2 = torch.bmm(scores2, V2)

        zs = torch.cat([z1, z2], dim=-1)

        z = self.W0_layer(zs)
        z = self.res_layer1(comb_proj_x + z)

        fz = self.ffnn_layer(z)
        z3 = self.res_layer2(z + fz)

        out = self.final_layer(z3)
        return out.squeeze(-1)


class SAKT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_assessments = args.offsets[0]
        self.M_layer = nn.Embedding((self.n_assessments + 1) * 2 - 1, args.attention_dim, padding_idx=0)
        self.E_layer = nn.Embedding((self.n_assessments + 1), args.attention_dim, padding_idx=0)
    
        self.comb_layer = EntireEmbedding(args)

        self.Poistion_layer = nn.Embedding(args.max_seq_len - 1, args.attention_dim)

        self.attentions = []
        for i in range(args.n_heads):
            self.attentions.append(
                nn.ModuleDict(
                    {
                        f'Q{i}': nn.Linear(args.attention_dim, (args.attention_dim // 3) * 2, bias=True),
                        f'K{i}': nn.Linear(args.attention_dim + args.hidden_dim, (args.attention_dim // 3) * 2, bias=True),
                        f'V{i}': nn.Linear(args.attention_dim + args.hidden_dim, (args.attention_dim // 3) * 2, bias=True),
                    }
                )
            )
        self.attentions = nn.ModuleList(self.attentions)


        self.W0_layer = nn.Linear(((args.attention_dim // 3) * 2) * args.n_heads, args.attention_dim, bias=False)
        self.res_layer1 = nn.LayerNorm(args.attention_dim)

        self.ffnn_layer = nn.Sequential(
            nn.Linear(args.attention_dim, args.ffnn_dim),
            nn.ReLU(),
            nn.Dropout(args.drop_out),
            nn.Linear(args.ffnn_dim, args.attention_dim)
        )

        self.res_layer2 = nn.LayerNorm(args.attention_dim)

        # final layer
        self.final_layer = FinalConnecting(args, args.attention_dim)


    
    def forward(self, cate_x: torch.Tensor, cont_x: torch.Tensor, mask: torch.Tensor, targets):
        new_mask = torch.ones((cate_x.size(0), self.args.max_seq_len - 1)).to(cate_x.get_device())
        new_mask = (torch.triu(new_mask, diagonal=-1) * 100000) - 100000
        new_mask.requires_grad = False

        assessments = cate_x[:, :-1, 0]
        interactions = targets.clone()[:, :-1]

        y = (assessments + interactions * (self.n_assessments)).long()
        next_assessments = cate_x[:, 1:, 0]

        positions = self.Poistion_layer(torch.arange(self.args.max_seq_len - 1).unsqueeze(0).to(cate_x.get_device()))
        
        M_hat = self.M_layer(y) + positions
        E_hat = self.E_layer(next_assessments) # 여기에 문제정보 더 추가해서 콘캣하는게 좋겠다.
        assessment_infos = self.comb_layer(cate_x, cont_x)
        E_hat = torch.cat([E_hat, assessment_infos[:, 1:, :]], dim=-1)

        E_for_K = E_hat
        E_for_V = E_hat

        Zs = []
        for i in range(self.args.n_heads):
            Q = self.attentions[i][f'Q{i}'](M_hat)
            K = self.attentions[i][f'K{i}'](E_for_K)
            V = self.attentions[i][f'V{i}'](E_for_V)
            score = torch.bmm(Q, K.transpose(-2, -1))
            score = torch.div(score, Q.size(-1) ** 0.5)
            score = score + new_mask.unsqueeze(-1).view(new_mask.size(0), 1, -1)
            score = torch.softmax(score, dim=-1)
            Z = torch.bmm(score, V)
            Zs.append(Z)

        Zs = torch.cat(Zs, dim=-1)

        z = self.W0_layer(Zs)

        fz = self.ffnn_layer(z)
        z = self.res_layer2(z + fz)

        out = self.final_layer(z)


        # Q1 = self.Q1_layer(E_hat)
        # K1 = self.K1_layer(M_hat)
        # V1 = self.V1_layer(M_hat)

        # scores1 = torch.bmm(Q1, K1.transpose(1, 2))
        # scores1 = torch.div(scores1, Q1.size(-1) ** 0.5)
        # scores1 = scores1 + new_mask.unsqueeze(-1).view(new_mask.size(0), 1, -1)
        # scores1 = self.softmax_layer1(scores1)
        # z1 = torch.bmm(scores1, V1)

        # Q2 = self.Q2_layer(E_hat)
        # K2 = self.K2_layer(M_hat)
        # V2 = self.V2_layer(M_hat)

        # scores2 = torch.bmm(Q2, K2.transpose(1, 2))
        # scores2 = torch.div(scores2, Q2.size(-1) ** 0.5)
        # scores2 = scores2 + new_mask.unsqueeze(-1).view(new_mask.size(0), 1, -1)
        # scores2 = self.softmax_layer2(scores2)
        # z2 = torch.bmm(scores2, V2)
        
        # zs = torch.cat([z1, z2], dim=-1)

        # z = self.W0_layer(zs)

        # fz = self.ffnn_layer(z)
        # z3 = self.res_layer2(z + fz)

        # out = self.final_layer(z3)
        return out.squeeze(-1)


class SAKT2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_assessments = args.offsets[0]
        self.M_layer = nn.Embedding((self.n_assessments + 1) * 2 - 1, args.attention_dim, padding_idx=0)
        self.E_layer = nn.Embedding((self.n_assessments + 1), args.attention_dim, padding_idx=0)

        # self.Poistion_layer = nn.Embedding(args.max_seq_len - 1, args.attention_dim)
        
        # self.new_mask = \
        #     (torch.triu(torch.ones((args.batch_size, args.max_seq_len - 1)).to(args.device), diagonal=-1) * 100000) - 100000
        # self.new_mask.requires_grad = False

        # head 1
        self.Q1_layer = nn.Linear(args.attention_dim, args.attention_dim // 2, bias=False)
        self.K1_layer = nn.Linear(args.attention_dim, args.attention_dim // 2, bias=False)
        self.V1_layer = nn.Linear(args.attention_dim, args.attention_dim // 2, bias=False)
        self.softmax_layer1 = nn.Softmax(dim=-1)

        # head 2
        self.Q2_layer = nn.Linear(args.attention_dim, args.attention_dim // 2, bias=False)
        self.K2_layer = nn.Linear(args.attention_dim, args.attention_dim // 2, bias=False)
        self.V2_layer = nn.Linear(args.attention_dim, args.attention_dim // 2, bias=False)
        self.softmax_layer2 = nn.Softmax(dim=-1)

        self.W0_layer = nn.Linear(args.attention_dim, args.attention_dim, bias=False)
        self.res_layer1 = nn.LayerNorm(args.attention_dim)

        self.ffnn_layer = nn.Sequential(
            nn.Linear(args.attention_dim, args.ffnn_dim),
            nn.ReLU(),
            # nn.Dropout(args.drop_out),
            nn.Linear(args.ffnn_dim, args.attention_dim)
        )

        self.res_layer2 = nn.LayerNorm(args.attention_dim)

        # gru 
        self.gru_layer = \
            nn.GRU(args.attention_dim, args.attention_dim, args.n_layers, batch_first=True)

        # final layer
        self.final_layer = FinalConnecting(args, args.attention_dim)


    
    def forward(self, cate_x: torch.Tensor, cont_x: torch.Tensor, mask: torch.Tensor, targets):
        new_mask = torch.ones((cate_x.size(0), self.args.max_seq_len - 1)).to(cate_x.get_device())
        new_mask = (torch.triu(new_mask, diagonal=-1) * 100000) - 100000
        new_mask.requires_grad = False

        assessments = cate_x[:, :-1, 0]
        interactions = targets.clone()[:, :-1]

        y = (assessments + interactions * (self.n_assessments)).long()
        next_assessments = cate_x[:, 1:, 0]

        # positions = self.Poistion_layer(torch.arange(self.args.max_seq_len - 1).unsqueeze(0).to(cate_x.get_device()))
        
        M_hat = self.M_layer(y)
        E_hat = self.E_layer(next_assessments)

        Q1 = self.Q1_layer(E_hat)
        K1 = self.K1_layer(M_hat)
        V1 = self.V1_layer(M_hat)

        scores1 = torch.bmm(Q1, K1.transpose(1, 2))
        scores1 = torch.div(scores1, Q1.size(-1) ** 0.5)
        scores1 = scores1 + new_mask.unsqueeze(-1).view(new_mask.size(0), 1, -1)
        scores1 = self.softmax_layer1(scores1)
        z1 = torch.bmm(scores1, V1)

        Q2 = self.Q2_layer(E_hat)
        K2 = self.K2_layer(M_hat)
        V2 = self.V2_layer(M_hat)

        scores2 = torch.bmm(Q2, K2.transpose(1, 2))
        scores2 = torch.div(scores2, Q2.size(-1) ** 0.5)
        scores2 = scores2 + new_mask.unsqueeze(-1).view(new_mask.size(0), 1, -1)
        scores2 = self.softmax_layer2(scores2)
        z2 = torch.bmm(scores2, V2)
        
        zs = torch.cat([z1, z2], dim=-1)

        z = self.W0_layer(zs)

        fz = self.ffnn_layer(z)
        z3 = self.res_layer2(z + fz)

        hs, hn = self.gru_layer(z3)
        # batch_first = True 이기 때문에.
        hs = hs.contiguous().view(hs.size(0), -1, self.args.attention_dim)

        out = self.final_layer(hs)
        return out.squeeze(-1)