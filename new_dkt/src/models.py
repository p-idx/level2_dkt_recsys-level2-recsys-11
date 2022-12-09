import torch
import torch.nn as nn

from .modules import *

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel
except:
    from transformers.models.bert.modeling_bert import (
        BertConfig,
        BertEncoder,
        BertModel,
    )


class CateEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cate_indices = [config.cate2idx[col] for col in config.cate_cols]
        self.cate_offsets = [config.cate_offsets[i] for i in self.cate_indices]
        self.embedding_layer = nn.Embedding(sum(self.cate_offsets) + 1, config.cate_embed_size, padding_idx=0)
        # print(sum(self.cate_offsets))
        self.norm = nn.LayerNorm(config.cate_embed_size * len(config.cate_cols))

    def forward(self, cate_x, mask):
        offset = 0
        for i in range(len(self.cate_offsets) - 1):
            cum_mask = mask.clone() * offset
            cate_x[:, :, self.cate_indices[i]] += cum_mask
            offset += self.cate_offsets[i+1]
        
        cum_mask = mask.clone() * offset
        cate_x[:, :, self.cate_indices[-1]] += cum_mask          

        emb_x = self.embedding_layer(cate_x[:, :, self.cate_indices])
        emb_x = self.norm(emb_x.view(cate_x.size(0), cate_x.size(1), -1))
        return emb_x


class ContEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cate_indices = [config.cont2idx[col] for col in config.cont_cols]
        self.embedding_layer = nn.Linear(len(config.cont_cols), config.cont_embed_size) 
        self.norm = nn.LayerNorm(config.cont_embed_size)

    def forward(self, cont_x):
        emb_x = self.embedding_layer(cont_x[:, :, self.cate_indices])
        emb_x = self.norm(emb_x)
        return emb_x


class LSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.selected_cates = ['assessmentItemID', 'testId', 'KnowledgeTag']
        # self.selected_cate2idx = [config.cate2idx[cate] for cate in self.selected_cates]
        # self.cate_embedding_layer = CateEmbedding(config, self.selected_cates)
        
        
        self.interaction_embedding_layer = nn.Sequential(
            nn.Embedding(3, config.cate_embed_size, padding_idx=0),
        )

        self.assessmentItemID_layer = nn.Sequential(
            nn.Embedding(config.cate_offsets[config.cate2idx['assessmentItemID']] + 1, config.cate_embed_size, padding_idx=0),
            nn.Linear(config.cate_embed_size, config.cate_embed_size),
            nn.LayerNorm(config.cate_embed_size)
        )

        self.knowledgetag_layer = nn.Sequential(
            nn.Embedding(config.cate_offsets[config.cate2idx['KnowledgeTag']] + 1, config.cate_embed_size, padding_idx=0),
            nn.Linear(config.cate_embed_size, config.cate_embed_size),
            nn.LayerNorm(config.cate_embed_size)
        )

        self.testid_layer = nn.Sequential(
            nn.Embedding(config.cate_offsets[config.cate2idx['testId']] + 1, config.cate_embed_size, padding_idx=0),
            nn.Linear(config.cate_embed_size, config.cate_embed_size),
            nn.LayerNorm(config.cate_embed_size)
        )

        self.testid_avg_layer = nn.Sequential(
            nn.Embedding(config.cate_offsets[config.cate2idx['testId_avg_rate']] + 1, config.cate_embed_size, padding_idx=0),
            nn.Linear(config.cate_embed_size, config.cate_embed_size),
            nn.LayerNorm(config.cate_embed_size)
        )

        
        self.assessmentItemID_avg_rate_layer = nn.Sequential(
            nn.Embedding(config.cate_offsets[config.cate2idx['assessmentItemID_avg_rate']] + 1, config.cate_embed_size, padding_idx=0),
            nn.Linear(config.cate_embed_size, config.cate_embed_size),
            nn.LayerNorm(config.cate_embed_size)
        )

        config.used_cols = ['interaction', 'assessmentItemID', 'KnowledgeTag', 'testId', 'testId_avg_rate', 'assessmentItemID_avg_rate']


        self.projection_layer = nn.Sequential(
            nn.Linear(config.cate_embed_size * len(config.used_cols), config.cate_embed_size),
            nn.Dropout(config.drop_out),
            nn.LayerNorm(config.cate_embed_size),
        )


        self.lstm_layer = nn.LSTM(
            input_size=config.cate_embed_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
        )

        self.fc = FinalConnecting(config, config.hidden_size)


    def forward(self, cate_x, cont_x, mask, answers):
        # interaction = (answers.clone().roll(1, dims=-1) + 1).long()
        # interaction[:, 0] = 0
        # interaction_emb = self.interaction_embedding_layer(interaction)
        interaction = answers.clone().roll(1, dims=-1).long() + 1
        interaction[:, 0] = 0

        inter_emb = self.interaction_embedding_layer(interaction)
        # cate_emb = self.cate_embedding_layer(cate_x, mask)
        # project_emb = self.projection_layer(cate_emb)
        assess_emb = self.assessmentItemID_layer(cate_x[:, :, self.config.cate2idx['assessmentItemID']])
        know_emb = self.knowledgetag_layer(cate_x[:, :, self.config.cate2idx['KnowledgeTag']])
        testid_emb = self.testid_layer(cate_x[:, :, self.config.cate2idx['testId']])
        testid_avg_emb = self.testid_avg_layer(cate_x[:, :, self.config.cate2idx['testId_avg_rate']])

        assess_avg_emb = self.assessmentItemID_avg_rate_layer(cate_x[:, :, self.config.cate2idx['assessmentItemID_avg_rate']])

        # x = torch.cat([inter_emb, assess_emb], dim=-1)
        x = torch.cat([inter_emb, assess_emb, know_emb, testid_emb, testid_avg_emb, assess_avg_emb], dim=-1)
        x = self.projection_layer(x)
        hs, _ = self.lstm_layer(x)

        out = self.fc(hs)
        return out.squeeze(-1) 

    

class SAKT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.M_layer = nn.Embedding((config.cate_offsets[0] + 1) * 2 - 1, config.attention_size, padding_idx=0)
        self.E_layer = nn.Embedding((config.cate_offsets[0] + 1), config.attention_size, padding_idx=0)
    
        # self.comb_layer = EntireEmbedding(config)

        self.Poistion_layer = nn.Embedding(config.seq_len - 1, config.attention_size)

        self.attentions = []
        for i in range(config.num_heads):
            self.attentions.append(
                nn.ModuleDict(
                    {
                        f'Q{i}': nn.Linear(config.attention_size, (config.attention_size // 3) * 2, bias=True),
                        f'K{i}': nn.Linear(config.attention_size, (config.attention_size // 3) * 2, bias=True),
                        f'V{i}': nn.Linear(config.attention_size, (config.attention_size // 3) * 2, bias=True),
                    }
                )
            )
        self.attentions = nn.ModuleList(self.attentions)


        self.W0_layer = nn.Linear(((config.attention_size // 3) * 2) * config.num_heads, config.attention_size, bias=False)
        self.res_layer1 = nn.LayerNorm(config.attention_size)

        self.ffnn_layer = nn.Sequential(
            nn.Linear(config.attention_size, config.ffnn_size),
            nn.ReLU(),
            nn.Dropout(config.drop_out),
            nn.Linear(config.ffnn_size, config.attention_size)
        )

        self.res_layer2 = nn.LayerNorm(config.attention_size)

        # final layer
        self.final_layer = FinalConnecting(config, config.attention_size)

    
    def forward(self, cate_x: torch.Tensor, cont_x: torch.Tensor, mask: torch.Tensor, targets):
        new_mask = torch.ones((cate_x.size(0), self.config.seq_len - 1)).to(cate_x.get_device())
        new_mask = (torch.triu(new_mask, diagonal=-1) * 100000) - 100000
        new_mask.requires_grad = False

        assessments = cate_x[:, :-1, 0]
        interactions = targets.clone()[:, :-1]

        y = (assessments + interactions * (self.config.cate_offsets[0])).long()
        next_assessments = cate_x[:, 1:, 0]

        positions = self.Poistion_layer(torch.arange(self.config.seq_len - 1).unsqueeze(0).to(cate_x.get_device()))
        
        M_hat = (self.M_layer(y) + positions).transpose(0, 1)
        E_hat = self.E_layer(next_assessments).transpose(0, 1) # 여기에 문제정보 더 추가해서 콘캣하는게 좋겠다.
        # assessment_infos = self.comb_layer(cate_x, cont_x)
        # E_hat = torch.cat([E_hat, assessment_infos[:, 1:, :]], dim=-1)

        E_for_K = E_hat
        E_for_V = E_hat

        Zs = []
        for i in range(self.config.num_heads):
            Q = self.attentions[i][f'Q{i}'](M_hat)
            K = self.attentions[i][f'K{i}'](E_for_K)
            V = self.attentions[i][f'V{i}'](E_for_V)
            score = torch.matmul(Q, K.transpose(-2, -1))
            score = torch.div(score, Q.size(-1) ** 0.5)
            score = score + new_mask.transpose(0, 1).unsqueeze(-1)
            score = torch.softmax(score, dim=-1)
            Z = torch.matmul(score, V)
            Zs.append(Z)

        Zs = torch.cat(Zs, dim=-1)

        z = self.W0_layer(Zs)

        fz = self.ffnn_layer(z)
        z = self.res_layer2(z + fz)

        out = self.final_layer(z)
        return out.transpose(0, 1).squeeze(-1)


class LSTMATTN(nn.Module):
    def __init__(self, config):
        super(LSTMATTN, self).__init__()
        self.config = config

        self.interaction_embedding_layer = nn.Sequential(
            nn.Embedding(3, config.inter_embed_size, padding_idx=0),
        )

        self.cate_embed_layer = CateEmbedding(config)

        self.cont_embed_layer = ContEmbedding(config)


        # 카테, 콘티 임베딩 합쳐주는 레이어
        self.projection_layer = nn.Sequential(
            nn.Linear(config.inter_embed_size + (config.cate_embed_size * len(config.cate_cols)) + config.cont_embed_size, config.attention_size),
            nn.Dropout(config.drop_out),
            # nn.LayerNorm(config.attention_size),
        )

        self.lstm = nn.LSTM(
            config.attention_size, config.hidden_size, batch_first=True
        )

        self.configs = BertConfig(
            3,  # not used
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_heads,
            intermediate_size=config.attention_size,
            hidden_dropout_prob=config.drop_out,
            attention_probs_dropout_prob=config.drop_out,
        )

        self.attn = BertEncoder(self.configs)

        # Fully connected layer
        self.fc = FinalConnecting(config, config.hidden_size)

        self.activation = nn.Sigmoid()

    def forward(self, cate_x, cont_x, mask, answers):

        interaction = answers.clone().roll(1, dims=-1).long() + 1
        interaction[:, 0] = 0

        inter_emb = self.interaction_embedding_layer(interaction)

        cate_emb = self.cate_embed_layer(cate_x, mask)
        
        cont_emb = self.cont_embed_layer(cont_x)

        x = torch.cat([inter_emb, cate_emb, cont_emb], dim=-1)
        x = self.projection_layer(x)

        out, _ = self.lstm(x)

        time_pad_mask = (mask.unsqueeze(-1) * 10000) - 10000 # B, S, 1 로 변경, 시계열 마스크
        last_query_mask = torch.zeros((mask.size(0), mask.size(1))).to(mask.get_device())
        last_query_mask[-1, :] += 10001
        last_query_mask -= 10000
        last_query_mask = last_query_mask.unsqueeze(-1)
        head_mask = [None] * self.config.num_layers

        encoded_layers = self.attn(out, last_query_mask + time_pad_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output)
        return out.squeeze(-1) 


class LastQuery(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.interaction_embedding_layer = nn.Sequential(
            nn.Embedding(3, config.inter_embed_size, padding_idx=0),
        )

        self.cate_embed_layer = CateEmbedding(config)

        self.cont_embed_layer = ContEmbedding(config)


        # 카테, 콘티 임베딩 합쳐주는 레이어
        self.projection_layer = nn.Sequential(
            nn.Linear(config.inter_embed_size + (config.cate_embed_size * len(config.cate_cols)) + config.cont_embed_size, config.attention_size),
            nn.Dropout(config.drop_out),
            # nn.LayerNorm(config.attention_size),
        )

        # 포지셔널 인코딩
        self.poistion_layer = nn.Embedding(config.seq_len, config.attention_size)

        # 멀티 헤드 셀프 어텐션
        self.attentions = []
        for i in range(config.num_heads):
            self.attentions.append(
                nn.ModuleDict(
                    {
                        f'Q{i}': nn.Sequential(
                            nn.Linear(config.attention_size, (config.attention_size // 3) * 2, bias=True),
                            nn.Dropout(config.drop_out)
                        ),
                        f'K{i}': nn.Sequential(
                            nn.Linear(config.attention_size, (config.attention_size // 3) * 2, bias=True),
                            nn.Dropout(config.drop_out)
                        ),
                        f'V{i}': nn.Sequential(
                            nn.Linear(config.attention_size, (config.attention_size // 3) * 2, bias=True),
                            nn.Dropout(config.drop_out)
                        )
                    }
                )
            )
        self.attentions = nn.ModuleList(self.attentions)

        # 헤드 합쳐주는 레이어
        self.W =  nn.Linear(((config.attention_size // 3) * 2) * config.num_heads, config.attention_size, bias=False)

        self.norm1 = nn.LayerNorm(config.attention_size)

        self.ffn = nn.Sequential(
            nn.Linear(config.attention_size, config.ffn_size),
            nn.ReLU(),
            nn.Linear(config.ffn_size, config.attention_size)
        )

        self.norm2 = nn.LayerNorm(config.attention_size)


        self.lstm_layer = nn.LSTM(
            input_size=config.attention_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True
        )

        self.fc = FinalConnecting(config, config.hidden_size)


    def forward(self, cate_x, cont_x, mask, answers):
        interaction = answers.clone().roll(1, dims=-1).long() + 1
        interaction[:, 0] = 0

        inter_emb = self.interaction_embedding_layer(interaction)

        cate_emb = self.cate_embed_layer(cate_x, mask)
        
        cont_emb = self.cont_embed_layer(cont_x)

        x = torch.cat([inter_emb, cate_emb, cont_emb], dim=-1)
        x = self.projection_layer(x)

        # position = self.position_layer(torch.arange(self.config.seq_len).unsqueeze(0).to(cate_x.get_device()))
        # x = x + position
        # x = x.transpose(0, 1) # S, B, embed_size 로 변경

        time_pad_mask = (mask.unsqueeze(-1) * 10000) - 10000 # B, S, 1 로 변경, 시계열 마스크
        last_query_mask = torch.zeros((mask.size(0), mask.size(1))).to(mask.get_device())
        last_query_mask[-1, :] += 10001
        last_query_mask -= 10000
        last_query_mask = last_query_mask.unsqueeze(-1)
        Zs = []
        for i in range(self.config.num_heads):
            Q = self.attentions[i][f'Q{i}'](x)
            K = self.attentions[i][f'K{i}'](x)
            V = self.attentions[i][f'V{i}'](x)
            score = torch.matmul(Q, K.transpose(-2, -1))
            score = torch.div(score, self.config.attention_size ** 0.5) + last_query_mask + time_pad_mask
            score = torch.softmax(score, dim=-1)
            Z = torch.matmul(score, V)
            Zs.append(Z)

        Zs = torch.cat(Zs, dim=-1)
  
        z = self.W(Zs)

        a = self.norm1(z + x) # 스킵 커넥션

        a = self.ffn(a)

        out = self.norm2(a + z) # 스킵 커넥션
        # out = self.fc(out[:, -1:, :])
        
        hs, _ = self.lstm_layer(out)

        if self.config.leak:
            out = self.fc(hs)
        else:
            out = self.fc(hs[:, -1:, :])
        return out.squeeze(-1) 


# class LastQuery2(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
        
#         self.interaction_embedding_layer = nn.Sequential(
#             nn.Embedding(3, config.inter_embed_size, padding_idx=0),
#         )

#         self.cate_embed_layer = CateEmbedding(config)

#         self.cont_embed_layer = ContEmbedding(config)


#         # 카테, 콘티 임베딩 합쳐주는 레이어
#         self.projection_layer = nn.Sequential(
#             nn.Linear(config.inter_embed_size + (config.cate_embed_size * len(config.cate_cols)) + config.cont_embed_size, config.attention_size),
#             nn.Dropout(config.drop_out),
#             # nn.LayerNorm(config.attention_size),
#         )

#         # 포지셔널 인코딩
#         self.position_layer = nn.Embedding(config.seq_len, config.attention_size)

#         # 멀티 헤드 셀프 어텐션
#         self.attentions = []
#         for i in range(config.num_heads):
#             self.attentions.append(
#                 nn.ModuleDict(
#                     {
#                         f'Q{i}': nn.Sequential(
#                             nn.Linear(config.attention_size, (config.attention_size // 3) * 2, bias=True),
#                             nn.Dropout(config.drop_out)
#                         ),
#                         f'K{i}': nn.Sequential(
#                             nn.Linear(config.attention_size, (config.attention_size // 3) * 2, bias=True),
#                             nn.Dropout(config.drop_out)
#                         ),
#                         f'V{i}': nn.Sequential(
#                             nn.Linear(config.attention_size, (config.attention_size // 3) * 2, bias=True),
#                             nn.Dropout(config.drop_out)
#                         )
#                     }
#                 )
#             )
#         self.attentions = nn.ModuleList(self.attentions)

#         # 헤드 합쳐주는 레이어
#         self.W =  nn.Linear(((config.attention_size // 3) * 2) * config.num_heads, config.attention_size, bias=False)

#         self.norm1 = nn.LayerNorm(config.attention_size)

#         self.ffn = nn.Sequential(
#             nn.Linear(config.attention_size, config.ffn_size),
#             nn.ReLU(),
#             nn.Linear(config.ffn_size, config.attention_size)
#         )

#         self.norm2 = nn.LayerNorm(config.attention_size)


#         self.lstm_layer = nn.LSTM(
#             input_size=config.attention_size,
#             hidden_size=config.hidden_size,
#             num_layers=config.num_layers,
#             batch_first=True
#         )

#         self.fc = FinalConnecting(config, config.attention_size)


#     def forward(self, cate_x, cont_x, mask, answers):
#         interaction = answers.clone().roll(1, dims=-1).long() + 1
#         interaction[:, 0] = 0

#         inter_emb = self.interaction_embedding_layer(interaction)

#         cate_emb = self.cate_embed_layer(cate_x, mask)
        
#         cont_emb = self.cont_embed_layer(cont_x)

#         x = torch.cat([inter_emb, cate_emb, cont_emb], dim=-1)
#         x = self.projection_layer(x)

#         position = self.position_layer(torch.arange(self.config.seq_len).unsqueeze(0).to(cate_x.get_device()))
#         x = x + position
#         # x = x.transpose(0, 1) # S, B, embed_size 로 변경

#         time_pad_mask = (mask.unsqueeze(-1) * 10000) - 10000 # B, S, 1 로 변경, 시계열 마스크
#         last_query_mask = torch.zeros((mask.size(0), mask.size(1))).to(mask.get_device())
#         last_query_mask[-1, :] += 10001
#         last_query_mask -= 10000
#         last_query_mask = last_query_mask.unsqueeze(-1)
#         Zs = []
#         for i in range(self.config.num_heads):
#             Q = self.attentions[i][f'Q{i}'](x)
#             K = self.attentions[i][f'K{i}'](x)
#             V = self.attentions[i][f'V{i}'](x)
#             score = torch.matmul(Q, K.transpose(-2, -1))
#             score = torch.div(score, self.config.attention_size ** 0.5) + last_query_mask + time_pad_mask
#             score = torch.softmax(score, dim=-1)
#             Z = torch.matmul(score, V)
#             Zs.append(Z)

#         Zs = torch.cat(Zs, dim=-1)
  
#         z = self.W(Zs)

#         a = self.norm1(z + x) # 스킵 커넥션

#         a = self.ffn(a)

#         out = self.norm2(a + z) # 스킵 커넥션
#         out = self.fc(out)
        
#         # hs, _ = self.lstm_layer(out)

#         # if self.config.leak:
#         #     out = self.fc(hs)
#         # else:
#         #     out = self.fc(hs[:, -1:, :]) # B, S 로 변경 후 대입
#         return out.squeeze(-1) 