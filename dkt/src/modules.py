import torch
import torch.nn as nn

class CateEmbeddingProjector(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # if args.emb_separate:
        #     self.emb_layers = \
        #         nn.ModuleList(
        #             [nn.Embedding(int(offset), int(np.log2(offset)), padding_idx=0) for offset in args.offsets]
        #         )
        #         # 오프셋들의 개수가 곧 cate_num
        #         # 그냥 리스트로 감싸면 파이토치가 디바이스 인식을 못함.
        #     self.proj_layer = nn.Sequential(
        #         nn.Linear(sum([int(np.log2(offset)) for offset in args.offsets]), args.cate_proj_dim),
        #         nn.LayerNorm(args.cate_proj_dim)
        #     )
        # else:
        self.emb_layer = nn.Embedding(args.offset, args.cate_emb_dim, padding_idx=0)
        self.proj_layer = nn.Sequential(
            nn.Linear(args.cate_emb_dim * args.cate_num, args.cate_proj_dim),
            nn.LayerNorm(args.cate_proj_dim)
        )


    def forward(self, cate_x):
        # if self.args.emb_separate:
        #     embs = [layer(cate_x[:, :, i]) for i, layer in enumerate(self.emb_layers)]
        #     embs_x = torch.cat(embs, dim=-1)
        #     proj_x = self.proj_layer(embs_x)
        # else:
        emb_x = self.emb_layer(cate_x)
        emb_x = emb_x.view(emb_x.size(0), self.args.max_seq_len, -1)
        proj_x = self.proj_layer(emb_x)
        return proj_x


class ContEmbeddingProjector(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.proj_layer = nn.Sequential(
            nn.Linear(args.cont_num, args.cont_proj_dim),
            nn.LayerNorm(args.cont_proj_dim)
        )

    def forward(self, cont_x):
        proj_x = self.proj_layer(cont_x)
        return proj_x


class CombProjector(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        cate_proj_dim = args.cate_proj_dim if args.cate_num else 0
        cont_proj_dim = args.cont_proj_dim if args.cont_num else 0

        self.comb_proj_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(cate_proj_dim + cont_proj_dim, args.hidden_dim),
            nn.LayerNorm(args.hidden_dim),
        )

    def forward(self, cate_proj_x, cont_proj_x):
        if self.args.cate_num and self.args.cont_num:
            comb_x = torch.cat([cate_proj_x, cont_proj_x], dim=2)
        elif self.args.cate_num:
            comb_x = cate_proj_x
        elif self.args.cont_num:
            comb_x = cont_proj_x

        return self.comb_proj_layer(comb_x)


class EntireEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cate_emb_proj_layer = CateEmbeddingProjector(args)
        self.cont_emb_proj_layer = ContEmbeddingProjector(args)
        self.comb_proj_layer = CombProjector(args)

    def forward(self, cate_x, cont_x):
        if self.args.cate_num:
            cate_x = self.cate_emb_proj_layer(cate_x)

        if self.args.cont_num:
            cont_x = self.cont_emb_proj_layer(cont_x)

        return self.comb_proj_layer(cate_x, cont_x)


class FinalConnecting(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()
        self.args = args
        self.final_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim), # 원래는 (args.hidden_dim, args.hidden_dim)
            nn.LayerNorm(input_dim),
            nn.Dropout(args.drop_out),
            nn.ReLU(),
            nn.Linear(input_dim, 1)
        )

    def forward(self, x: torch.Tensor):
        return self.final_layer(x)


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model, n):
        super().__init__() # nn.Module 초기화
        
        # encoding : (seq_len, d_model)
        self.encoding = torch.zeros(seq_len, d_model)
        self.encoding.requires_grad = False
        
        # (seq_len, )
        pos = torch.arange(0, seq_len)
        # (seq_len, 1)         
        pos = pos.float().unsqueeze(dim=1) # int64 -> float32 (없어도 되긴 함)
        
        _2i = torch.arange(0, d_model, step=2).float()
        
        self.encoding[:, ::2] = torch.sin(pos / (n ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (n ** (_2i / d_model)))
        
        
    def forward(self, x):
        # x.shape : (batch, seq_len) or (batch, seq_len, d_model)
        seq_len = x.size()[1] 
        # return : (seq_len, d_model)
        # return matrix will be added to x by broadcasting
        return self.encoding[:seq_len, :]