import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import torch.nn.init as init

from Metis.bitlinear import *


        

class MultiheadAttention(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.embed_dim = args.embed_dim
        self.heads_num = args.heads
        self.window_size = args.win_size
        assert args.embed_dim % args.heads == 0, 'Embedding dimension must be divisible by number of heads.'
        print(f'embed_dim: {args.embed_dim}, heads_num: {args.heads}, windowssize: {args.win_size}')

        self.key = BitLinear(args.embed_dim, args.embed_dim, args=args)
        self.query = BitLinear(args.embed_dim, args.embed_dim, args=args)
        self.value = BitLinear(args.embed_dim, args.embed_dim, args=args)
        self.proj = BitLinear(args.embed_dim, args.embed_dim, args=args)
        # self.key = nn.Linear(embed_dim, embed_dim)
        # self.query = nn.Linear(embed_dim, embed_dim)
        # self.value = nn.Linear(embed_dim, embed_dim)
        # self.proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(args.dropout_prob)
        self.proj_dropout = nn.Dropout(args.dropout_prob)
        self.register_buffer('mask',
            torch.tril(torch.ones(1, 1, self.window_size, self.window_size, device=args.device), diagonal=0)
        )

        self.mask_zero = torch.zeros(1, device=args.device)

    def forward(self, x):
        bs = x.size(0)
        seq_len = x.size(1)

        # x = [bs, seq_len, embed_dim]
        k = self.key(x).view(bs, seq_len, self.heads_num, self.embed_dim // self.heads_num).transpose(1, 2)
        q = self.query(x).view(bs, seq_len, self.heads_num, self.embed_dim // self.heads_num).transpose(1, 2)
        v = self.value(x).view(bs, seq_len, self.heads_num, self.embed_dim // self.heads_num).transpose(1, 2)
        # k, q, v = [bs, heads_num, seq_len, embed_dim // heads_num]

        # [b, h, n, d] * [b, h, d, n] = [b, h, n, n]
        attn = (torch.matmul(q, k.transpose(-2, -1))) / math.sqrt(self.embed_dim // self.heads_num)
        mask = self.mask[:, :, :seq_len, :seq_len] #[1, 1, n, n]
        attn = attn.masked_fill(mask == self.mask_zero, float('-inf')) 

        # attn[b, 0, n] = q[b, 0, d] * k[b, d, n] * mask_fill
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # [b, h, n, n] * [b, h, n, d] = [b, h, n, d]     x[b, 0, d] = attn[b, 0, n] * v[b, n, d]
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).contiguous().view(bs, seq_len, self.embed_dim)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.feed_fwd = nn.Sequential(
            BitLinear(args.embed_dim, 4 * args.embed_dim, args=args),
            nn.GELU(),
            BitLinear(4 * args.embed_dim, args.embed_dim, args=args),
            nn.Dropout(args.dropout_prob)
        )

    def forward(self, x):
        return self.feed_fwd(x)


class Decoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(args.embed_dim, device=args.device)
        self.ln2 = nn.LayerNorm(args.embed_dim, device=args.device)
        self.attn = MultiheadAttention(args)
        self.feed_fwd = FeedForward(args)
        
        self.get_attn_output_hook = lambda x, y, z: None
        self.get_ffn_output_hook = lambda x, y, z: None

    def forward(self, x):
        if isinstance(x, tuple):
            x, _ = x
        x = self.get_attn_output(x)
        x = self.get_ffn_output(x)

        return x
    
    def get_attn_output(self, x):
        if isinstance(x, tuple):
            x, _ = x
        attn_out = self.attn(x)
        out = attn_out + x
        self.get_attn_output_hook(attn_out, x, out)
        return out
    
    def get_ffn_output(self, x):
        ffn_out = self.get_ffn_output_wo_ln(x)
        out = ffn_out + x
        self.get_ffn_output_hook(ffn_out, x, out)
        out = self.ln2(out)
        return out
    
    def get_ffn_output_wo_ln(self, x):
        if isinstance(x, tuple):
            x, _ = x
        x = self.feed_fwd(x)
        return x
    
    def ffn_ln(self, x):
        return self.ln2(x)

class GPT(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(
            args.vocab_size, 
            args.embed_dim, 
            padding_idx=args.vocab_size-1, 
            device=args.device
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, args.win_size, args.embed_dim).to(device=args.device))
        self.dropout = nn.Dropout(args.dropout_prob)
        self.decoders = nn.Sequential(*[Decoder(args) for _ in range(args.layers)])
        self.ln = nn.LayerNorm(args.embed_dim, device=args.device)
        self.fc = nn.Linear(args.embed_dim, args.vocab_size, bias=False, device=args.device)
        print(f'vocab_size: {args.vocab_size}, embed_dim: {args.embed_dim}, layers: {args.layers}, pdrop: {args.dropout_prob}')


    def forward(self, x):
        x = self.get_decoder_output(x, len(self.decoders) - 1)
        x = self.decode(x)

        return x

    def get_decoder_output(self, x, i, prev = None):
        if prev is None:
            x = self.embed(x)
            for j in range(i + 1):
                x = self.decoders[j](x)
            return x
        else:
            return self.decoders[i](prev)

    def get_attn_output(self, x, layer):
        x = self.get_decoder_output(x, layer - 1)
        x = self.decoders[layer].get_attn_output(x)
        return x

    def decode(self, x):
        x = self.fc(self.ln(x))
        return x

    def embed(self, x):
        seq_len = x.size(1)
        tok_x = self.tok_emb(x)
        pos_emb = self.pos_emb[:, :seq_len, :]
        # x = self.dropout(tok_x) + pos_emb
        x = self.dropout(tok_x)
        return x

    # def embed(self, x, eps: float = 1e-12):
    #     """
    #     - 常规 embedding 路径得到 x_emb: [B, T, C]
    #     - 展平成二维 X: [B*T, C] 做标准 SVD（full_matrices=False）
    #     - 将所有奇异值设为相同的 s，使得 ||X||_F 保持不变
    #     - 低秩/满秩都不区分，直接用所有奇异值
    #     - 使用 STE：X_out = X + (X_equalized - X).detach() 使梯度不穿过 SVD

    #     返回：
    #     [B, T, C] 与原 embedding 同形状的张量
    #     """
    #     B, T = x.size(0), x.size(1)

    #     # 常规 embedding
    #     tok_x   = self.tok_emb(x)                # [B, T, C]
    #     pos_emb = self.pos_emb[:, :T, :]         # [1, T, C]
    #     x_emb   = self.dropout(tok_x) + pos_emb  # [B, T, C]

    #     BT, C = B * T, x_emb.size(-1)
    #     X = x_emb.reshape(BT, C)                  # [BT, C]

    #     # —— 前向分解在 float32 中做，避免数值问题；并阻断梯度穿过 SVD ——
    #     with torch.no_grad():
    #         X32 = X.to(torch.float32)

    #         # 标准 SVD（使用全部奇异值）
    #         U, S, Vh = torch.linalg.svd(X32, full_matrices=False)  # U:[BT,r], S:[r], Vh:[r,C]
    #         r = S.shape[0]                                         # r = min(BT, C)

    #         # Frobenius 范数保持：||X||_F^2 = sum_i S_i^2
    #         fro = torch.linalg.norm(S, ord=2)  # 等价于 ||X||_F
    #         # 若极端退化，防止除零
    #         s_equal = fro / (r ** 0.5 + eps)

    #         # 等幅重构：X_equalized = s * (U @ Vh)
    #         X_equalized_32 = s_equal * (U @ Vh)
    #         X_equalized = X_equalized_32.to(dtype=X.dtype, device=X.device)

    #     # STE：前向替换为等幅结果；反向梯度视作恒等映射，避免穿过 SVD
    #     X_out = X + (X_equalized - X).detach()

    #     return X_out.reshape(B, T, C)
