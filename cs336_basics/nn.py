import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        factory_kwargs = {
            'device': device,
            'dtype': dtype
        }
        #定义权重参数 W
        self.weight = nn.Parameter(torch.empty((out_features,in_features), **factory_kwargs))

        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum('...i, oi -> ...o', x, self.weight)

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.weight = nn.Parameter(torch.empty((num_embeddings,embedding_dim),**factory_kwargs))        
        std = 1.0
        nn.init.trunc_normal_(self.weight,mean=0.0,std=1.0,a=-3.0,b=3.0)

    def forward(self,token_ids:torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))
        self.eps = eps

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_float = x.to(torch.float32)

        #计算均方根
        ms = x_float.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(ms + self.eps)

        result = (x_float / rms) * self.weight
        return result.to(in_dtype)

def silu_fn(in_features):
    return in_features * torch.sigmoid(in_features)

class SwiGLU(nn.Module):
    def __init__(self, d_model:int, d_ff:int, device: None, dtype:None):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        #w1和w3进行升维操作
        self.w1 = Linear(d_model,d_ff,device,dtype)
        self.w3 = Linear(d_model,d_ff,device,dtype)
        #w2进行降维操作
        self.w2 = Linear(d_ff,d_model,device,dtype)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        gate = silu_fn(self.w1(x))
        signal = self.w3(x)
        return self.w2(gate * signal)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k

        #构建频率 omega_k = theta^((2k-2)/d)
        powers = torch.arange(0,d_k,2,device=device).float() / d_k
        freqs = 1.0 / (theta ** powers)

        #创建位置序列
        t = torch.arange(max_seq_len,device=device).float()

        #做外积计算所有角度
        freqs_matrix = torch.outer(t,freqs)

        # 预计算 cos 和 sin 并作为 buffer 注册
        self.register_buffer("cos_cached", freqs_matrix.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs_matrix.sin(), persistent=False)

    def forward(self, x:torch.Tensor, token_positions:torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        #维度对齐
        if x.ndim > cos.ndim and cos.ndim >= 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        # 拆分并旋转
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        output = torch.empty_like(x)
        output[..., 0::2] = x_even * cos - x_odd * sin
        output[..., 1::2] = x_even * sin + x_odd * cos

        return output

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - x_max

    #计算指数
    exp_x = torch.exp(x_stable)
    #计算分母
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)

    return exp_x / sum_exp

def scaled_dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None
)-> torch.Tensor:

    d_k = Q.size(-1)
    #计算分数
    scores = torch.einsum('...nk, ...mk-> ...nm', Q, K) / math.sqrt(d_k)
    #进行掩码
    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))

    probs = softmax(scores, dim=-1)
    output = torch.einsum('...nm,...mk->...nk',probs,V)

    return output





