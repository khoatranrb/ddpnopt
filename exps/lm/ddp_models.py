import numpy as np
import math
from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
import copy

def _layer_type(layer: nn.Module) -> str:
    return layer.__class__.__name__

class Step(nn.Module):
    def __init__(self, module, alpha=0.99, eps=1e-8):
        super(Step, self).__init__()
        self.x = None
        self.x_old = None
        self.mod = module
        self.type = _layer_type(module)
        self.x_hess = 0
        self.alpha = alpha
        self.eps = eps
    def __str__(self):
        return 'DDP'
    def forward(self, inp=None, mask=None, update=False, opt=None):
        if not update:
            self.x_old = inp
            self.x = nn.Parameter(inp)
            return (self.mod(self.x, mask)+self.mod(inp, mask))/2
        else:
            return self.update(inp, mask, opt)
            # try: return self.update(inp, mask, opt)
            # except: return self.update(self.x, mask, opt)

    def forward_wo_train(self, inp, mask):
        return self.mod(inp, mask)

    @torch.no_grad()
    def update(self, inp, mask, opt):
        if opt.first_time:
            for p in self.mod.parameters():
                opt.state[p]['feedback'] = True
                return self.forward_wo_train(inp, mask)
        for p in self.mod.parameters():
            Q_u = p.grad
            Q_uu_raw = opt.state[p]['square_avg'] * opt.alpha + (Q_u * Q_u) * (1- opt.alpha)
            Q_uu = Q_uu_raw.sqrt().add_(self.eps)
            break
        Q_x = self.x.grad.mean(dim=0).mean(dim=0)
        
        Q_ux = calc_q_ux_fc(Q_u, Q_x.unsqueeze(-1))
        big_k = calc_big_k_fc(Q_uu, Q_ux)
        term2 = torch.einsum('xy,zt->xt', big_k, (inp - self.x_old).mean(dim=0).mean(dim=0).unsqueeze(-1)).squeeze(-1).reshape(
            p.shape)
        Q_u -= opt.lrddp*(term2 * Q_uu)
        
        for p in self.mod.parameters():
            opt.update([p], [Q_u])
            break
        out = self.forward_wo_train(inp, mask)
        del self.x
        del self.x_old
        return out


def calc_q_ux_fc(q_u, q_x):
    return torch.einsum('xy,zt->xt', q_u.flatten(start_dim=0).unsqueeze(-1), torch.transpose(q_x,0,1))

def calc_small_k(q_uu, q_u):
    return -q_u/q_uu
        
def calc_big_k_fc(q_uu, q_ux):
    return -torch.einsum('x,xt->xt', 1/q_uu.flatten(start_dim=0), q_ux)

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.N = nlayers
        self.transformer_encoder = get_clones(Step(TransformerEncoderLayer(d_model, nhead, d_hid, dropout)), self.N)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor, update=False, opt=None) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]
        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        for i in range(self.N):
            src = self.transformer_encoder[i](src, src_mask, update, opt)
        output = self.decoder(src)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
