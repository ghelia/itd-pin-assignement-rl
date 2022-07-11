from typing import Tuple, Optional, Dict

import torch
import numpy as np
from torch.nn import MultiheadAttention as MHA

from config import Config



class Linear(torch.nn.Module):
    def __init__(self, ins: int, outs: int, use_bias: bool = False) -> None:
        super().__init__()
        self.bias: Optional[torch.nn.Parameter] = None
        if use_bias is True:
            self.bias = torch.nn.Parameter(
                data=(torch.rand([outs], device=Config.device) - .5) * .05,
                requires_grad = True)
        self.weights = torch.nn.Parameter(
            data=(torch.rand([ins, outs], device=Config.device) - .5) * .05,
            requires_grad = True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            return torch.matmul(inputs, self.weights) + self.bias
        return torch.matmul(inputs, self.weights)


class Dense(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.hidden = Linear(in_dim, hidden_dim, use_bias=True)
        self.out = Linear(hidden_dim, out_dim, use_bias=True)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.out(self.relu(self.hidden(inputs)))


class Compatibility(torch.nn.Module):
    def __init__(self, k_in_dim: int, q_in_dim: int, kq_dim: int, clip: Optional[int] = 10) -> None:
        super().__init__()
        self.query = Linear(q_in_dim, kq_dim)
        self.key = Linear(k_in_dim, kq_dim)
        self.clip = clip
        self.kq_dim = kq_dim

    def forward(self, k_inputs: torch.Tensor, q_inputs: torch.Tensor, adjancy: torch.Tensor) -> torch.Tensor:
        k = self.key(k_inputs)
        q = self.query(q_inputs)
        compatibility = torch.matmul(q,k.transpose(2,1)) / torch.sqrt(torch.tensor(self.kq_dim))
        if self.clip is not None:
            compatibility = torch.tanh(compatibility) * self.clip
        compatibility = torch.where(adjancy.bool(), compatibility, torch.full(compatibility.shape, -np.inf, device=Config.device))
        return compatibility.softmax(2)


class BatchNorm(torch.nn.Module):
    def __init__(self, nfeatures:int) -> None:
        super().__init__()
        self.bn1d = torch.nn.BatchNorm1d(nfeatures, device=Config.device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.bn1d(inputs.transpose(2,1)).transpose(2,1).to(Config.device)


class AttentionEncoderLayer(torch.nn.Module):
    def __init__(self,
                emb_dim: int,
                nhead: int,
                dense_hidden_dim: int,
                name: str
               ) -> None:
        super().__init__()
        self.mha = MHA(emb_dim, nhead, batch_first=True, device=Config.device)
        self.nhead = nhead
        self.ff = Dense(emb_dim, dense_hidden_dim, emb_dim)
        self.mha_batchnorm = BatchNorm(emb_dim)
        self.ff_batchnorm = BatchNorm(emb_dim)
        self.name = name

    def forward(self,
                queries: torch.Tensor,
                keys_values: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                attention_view: Optional[Dict[str, np.ndarray]]= None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = queries.shape[0]
        assert keys_values.shape[0] == batch_size
        nqueries = queries.shape[1]
        nkvs = keys_values.shape[1]
        if mask is not None:
            mask = mask.logical_not()
            mask = mask.repeat([1,self.nhead,1])
            mask = mask.reshape([batch_size*self.nhead, nqueries, nkvs])
            mask = mask.to(Config.device)
        mha_outputs, attention = self.mha(queries, keys_values, keys_values, attn_mask=mask)
        assert mha_outputs.shape == queries.shape
        mha_outputs = self.mha_batchnorm(mha_outputs + queries)
        outputs = self.ff(mha_outputs)
        assert mha_outputs.shape == outputs.shape
        outputs = self.ff_batchnorm(mha_outputs + outputs)
        if attention_view is not None:
            attention_view[self.name] = attention.detach().cpu().numpy()
        return outputs, attention
