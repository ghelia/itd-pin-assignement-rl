"""Basic neural network layers"""
from typing import Optional, List, Callable, Union

import torch
import numpy as np

from ..utils.config import Config

class Linear(torch.nn.Module):
    """Linear layer
    Parameters
    ----------
    ins: int
        dimensions of the inputs
    outs: int
        dimensions of the outputs
    use_bias: bool
        wether to use bias or not (default: False)
    """
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
        """linear transformation of the inputs
        Parameters
        ----------
        inputs: torch.Tensor
            input tensor
        """
        if self.bias is not None:
            return torch.matmul(inputs, self.weights) + self.bias
        return torch.matmul(inputs, self.weights)

class Gather(torch.nn.Module):
    def _index_offsets(self, shape: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.linspace(
            0,
            int((shape[:dim + 1].prod() - shape[dim]).item()),
            int(shape[:dim].prod().item()),
            device=Config.device
        )

    def forward(self, inputs: torch.Tensor, indexs: torch.Tensor, inplace: bool = False) -> Union[torch.Tensor, np.ndarray]:
        dim = len(indexs.shape) - 1
        assert(len(indexs) > 0)
        assert(indexs.shape[-1] == 1)
        assert(indexs.shape[:-1] == inputs.shape[:dim])
        indexs = indexs.flatten()
        flat_inputs = inputs.view([-1] + list(inputs.shape[dim+1:]))
        idx_offsets = self._index_offsets(torch.tensor(inputs.shape), dim)
        idx = indexs + idx_offsets
        if inplace is False:
            return flat_inputs[idx.long()].view(list(inputs.shape[:dim]) + list(inputs.shape[dim+1:]))
        np_tensors = np.empty(len(idx), dtype=object)
        for npi, ti in enumerate(idx.long()):
            np_tensors[npi] = flat_inputs[ti]
        return np_tensors.reshape(list(inputs.shape[:dim]))


"""Modules for transformed based networks"""
from typing import Optional, List, Callable

import torch
import numpy as np
from .layers import Linear
from ..utils.config import Config


class Compatibility(torch.nn.Module):
    def __init__(self, k_in_dim: int, q_in_dim: int, kq_dim: int, clip: Optional[int] = None) -> None:
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

class AttentionHead(torch.nn.Module):

    def __init__(self, kv_in_dim: int, q_in_dim: int, v_dim: int, kq_dim: int) -> None:
        super().__init__()
        self.value = Linear(kv_in_dim, v_dim)
        self.out = Linear(v_dim, q_in_dim)
        self.v_dim = v_dim
        self.kq_dim = kq_dim
        self.compatibility = Compatibility(kv_in_dim, q_in_dim, kq_dim)

    def forward(self, kv_inputs: torch.Tensor, q_inputs: torch.Tensor, adjancy: torch.Tensor) -> torch.Tensor:
        B = q_inputs.shape[0]
        QL = q_inputs.shape[1]
        KVL = kv_inputs.shape[1]
        D = self.v_dim

        v = self.value(kv_inputs)
        rv = v.repeat([1,QL,1]).view([B,QL,KVL,D])

        attention = self.compatibility(kv_inputs, q_inputs, adjancy)
        ra = attention.view([B,QL,KVL,1]).repeat([1,1,1,D])

        return self.out((ra*rv).sum(2))

class MultiHeadAttention(torch.nn.Module):

    def __init__(self, nheads: int, kv_in_dim: int, q_in_dim: int) -> None:
        super().__init__()
        self.heads = torch.nn.ModuleList([AttentionHead(kv_in_dim, q_in_dim, kv_in_dim // nheads, kv_in_dim // nheads) for n in range(nheads)])

    def forward(self, kv_inputs: torch.Tensor, q_inputs: torch.Tensor, adjancy: torch.Tensor) -> torch.Tensor:
        outs = torch.zeros(q_inputs.shape, device=Config.device)
        for head in self.heads:
            outs += head(kv_inputs, q_inputs, adjancy)
        return outs

class FullyConnected(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.hidden = Linear(in_dim, hidden_dim, use_bias=True)
        self.out = Linear(hidden_dim, out_dim, use_bias=True)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.out(self.relu(self.hidden(inputs)))

class AttentionEncoder(torch.nn.Module):
    def __init__(self,
                 mha_in_dim: int,
                 ff_hidden_dim: int,
                 mha_nheads: int,
                 n_nodes: int) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(mha_nheads, mha_in_dim, mha_in_dim)
        self.ff = FullyConnected(mha_in_dim, ff_hidden_dim, mha_in_dim)
        self.mha_batchnorm = torch.nn.BatchNorm1d(n_nodes, device=Config.device)
        self.ff_batchnorm = torch.nn.BatchNorm1d(n_nodes, device=Config.device)

    def forward(self, inputs: torch.Tensor, adjancy: torch.Tensor) -> torch.Tensor:
        mha_outputs = self.mha_batchnorm(self.mha(inputs, inputs, adjancy) + inputs)
        ff_outputs = self.ff_batchnorm(self.ff(mha_outputs) + mha_outputs)
        return ff_outputs


class AttentionDecoder(torch.nn.Module):
    def __init__(self,
                 kv_in_dim: int,
                 q_in_dim: int,
                 nheads: int) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(nheads, kv_in_dim, q_in_dim)

    def forward(self, kv_inputs: torch.Tensor, q_inputs: torch.Tensor, adjancy: torch.Tensor) -> torch.Tensor:
        return self.mha(kv_inputs, q_inputs, adjancy)


class AttentionProbability(torch.nn.Module):
    def __init__(self, k_in_dim: int, q_in_dim: int, kq_dim: int, clip: Optional[int] = 10) -> None:
        super().__init__()
        self.compatibility = Compatibility(k_in_dim, q_in_dim, kq_dim, clip)

    def forward(self, kv_inputs: torch.Tensor, q_inputs: torch.Tensor, adjancy: torch.Tensor) -> torch.Tensor:
        compatibility = self.compatibility(kv_inputs, q_inputs, adjancy)
        return compatibility
