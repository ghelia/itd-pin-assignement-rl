import torch

from attention_layer import AttentionEncoderLayer, Compatibility,  Linear
from config import Config


class ItemEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = Linear(Config.ntypes*2, Config.items_emb_dim)
        self.attentions = [AttentionEncoderLayer(
                Config.items_emb_dim,
                Config.items_nheads,
                Config.items_dense_hidden_dim
        ) for _ in range(Config.items_nlayers)]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.linear(inputs)
        for attention in self.attentions:
            outputs, _ = attention(outputs, outputs)
        return outputs


class NodeEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = Linear(Config.ntypes*2, Config.nodes_emb_dim)
        self.attentions = [AttentionEncoderLayer(
                Config.nodes_emb_dim,
                Config.nodes_nheads,
                Config.nodes_dense_hidden_dim
        ) for _ in range(Config.nodes_nlayers)]

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        outputs = self.linear(inputs)
        for attention in self.attentions:
            outputs, _ = attention(outputs, outputs, mask)
        return outputs

class ItemSelectionPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = AttentionEncoderLayer(
                Config.items_emb_dim,
                Config.items_query_nheads,
                Config.items_query_dense_hidden_dim
        )
        self.linear = Linear(Config.nodes_emb_dim, 1)
        self.softmax = torch.nn.Softmax(1)

    def forward(self, vitems: torch.Tensor, vnodes: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.attention(vitems, vnodes)
        outputs = self.linear(outputs)
        outputs = outputs.reshape(outputs.shape[:-1])
        outputs = self.softmax(outputs)
        return outputs
