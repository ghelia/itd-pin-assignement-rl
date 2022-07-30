from typing import Tuple, List, Optional, Dict
import torch
import numpy as np

from .config import Config as MainConfig
from .attention_layer import AttentionEncoderLayer, Compatibility,  Linear
from .network import NodeItemEncoder, ItemSelectionDecoder, NodeSelectionDecoder

class Config:
    items_emb_dim = 128
    items_dense_hidden_dim = 512
    items_nheads = 8
    items_nlayers = 3

    items_query_dense_hidden_dim = 512
    items_query_nheads = 8

    glimpse_dense_hidden_dim = 512
    glimpse_nheads = 8

    compatibility_emb = 128

    nodes_emb_dim = 128
    nodes_dense_hidden_dim = 512
    nodes_nheads = 8
    nodes_nlayers = 3


class MHAItemEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = Linear(MainConfig.item_dims, Config.items_emb_dim)
        self.attentions = torch.nn.Sequential(*[AttentionEncoderLayer(
                Config.items_emb_dim,
                Config.items_nheads,
                Config.items_dense_hidden_dim,
                f"item_encoder_{idx}"
        ) for idx in range(Config.items_nlayers)])

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor, attention_view: Optional[Dict[str, np.ndarray]] = None) -> torch.Tensor:
        inputs = inputs.to(MainConfig.device)
        outputs = self.linear(inputs)
        for attention in self.attentions:
            outputs, _ = attention(outputs, outputs, mask, attention_view=attention_view)
        return outputs


class MHANodeEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = Linear(MainConfig.node_dims, Config.nodes_emb_dim)
        self.attentions = torch.nn.Sequential(*[AttentionEncoderLayer(
                Config.nodes_emb_dim,
                Config.nodes_nheads,
                Config.nodes_dense_hidden_dim,
                f"node_encoder_{idx}"
        ) for idx in range(Config.nodes_nlayers)])

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor, attention_view: Optional[Dict[str, np.ndarray]] = None) -> torch.Tensor:
        inputs = inputs.to(MainConfig.device)
        mask = mask.to(MainConfig.device)
        outputs = self.linear(inputs)
        for attention in self.attentions:
            outputs, _ = attention(outputs, outputs, mask, attention_view=attention_view)
        return outputs


class MHANodeItemEncoder(NodeItemEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.i_encoder = MHAItemEncoder()
        self.n_encoder = MHANodeEncoder()

    def forward(self,
                items: torch.Tensor,
                nodes: torch.Tensor,
                edges: torch.Tensor,
                available_mask: torch.Tensor,
                attention_view: Optional[Dict[str, np.ndarray]] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
       vitems = self.i_encoder(items, available_mask, attention_view=attention_view)
       vnodes = self.n_encoder(nodes, edges.bool(), attention_view=attention_view)
       return (vitems, vnodes)


class MHAItemSelectionDecoder(ItemSelectionDecoder):
    def __init__(self) -> None:
        super().__init__()
        self.attention = AttentionEncoderLayer(
                Config.items_emb_dim,
                Config.items_query_nheads,
                Config.items_query_dense_hidden_dim,
                "item_selection"
        )
        self.linear = Linear(Config.nodes_emb_dim, 1)
        self.softmax = torch.nn.Softmax(1)

    def forward(self, vitems: torch.Tensor, vnodes: torch.Tensor, already_selected: torch.Tensor, attention_view: Optional[Dict[str, np.ndarray]] = None) -> torch.Tensor:
        outputs, _ = self.attention(vitems, vnodes, attention_view=attention_view)
        outputs = self.linear(outputs)
        outputs = outputs.reshape(outputs.shape[:-1])
        outputs[already_selected] = -np.inf
        outputs = self.softmax(outputs)
        return outputs


class MHANodeSelectionDecoder(NodeSelectionDecoder):
    def __init__(self) -> None:
        super().__init__()
        self.glimpse = AttentionEncoderLayer(
                Config.items_emb_dim,
                Config.glimpse_nheads,
                Config.glimpse_dense_hidden_dim,
                "node_selection_glimpse"
        )
        self.compatibility = Compatibility(
            Config.nodes_emb_dim,
            Config.items_emb_dim,
            Config.compatibility_emb,
        )

    def forward(self, vitem: torch.Tensor, vnodes: torch.Tensor, possible_placement: torch.Tensor, attention_view: Optional[Dict[str, np.ndarray]] = None) -> torch.Tensor:
        glimpse, _ = self.glimpse(vitem, vnodes, attention_view=attention_view)
        outputs = self.compatibility(vnodes, glimpse, possible_placement)
        if attention_view is not None:
            attention_view["node_selection_compatibility"] = outputs.detach().cpu().numpy()
        return outputs.reshape([-1, MainConfig.nitems])
