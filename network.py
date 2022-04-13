from typing import Tuple, List

import torch
import numpy as np

from attention_layer import AttentionEncoderLayer, Compatibility,  Linear
from config import Config
from generator import put_items, check_placements


class ItemEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = Linear(Config.item_dims, Config.items_emb_dim)
        self.attentions = [AttentionEncoderLayer(
                Config.items_emb_dim,
                Config.items_nheads,
                Config.items_dense_hidden_dim
        ) for _ in range(Config.items_nlayers)]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.to(Config.device)
        outputs = self.linear(inputs)
        for attention in self.attentions:
            outputs, _ = attention(outputs, outputs)
        return outputs


class NodeEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = Linear(Config.node_dims, Config.nodes_emb_dim)
        self.attentions = [AttentionEncoderLayer(
                Config.nodes_emb_dim,
                Config.nodes_nheads,
                Config.nodes_dense_hidden_dim
        ) for _ in range(Config.nodes_nlayers)]

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        inputs = inputs.to(Config.device)
        mask = mask.to(Config.device)
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


class ItemPlacementPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.glimpse = AttentionEncoderLayer(
                Config.items_emb_dim,
                Config.glimpse_nheads,
                Config.glimpse_dense_hidden_dim
        )
        self.compatibility = Compatibility(
            Config.nodes_emb_dim,
            Config.items_emb_dim,
            Config.compatibility_emb
        )

    def forward(self, vitem: torch.Tensor, vnodes: torch.Tensor, possible_placement: torch.Tensor) -> torch.Tensor:
        glimpse, _ = self.glimpse(vitem, vnodes)
        outputs = self.compatibility(vnodes, glimpse, possible_placement)
        return outputs.reshape([-1, Config.nitems])


class Agent(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.i_encoder = ItemEncoder()
        self.n_encoder = NodeEncoder()
        self.selection_policy = ItemSelectionPolicy()
        self.placement_policy = ItemPlacementPolicy()

    def forward(self,
                items: torch.Tensor,
                nodes: torch.Tensor,
                edges: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Arguments
        ---------
            items
            nodes
            edges
        Returns
        -------
            log_probs
            actions
            rewards
        """
        all_log_probs: List[torch.Tensor] = []
        all_actions: List[torch.Tensor] = []
        all_rewards: List[torch.Tensor] = []
        bsize = len(nodes)
        assert len(items) == bsize
        assert len(edges) == bsize
        vitems = self.i_encoder(items)
        batch_range = torch.arange(bsize, dtype=torch.long, device=Config.device)
        for step in range(Config.nitems):
            vnodes = self.n_encoder(nodes, edges.bool())
            selection_probs = self.selection_policy(vitems, vnodes)
            selection_distribution = torch.distributions.categorical.Categorical(selection_probs)
            selections = selection_distribution.sample()
            # selections = selection_probs.argmax(1).cpu().numpy()
            log_probs = selection_distribution.log_prob(selections)
            all_log_probs.append(log_probs)
            all_actions.append(selections)

            vitem = vitems[batch_range, selections].reshape([bsize, 1, Config.items_emb_dim])
            placement_probs = self.placement_policy(
                vitem,
                vnodes,
                nodes[:, :, Config.placed_flag_index].reshape([bsize, 1, Config.nitems]).bool().logical_not()
            )
            placement_distribution = torch.distributions.categorical.Categorical(placement_probs)
            places = placement_distribution.sample()
            # places = placement_probs.argmax(1).cpu().numpy()
            log_probs = placement_distribution.log_prob(places)
            all_log_probs.append(log_probs)
            all_actions.append(places)

            nodes = nodes.clone()
            put_items(selections, places, items, nodes)
            success = check_placements(places, nodes, edges)
            rewards = torch.tensor(success, device=Config.device).int() - 1.
            all_rewards.append(rewards)
        return (torch.stack(all_log_probs, dim=1), torch.stack(all_actions, dim=1), torch.stack(all_rewards, dim=1))
