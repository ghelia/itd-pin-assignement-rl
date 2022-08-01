from abc import abstractmethod
from typing import Tuple, List, Optional, Dict

import torch
import numpy as np

from .config import Config
from .generator import put_items, check_placements


class NodeItemEncoder(torch.nn.Module):
    @abstractmethod
    def forward(self,
                items: torch.Tensor,
                nodes: torch.Tensor,
                edges: torch.Tensor,
                available_mask: torch.Tensor,
                attention_view: Optional[Dict[str, np.ndarray]] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

class ItemSelectionDecoder(torch.nn.Module):
    @abstractmethod
    def forward(self,
                vitems: torch.Tensor,
                vnodes: torch.Tensor,
                already_selected: torch.Tensor,
                attention_view: Optional[Dict[str, np.ndarray]] = None
               ) -> torch.Tensor:
        raise NotImplementedError()

class NodeSelectionDecoder(torch.nn.Module):
    @abstractmethod
    def forward(self,
                vitem: torch.Tensor,
                vnodes: torch.Tensor,
                possible_placement: torch.Tensor,
                attention_view: Optional[Dict[str, np.ndarray]] = None
               ) -> torch.Tensor:
        raise NotImplementedError()

class Agent(torch.nn.Module):
    def __init__(self,
                 encoder: NodeItemEncoder,
                 idecoder: ItemSelectionDecoder,
                 ndecoder: NodeSelectionDecoder
                ) -> None:
        super().__init__()
        self.encoder = encoder
        self.idecoder = idecoder
        self.ndecoder = ndecoder

    def forward(self,
                items: torch.Tensor,
                nodes: torch.Tensor,
                edges: torch.Tensor,
                greedy: bool = False,
                return_info: bool = False,
                random_item_selection: bool = False
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        items_probs: List[torch.Tensor] = []
        nodes_probs: List[torch.Tensor] = []
        items_log_probs: List[torch.Tensor] = []
        nodes_log_probs: List[torch.Tensor] = []
        all_actions: List[torch.Tensor] = []
        all_rewards: List[torch.Tensor] = []
        bsize = len(nodes)
        assert len(items) == bsize
        assert len(edges) == bsize
        batch_range = torch.arange(bsize, dtype=torch.long, device=Config.device)
        already_selected = torch.zeros([bsize, Config.nitems], device=Config.device).bool()
        available_mask = torch.ones([bsize, Config.nitems, Config.nitems], device=Config.device).bool()
        attention_views = []
        for step in range(Config.nitems):
            attention_view = {}
            vitems, vnodes = self.encoder(items, nodes, edges, available_mask, attention_view)
            selection_probs = self.idecoder(vitems, vnodes, already_selected, attention_view=attention_view)
            if random_item_selection:
                selection_probs = torch.rand(selection_probs.shape, device=Config.device)
                selection_probs[already_selected] = -np.inf
                selection_probs = selection_probs.softmax(1)
            items_probs.append(selection_probs)
            selection_distribution = torch.distributions.categorical.Categorical(selection_probs)

            # if self.training:
            if not greedy:
                selections = selection_distribution.sample()
            else:
                selections = selection_probs.argmax(1)
            log_probs = selection_distribution.log_prob(selections)
            already_selected = already_selected.clone()
            already_selected[batch_range,selections] = True
            available_mask[batch_range, selections] = False
            available_mask[batch_range, :, selections] = False
            available_mask[batch_range, selections, selections] = True

            items_log_probs.append(log_probs)
            all_actions.append(selections)

            vitem = vitems[batch_range, selections].reshape([bsize, 1, vitems.shape[-1]])
            placement_probs = self.ndecoder(
                vitem,
                vnodes,
                nodes[:, :, Config.placed_flag_index].reshape([bsize, 1, Config.nitems]).bool().logical_not(),
                attention_view=attention_view
            )
            nodes_probs.append(placement_probs)
            placement_distribution = torch.distributions.categorical.Categorical(placement_probs)
            # if self.training:
            if not greedy:
                places = placement_distribution.sample()
            else:
                places = placement_probs.argmax(1)
            log_probs = placement_distribution.log_prob(places)
            nodes_log_probs.append(log_probs)
            all_actions.append(places)

            nodes = nodes.clone()
            put_items(selections, places, items, nodes)
            success = check_placements(places, nodes, edges, Config.check_neighbors)
            rewards = torch.tensor(success, device=Config.device).int() - 1.
            all_rewards.append(rewards)
            attention_views.append(attention_view)
        final_rewards = torch.stack(all_rewards, dim=1)
        final_rewards = (final_rewards.sum(1).bool().float() * -2) + 1
        if not return_info:
            return (
                torch.stack(items_probs, dim=1),
                torch.stack(nodes_probs, dim=1),
                torch.stack(items_log_probs, dim=1),
                torch.stack(nodes_log_probs, dim=1),
                torch.stack(all_actions, dim=1),
                final_rewards
            )
        return (
            torch.stack(items_probs, dim=1),
            torch.stack(nodes_probs, dim=1),
            torch.stack(items_log_probs, dim=1),
            torch.stack(nodes_log_probs, dim=1),
            torch.stack(all_actions, dim=1),
            final_rewards,
            torch.stack(all_rewards, dim=1),
            attention_views
        )
