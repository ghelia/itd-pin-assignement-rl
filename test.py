from generator import *
from network import ItemEncoder, NodeEncoder, ItemSelectionPolicy
import torch
i_encoder = ItemEncoder()
n_encoder = NodeEncoder()
selection_policy = ItemSelectionPolicy()
items, nodes, edges = batch()

vitems = i_encoder(torch.tensor(items, dtype=torch.float32))
print(items.shape)
print(vitems.shape)

vnodes = n_encoder(torch.tensor(nodes, dtype=torch.float32), torch.tensor(edges, dtype=torch.bool))
print(nodes.shape)
print(vnodes.shape)

probs = selection_policy(vitems, vnodes)
