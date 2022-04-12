from generator import *
from config import Config
from network import ItemEncoder, NodeEncoder, ItemSelectionPolicy, ItemPlacementPolicy
import torch

i_encoder = ItemEncoder()
n_encoder = NodeEncoder()
selection_policy = ItemSelectionPolicy()
placement_policy = ItemPlacementPolicy()
items, nodes, edges = batch()

vitems = i_encoder(torch.tensor(items, dtype=torch.float32))
print("vitems : ", vitems.shape)

vnodes = n_encoder(torch.tensor(nodes, dtype=torch.float32), torch.tensor(edges, dtype=torch.bool))
print("vnodes : ", vnodes.shape)

selection_probs = selection_policy(vitems, vnodes)
print("selection probs : ", selection_probs.shape)

selections = selection_probs.argmax(1).cpu().numpy()
ar = torch.arange(Config.batch_size, dtype=torch.long, device = Config.device)
vitem = vitems[ar, selections].reshape([Config.batch_size, 1, Config.items_emb_dim])

possibile_placements = torch.ones([Config.batch_size, 1, Config.nitems], dtype=torch.bool, device=Config.device)
placement_probs = placement_policy(vitem, vnodes, possibile_placements)
print("placement probs : ", placement_probs.shape)

places = placement_probs.argmax(1).cpu().numpy()
possibile_placements[ar, 0, places.reshape([-1])] = False

put_items(selections, places, items, nodes)
print(check_placements(places, nodes, edges))


