from generator import *
from config import Config
from network import Agent
import torch

agent = Agent()
items, nodes, edges = batch()
log_probs, actions, rewards = agent(items, nodes, edges)

# selection_probs, selections, placement_probs, places = agent(items, nodes, edges, None, None)
# put_items(selections, places, items, nodes)
# 
# bsize = len(nodes)
# batch_range = torch.arange(bsize, dtype=torch.long, device = Config.device)

# success = check_placements(places, nodes, edges)
