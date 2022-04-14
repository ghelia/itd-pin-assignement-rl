from generator import *
from config import Config
from network import Agent
import torch

agent = Agent()
baseline = Agent()
agent.eval()
baseline.eval()
baseline.load_state_dict(agent.state_dict())
items, nodes, edges = batch()


# selection_probs, selections, placement_probs, places = agent(items, nodes, edges, None, None)
# put_items(selections, places, items, nodes)
# 
# bsize = len(nodes)
# batch_range = torch.arange(bsize, dtype=torch.long, device = Config.device)

# success = check_placements(places, nodes, edges)
