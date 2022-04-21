from generator import *
from config import Config
from network import Agent
import torch

Config.batch_size = 300
Config.nitems = 21
agent = Agent()
baseline = Agent()
agent.train()
baseline.train()
items, nodes, edges = batch(npins=Config.nitems)

agent.load_state_dict(torch.load("./agent-before-update.chkpt"))
baseline.load_state_dict(torch.load("./baseline-before-update.chkpt"))

_, _, _, _, _, agent_rewards = agent(items, nodes, edges, greedy=True)
with torch.no_grad():
    _, _, _, _, _, baseline_rewards = baseline(items, nodes, edges, greedy=True)

print("agent before : ", agent_rewards.mean())
print("baseline before : ", baseline_rewards.mean())

baseline.load_state_dict(agent.state_dict())

_, _, _, _, _, agent_rewards = agent(items, nodes, edges, greedy=True)
with torch.no_grad():
    # baseline.eval()  # DRASTICALLY DECREASE MODEL ACCURACY
    _, _, _, _, _, baseline_rewards = baseline(items, nodes, edges, greedy=True)

print("agent after : ", agent_rewards.mean())
print("baseline after : ", baseline_rewards.mean())

# selection_probs, selections, placement_probs, places = agent(items, nodes, edges, None, None)
# put_items(selections, places, items, nodes)
# 
# bsize = len(nodes)
# batch_range = torch.arange(bsize, dtype=torch.long, device = Config.device)

# success = check_placements(places, nodes, edges)
