import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from generator import *
from config import Config
from network import Agent

Config.batch_size = 10
Config.device = "cpu"



agent = Agent()
agent.train()
agent.load_state_dict(torch.load("./docker3.chkpt", map_location="cpu"))



results = []
for nitems in tqdm(range(3, 40)):
    Config.nitems = nitems
    items, nodes, edges = batch(npins=Config.nitems, batch_size=Config.batch_size)
    _, _, _, _, _, rewards = agent(items, nodes, edges, greedy=True)
    success = rewards.gt(0).sum().item()
    total = len(rewards)
    print(f"Evaluation {success} / {total}")
    print("agent : ", rewards.mean())
    results.append(success/total)
plt.plot(results)
plt.show()
