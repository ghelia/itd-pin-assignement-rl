import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from generator import *
from config import Config
from network import Agent

Config.batch_size = 100
Config.overlap_ratio = 0.33



agent = Agent()
agent.train()
agent.load_state_dict(torch.load("./agent-5261-increase-size.chkpt"))



results = []
for nitems in tqdm(range(4, 200)):
    success = 0
    total = 0
    for _ in range(1):
        Config.nitems = nitems
        items, nodes, edges = batch(npins=Config.nitems, batch_size=Config.batch_size)
        with torch.no_grad():
            _, _, _, _, _, rewards = agent(items, nodes, edges, greedy=True)
        success += rewards.gt(0).sum().item()
        total += len(rewards)
    results.append(success/total)
results = np.array(results)
plt.plot(results)
plt.fill_between(np.arange(len(results)), results, color="blue", alpha=0.3)
plt.xlabel("number of items")
plt.ylabel("success ratio")
plt.title(f"Overlap ratio : {Config.overlap_ratio}")
plt.show()