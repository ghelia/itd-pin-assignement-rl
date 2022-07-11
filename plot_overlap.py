import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from generator import *
from config import Config
from network import Agent

Config.batch_size = 200
Config.nitems = 50



agent = Agent()
agent.train()
agent.load_state_dict(torch.load("./docker2.chkpt"))


start = 0.24
end = 0.48
gap = 0.004

results = []
for overlap in tqdm(np.arange(start, end, gap)):
    success = 0
    total = 0
    for _ in range(1):
        Config.overlap_ratio = overlap
        items, nodes, edges = batch(npins=Config.nitems, batch_size=Config.batch_size, overlap_ratio=Config.overlap_ratio)
        with torch.no_grad():
            _, _, _, _, _, rewards = agent(items, nodes, edges, greedy=True)
        success += rewards.gt(0).sum().item()
        total += len(rewards)
    results.append(success/total)
results = np.array(results)
plt.plot(results)
plt.fill_between(np.arange(len(results)), results, color="blue", alpha=0.3)
ticks = np.arange(start, end, gap*4).round(3)
plt.xticks(np.arange(len(ticks))*4, ticks)
plt.xlabel("overlap percent")
plt.ylabel("success ratio")
plt.title(f"Number items : {Config.nitems}")
plt.show()
