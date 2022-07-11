import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from generator import *
from config import Config
from network import Agent

Config.batch_size = 200
Config.overlap_ratio = 0.33


def get_results(agent, start, end, random_item):
    results = []
    for nitems in tqdm(range(start, end)):
        success = 0
        total = 0
        for _ in range(1):
            Config.nitems = nitems
            items, nodes, edges = batch(npins=Config.nitems, batch_size=Config.batch_size)
            with torch.no_grad():
                _, _, _, _, _, rewards = agent(items, nodes, edges, greedy=True, random_item_selection=random_item)
            success += rewards.gt(0).sum().item()
            total += len(rewards)
        results.append(success/total)
    return np.array(results)


agent = Agent()
agent.train()
# agent.load_state_dict(torch.load("./agent-5261-increase-size.chkpt"))
agent.load_state_dict(torch.load("./agent-before-update.chkpt"))

start = 3
end = 100



results = get_results(agent, start, end, False)
plt.plot(results, color="blue", label="use item selection policy")
plt.fill_between(np.arange(len(results)), results, color="blue", alpha=0.3)

results = get_results(agent, start, end, True)
plt.plot(results, color="red", label="random item selection")
plt.fill_between(np.arange(len(results)), results, color="red", alpha=0.3)

plt.xlabel("number of items")
plt.xticks(np.arange(len(results)), np.arange(start, start + len(results)))
plt.ylabel("success ratio")
plt.title(f"Overlap ratio : {Config.overlap_ratio}")
plt.legend()
plt.show()
