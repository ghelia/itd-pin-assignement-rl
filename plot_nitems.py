import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from pa.generator import *
from pa.config import Config
from pa.network import Agent

Config.batch_size = 10
Config.overlap_ratio = 0.33
START = 5
END = 50
LENGTH = END - START

class AgentWrapper:
    def __init__(self, name: str, checkpoint: str, greedy: bool, random_item: bool, color: str) -> None:
        self.name = name
        agent = Agent()
        agent.train()
        agent.load_state_dict(torch.load(checkpoint, map_location=Config.device))
        self.model = agent
        self.greedy = greedy
        self.random_item = random_item
        self.color = color

    def get_results(self, start: int, end: int) -> np.ndarray:
        results = []
        for nitems in tqdm(range(start, end)):
            success = 0
            total = 0
            for _ in range(1):
                Config.nitems = nitems
                items, nodes, edges = batch(npins=Config.nitems, batch_size=Config.batch_size)
                with torch.no_grad():
                    _, _, _, _, _, rewards = self.model(items, nodes, edges, greedy=self.greedy, random_item_selection=self.random_item)
                success += rewards.gt(0).sum().item()
                total += len(rewards)
            results.append(success/total)
        return np.array(results)

    def plot(self, start: int, end: int) -> None:
        results = self.get_results(start, end)
        plt.plot(results, color=self.color, label=self.name)
        plt.fill_between(np.arange(len(results)), results, color=self.color, alpha=0.3)



# with/without item selection policy
# model1 = AgentWrapper("use item selection policy", "./trained/agent-before-update.chkpt", greedy=True, random_item=False, color="blue")
# model2 = AgentWrapper("random item selection", "./trained/agent-before-update.chkpt", greedy=True, random_item=True, color="red")
# model1.plot(START, END)
# model2.plot(START, END)

# trained with different policy selection weights
model = AgentWrapper("weight=0.0", "./trained/policy_sensitivity/w0.0.chkpt", greedy=True, random_item=False, color="red")
model.plot(START, END)
model = AgentWrapper("weight=0.0001", "./trained/policy_sensitivity/w0.0001.chkpt", greedy=True, random_item=False, color="orange")
model.plot(START, END)
model = AgentWrapper("weight=0.001", "./trained/policy_sensitivity/w0.001.chkpt", greedy=True, random_item=False, color="pink")
model.plot(START, END)
model = AgentWrapper("weight=0.01", "./trained/policy_sensitivity/w0.01.chkpt", greedy=True, random_item=False, color="yellow")
model.plot(START, END)
model = AgentWrapper("weight=0.1", "./trained/policy_sensitivity/w0.1.chkpt", greedy=True, random_item=False, color="brown")
model.plot(START, END)
model = AgentWrapper("weight=0.25", "./trained/policy_sensitivity/w0.25.chkpt", greedy=True, random_item=False, color="purple")
model.plot(START, END)
model = AgentWrapper("weight=0.5", "./trained/policy_sensitivity/w0.5.chkpt", greedy=True, random_item=False, color="green")
model.plot(START, END)
model = AgentWrapper("weight=0.75", "./trained/policy_sensitivity/w0.75.chkpt", greedy=True, random_item=False, color="blue")
model.plot(START, END)

plt.xlabel("number of items")
plt.xticks(np.arange(LENGTH), np.arange(START, START + LENGTH))
plt.ylabel("success ratio")
plt.title(f"Overlap ratio : {Config.overlap_ratio}")
plt.legend()
plt.show()
