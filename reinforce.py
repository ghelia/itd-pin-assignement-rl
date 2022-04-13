import os
from typing import Tuple, List
import torch
import numpy as np

from config import Config
from network import Agent
from generator import batch


def reinforce(agent: Agent, baseline: Agent) -> None:
    optimizer = torch.optim.Adam(agent.parameters(), lr=Config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=Config.learning_rate_decay)
    agent.train()
    baseline.eval()
    baseline.load_state_dict(agent.state_dict())

    best_score = -np.inf
    for E in range(Config.n_epoch):
        print(f"Epoch {E}")
        all_rewards = []
        all_baseline_rewards = []
        for e in range(Config.n_episode):
            optimizer.zero_grad()
            items, nodes, edges = batch()
            log_probs, actions, rewards = agent(items, nodes, edges)

            with torch.no_grad():
                _, _, baseline_rewards = baseline(items, nodes, edges)
            all_rewards.append(rewards.sum(1).mean().item())
            all_baseline_rewards.append(baseline_rewards.sum(1).mean().item())

            loss = ((baseline_rewards.sum(1) - rewards.sum(1)) * log_probs.sum(1)).mean()
            loss.backward()
            optimizer.step()
        if np.mean(all_rewards) > best_score:
            best_score = np.mean(all_rewards)
            # torch.save(policy.state_dict(), os.path.join(save_path, f'{E}-{best_score}.chkpt'))
        if (np.mean(all_rewards) - np.mean(all_baseline_rewards)) / np.abs(np.mean(all_baseline_rewards)) > Config.paired_test_alpha:
            print("Update baseline policy")
            baseline.load_state_dict(agent.state_dict())
        print(f"Score {np.mean(all_rewards)}")
        scheduler.step()
