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
    baseline.train()
    baseline.load_state_dict(agent.state_dict())

    best_score = -np.inf
    for E in range(Config.n_epoch):
        agent.train()
        baseline.eval()
        print(f"")
        print(f"Epoch {E}")
        print(f"size {Config.nitems}")
        all_losses = []
        all_rewards = []
        all_baseline_rewards = []
        for e in range(Config.n_episode):
            optimizer.zero_grad()
            items, nodes, edges = batch(npins=Config.nitems)

            with torch.no_grad():
                _, _, baseline_rewards = baseline(items, nodes, edges)
            log_probs, actions, rewards = agent(items, nodes, edges)
            all_rewards.append(rewards.sum(1).mean().item())
            all_baseline_rewards.append(baseline_rewards.sum(1).mean().item())

            loss = -((rewards.sum(1) - baseline_rewards.sum(1)) * log_probs.sum(1)).mean()
            # loss = ((rewards.sum(1)) * log_probs.sum(1)).mean()
            all_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        if np.mean(all_rewards) > best_score:
            best_score = np.mean(all_rewards)
            # torch.save(policy.state_dict(), os.path.join(save_path, f'{E}-{best_score}.chkpt'))
        if (np.mean(all_rewards) - np.mean(all_baseline_rewards)) / np.abs(np.mean(all_baseline_rewards)) > Config.paired_test_alpha:
            print("Update baseline policy")
            baseline.load_state_dict(agent.state_dict())
        print(f"Baseline Score {np.mean(all_baseline_rewards)}")
        print(f"Score {np.mean(all_rewards)}")
        print(f"Loss {np.mean(all_losses)}")
        scheduler.step()

        agent.eval()
        with torch.no_grad():
            items, nodes, edges = batch(npins=Config.nitems)
            _, _, rewards = agent(items, nodes, edges)
        success = rewards.bool().sum(1).bool().logical_not().sum().item()
        total = len(rewards)
        print(f"Evaluation {success} / {total}")
        if success/total > 0.98:
            print()
            print(f"Increase size")
            print()
            Config.nitems += 1

