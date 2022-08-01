import os
from typing import Tuple, List

import torch
import numpy as np
from tqdm import tqdm

from .config import Config
from .network import Agent
from .generator import batch
from .recorder import Recorder


def reinforce(agent: Agent, baseline: Agent, recorder: Recorder, save_path: str) -> None:
    optimizer = torch.optim.Adam(agent.parameters(), lr=Config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=Config.learning_rate_decay)
    agent.train()
    baseline.train()
    baseline.load_state_dict(agent.state_dict())
    random_item_selection = Config.random_item_selection

    for E in range(Config.n_epoch):
        agent.train()
        baseline.train()
        print(f"")
        print(f"Epoch {E}")
        print(f"size {Config.nitems}")
        all_losses = []
        all_rewards = []
        all_baseline_rewards = []
        for e in tqdm(range(Config.n_episode)):
            optimizer.zero_grad()
            items, nodes, edges = batch(npins=Config.nitems)

            with torch.no_grad():
                _, _, _, _, _, baseline_rewards = baseline(items, nodes, edges, greedy=True)
            (
                items_probs,
                nodes_probs,
                items_log_probs,
                nodes_log_probs,
                actions,
                rewards
            ) = agent(items, nodes, edges, random_item_selection=random_item_selection)
            all_rewards.append(rewards.mean().item())
            all_baseline_rewards.append(baseline_rewards.mean().item())

            log_probs = items_log_probs.sum(1)*Config.selection_policy_weight + nodes_log_probs.sum(1)*(1. - Config.selection_policy_weight)
            loss = -((rewards - baseline_rewards) * log_probs).mean()
            # loss = ((rewards.sum(1)) * log_probs.sum(1)).mean()
            all_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            recorder.scalar(loss.item(), "loss")
            recorder.scalar(rewards.mean().item(), "reward")
            recorder.scalar(scheduler.get_last_lr()[0], "learning rate")
        if (np.mean(all_rewards) - np.mean(all_baseline_rewards)) / np.abs(np.mean(all_baseline_rewards)) > Config.paired_test_alpha:
            torch.save(agent.state_dict(), os.path.join(save_path, f'agent-{E}-before-baseline-update.chkpt'))
            torch.save(baseline.state_dict(), os.path.join(save_path, f'baseline-{E}-before-baseline-update.chkpt'))
            print("Update baseline policy")
            baseline.load_state_dict(agent.state_dict())
            print("Save models")
        print(f"Baseline Score {np.mean(all_baseline_rewards)}")
        print(f"Score {np.mean(all_rewards)}")
        print(f"Loss {np.mean(all_losses)}")
        recorder.hist(rewards.view([-1]).tolist(), "rewards distribution")
        recorder.hist(items_probs.view([-1]).tolist(), "selection probabilities")
        recorder.hist(nodes_probs.view([-1]).tolist(), "placement probabilities")
        recorder.gradients_and_weights(agent)
        recorder.gradients_and_weights(baseline)

        recorder.end_epoch()
        scheduler.step()

        # agent.eval()
        with torch.no_grad():
            items, nodes, edges = batch(npins=Config.nitems)
            _, _, _, _, _, rewards = agent(items, nodes, edges, greedy=True)
        success = rewards.gt(0).sum().item()
        total = len(rewards)
        print(f"Evaluation {success} / {total}")
        if (E + 1) % 100 == 0:
            torch.save(agent.state_dict(), os.path.join(save_path, f'agent-{E}.chkpt'))
        if success/total > 0.95:
            torch.save(agent.state_dict(), os.path.join(save_path, f'agent-{E}-increase-size.chkpt'))
            print()
            print(f"Increase size")
            print()
            Config.nitems += 1

