import os
from datetime import datetime

import torch

from reinforce import reinforce
from network import Agent
from recorder import Recorder

session = datetime.now().strftime("%m_%d_%Y, %H:%M:%S")
if not os.path.exists(os.path.join("./saves", session)):
    os.makedirs(os.path.join("./saves", session))
recorder = Recorder(os.path.join("./logs", session))
save_path = os.path.join("./saves", session)
agent = Agent()
baseline = Agent()
# agent.load_state_dict(torch.load("./agent-before-update.chkpt"))
# baseline.load_state_dict(torch.load("./baseline-before-update.chkpt"))
reinforce(agent, baseline, recorder, save_path)
