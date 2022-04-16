import os
from datetime import datetime

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
reinforce(agent, baseline, recorder, save_path)
