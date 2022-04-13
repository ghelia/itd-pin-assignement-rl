from reinforce import reinforce
from network import Agent

agent = Agent()
baseline = Agent()
reinforce(agent, baseline)
