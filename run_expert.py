# uv run python run_expert.py

import torch

from sim import Sim
from expert import Expert

from icecream import ic

sim = Sim()
sim.reset(torch.ones(sim.n, dtype=torch.bool, device=sim.device))
expert = Expert(sim)

for i in range(600 * 3):
    qdot = expert.act()
    obs, done, success = sim.step(qdot)
    if done.any():
        expert.reset(done)
        ic(i, done.sum().item(), success[done].float().mean().item())