# uv run python run_expert.py

import torch

from sim import Sim
from expert import Expert

sim = Sim()
sim.reset(torch.ones(sim.n, dtype=torch.bool, device=sim.device))
expert = Expert(sim)

for i in range(600):
    qdot = expert.act()
    sim.step(qdot)
