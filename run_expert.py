# uv run python run_expert.py

import torch
from sim import Sim
from expert import Expert
from utils import Timer
from icecream import ic

# steps, episodes
T = 600
episodes = 3

# sim, expert
sim = Sim()
expert = Expert(sim)

zero_qdot = torch.zeros(sim.n, 3, device=sim.device)
true_mask = torch.ones(sim.n, dtype=torch.bool, device=sim.device)

# torch.compile
act_c = torch.compile(expert.act)
step_c = torch.compile(sim.step)

# warmup
sim.reset(true_mask)

with Timer() as tm:
    for _ in range(10):
        _ = act_c()
        _ = step_c(zero_qdot)

# exit runs when the with block ends
warmup = round(tm.ms / 1000, 2)  # seconds

# loop
sim.reset(true_mask)

with Timer() as tm:
    for i in range(T * episodes):
        qdot = act_c()
        obs, done, success = step_c(qdot)
        if done.any():
            expert.reset(done)
            ic(i, done.sum().item(), success[done].float().mean().item())

# exit runs when the with block ends
loop = round(tm.ms / 1000, 2)  # seconds


ic(T, episodes, warmup, loop)