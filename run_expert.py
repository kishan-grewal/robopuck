# uv run python run_expert.py

import torch
from sim import Sim
from expert import Expert
from utils import Timer
from data import Buffer
from icecream import ic

# sim, expert, buffer
sim = Sim()
expert = Expert(sim)
buffer = Buffer(T=600, N=sim.n)
episodes = 3

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

# first obs
# obs = torch.cat([self.x, self.v, self.q], dim=-1)
obs = torch.cat([sim.x, sim.v, sim.q], dim=-1)  # (N, 7)

with Timer() as tm:
    for i in range(buffer.T * 1):
        # the buffer needs pairs of CORRESPONDING
        # obs and qdots
        # so in this loop the PREV obs leads to this act_c
        # only functions and classes make their own scope in python
        # with, for, if, while do not
        qdot = act_c()
        buffer.add(i, obs, qdot)

        # new obs defined
        obs, done, success = step_c(qdot)
        if done.any():
            expert.reset(done)
            ic(i, done.sum().item(), success[done].float().mean().item())

# we have our last success here (overwritten each time by the for, and in global scope)
flat_obs, flat_act = buffer.finish(success)
# (T * M, 7) and (T * M, 3)

# exit runs when the with block ends
loop = round(tm.ms / 1000, 2)  # seconds

ic(buffer.T, warmup, loop)
ic(flat_obs[0], flat_act[0])
print('\n')
print("min max", flat_act.min(), flat_act.max())