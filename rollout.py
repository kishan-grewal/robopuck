import torch
from policies import MSEPolicy
from sim import Sim
from record_demos import record
from icecream import ic
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rerun as rr

checkpoints = Path(__file__).parent / "checkpoints"
device = "cuda"

# load back onto the GPU
try:
    state_dict = torch.load(checkpoints / "mse.pt", map_location="cuda")
except FileNotFoundError:
    print(f"EXCEPTION: Failed to load {checkpoints / "mse.pt"}")
    # else still prints full error traceback

mse_policy = MSEPolicy()
mse_policy.load_state_dict(state_dict)
mse_policy.to(device)

sim = Sim()
true_mask = torch.ones(sim.n, dtype=torch.bool, device=device)
sim.reset(true_mask)

# rerun
# rr.init("robopuck", spawn=True)
# sim.log_scene()

# we could compile policies.py, but we learnt enough

obs_raw = torch.cat([sim.x, sim.v, sim.q], dim=-1)

k = 8
for t in range(0, sim.T, k):
    chunk = mse_policy.act(obs_raw)  # (N, 7) -> (N, 16, 3)
    for act_i in range(k):
        # sim.log(t + act_i, env=0)

        qdot = chunk[:, act_i]  # (N, 3)
        obs_raw, _, success = sim.step(qdot)

ic(k, success.sum() / sim.n)

plots = Path(__file__).parent / "plots"
plots.mkdir(exist_ok=True)  # make the folder, unless it already exists then don't overwrite







