# step 1 check, throwaway. run from the repo root: PYTHONPATH=. uv run python old_python/run_expert_check.py

import torch
import rerun as rr

from sim import Sim
from expert import Expert

sim = Sim()
sim.reset(torch.ones(sim.n, dtype=torch.bool, device=sim.device))
expert = Expert(sim)

# rerun
rr.init("robopuck", spawn=True)
sim.log_scene()

# logging
goal = sim.route[-1]  # (3, 2) -> (2,)
# print(goal.shape)
reached = torch.zeros(sim.n, dtype=bool, device=sim.device)
# x = torch.full(), .full gives a prefilled value like a paint bucket
first_step = torch.full((sim.n,), -1, dtype=torch.long, device=sim.device)
closest = torch.full((sim.n,), float('inf'), device=sim.device)
# .full takes (N,) whereas .zeros and .ones take N as a size/shape

# print(first_step.shape, closest.shape)
# exit()
# Traceback (most recent call last):
#   File "/home/kishan/robopuck/run_expert.py", line 22, in <module>
#     first_step = torch.full(sim.n, -1, dtype=torch.long, device=sim.device)
#                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# TypeError: full(): argument 'size' (position 1) must be tuple of ints, not int

for i in range(600):
    # step sim
    qdot = expert.act()
    sim.step(qdot)

    # logging
    # now I remember that -1 removes a dim, None adds one, and ... and : gloss over
    d_goal = torch.norm(sim.x - goal, dim=-1)  # no keepdim ( we want (N,) )
    # (N, 2) - (2,) broadcasts from the right dim (tick) to (N, 2)
    now_reach = (d_goal < 0.03)  # is distance under 3cm at each env
    first_step = torch.where(now_reach & ~reached, i, first_step)  # close now, never close before
    reached = reached | now_reach
    closest = torch.minimum(closest, d_goal)

    # rerun
    sim.log(i)

# logging
arrived = first_step[reached]  # (K,) only the envs that got there
print(f"reached {reached.float().mean():.1%}")
print(f"first step mean {arrived.float().mean():.0f} min {arrived.min()} max {arrived.max()}")
print(f"closest mean {closest.mean():.4f} m")
print(f"final dist mean {d_goal.mean():.4f} m within 3 cm {now_reach.float().mean():.1%}")
print(f"lifts mean {expert.lifts.float().mean():.2f} max {expert.lifts.max()}")