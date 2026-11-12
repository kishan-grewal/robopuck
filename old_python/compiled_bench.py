# uv run python compile_bench.py

import time

import torch

from sim import Sim
from expert import Expert

from icecream import ic

sim = Sim()
# episode starts at -1 (N,)
sim.reset(torch.ones(sim.n, dtype=torch.bool, device=sim.device))
expert = Expert(sim)
qdot = torch.zeros(sim.n, 3, device=sim.device)

# qdot = expert.act()
# obs, done, success = sim.step(qdot)

def eager_compiled(func, compiled=False, warmup_repeats=20, timer_repeats=200):
    foo = torch.compile(func) if compiled else func

    for i in range(warmup_repeats):
        _ = foo()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()

    for i in range(timer_repeats):
        _ = foo()
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / timer_repeats

# # using perf timer instead
# torch.cuda.synchronize()
# t0 = time.perf_counter()
# for i in range(timer_repeats):
#     _ = sim.step(qdot)
# torch.cuda.synchronize()
# t1 = time.perf_counter()
# perf = (t1 - t0) / timer_repeats * 1000

step_qdot = lambda: sim.step(qdot)

eager_step = eager_compiled(step_qdot, compiled=False)
compiled_step = eager_compiled(step_qdot, compiled=True)
eager_expert = eager_compiled(expert.act, compiled=False)
compiled_expert = eager_compiled(expert.act, compiled=True)

def combined():
    _ = expert.act()
    _ = sim.step(qdot)

eager_combined = eager_compiled(combined, compiled=False)
compiled_combined = eager_compiled(combined, compiled=True)

# combined only helps 10%, maybe at the 30% mark consider
# if the cost in debugging ease is worth it to combine functions 
ic(qdot[0].tolist(), eager_step, compiled_step, eager_expert, compiled_expert, eager_combined, compiled_combined)






