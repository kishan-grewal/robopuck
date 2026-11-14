# uv run python record_demos.py

import torch
from sim import Sim
from expert import Expert
from utils import Timer
from data import Buffer
from icecream import ic

def record():
    # sim, expert, buffer
    sim = Sim()
    expert = Expert(sim)
    buffer = Buffer(T=sim.T, N=sim.n)

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
        for t in range(sim.T):
            # the buffer needs pairs of CORRESPONDING
            # obs and qdots
            # so in this loop the PREV obs leads to this act_c
            # only functions and classes make their own scope in python
            # with, for, if, while do not
            qdot = act_c()
            buffer.add(t, obs, qdot)

            # new obs defined
            obs, done, success = step_c(qdot)
            if done.any():
                expert.reset(done)

    # we have our last success here (overwritten each time by the for, and in global scope)
    # measure memory
    mem0 = torch.cuda.memory_allocated()
    demos = buffer.finish(success)
    mem1 = torch.cuda.memory_allocated()
    del buffer
    mem2 = torch.cuda.memory_allocated()

    memory = {"mem0": mem0, "mem1": mem1, "mem2": mem2}

    # exit runs when the with block ends
    loop = round(tm.ms / 1000, 2)  # seconds

    return demos, sim.T, warmup, loop, memory

def main():
    demos, T, warmup, loop, memory = record()
    ic(T, warmup, loop, memory)
    ic(demos.obs_dem[0, 0], demos.act_dem[0, 0])

# when you run uv run python train.py, you set the __name__ of train.py to __main__
# and any other files run from train.py e.g. import record_demos is set to record_demos
# import file runs the entire file, unless the __name__ != "check"
if __name__ == "__main__":
    main()