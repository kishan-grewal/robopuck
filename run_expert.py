import torch
import rerun as rr

from sim import Sim
from expert import Expert

sim = Sim()
sim.reset(torch.ones(sim.n, dtype=torch.bool, device=sim.device))
expert = Expert(sim)

rr.init("robopuck")  # rr.init("robopuck", spawn=True)
rr.connect_grpc()  # nothing without ssh
sim.log_scene()
for i in range(600):
    # step sim
    qdot = expert.act()
    sim.step(qdot)
    sim.log(i)
