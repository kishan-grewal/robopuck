import torch
from torch.profiler import profile, ProfilerActivity
from sim import Sim
from expert import Expert
from icecream import ic

sim = Sim()
sim.reset(torch.ones(sim.n, dtype=torch.bool, device=sim.device))
expert = Expert(sim)
act_c = torch.compile(expert.act)

# eager
def foo(func, n=5):
    for i in range(10):
        _ = func()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as p:
        _ = func()
        torch.cuda.synchronize()

    # gpu
    cuda_kernels = [e for e in p.events() if (e.device_type.name == "CUDA" and e.name[:2] != "##")]
    kernel_times = [e.device_time for e in cuda_kernels]
    biggest_n = sorted(cuda_kernels, key=lambda e: e.device_time, reverse=True)[:n]
    
    count = len(cuda_kernels)
    total = round(sum(kernel_times), 1)
    top = [(e.name[:40], round(e.device_time, 1)) for e in biggest_n]

    # cpu
    shapes = [(e.name, e.input_shapes) for e in p.events() if (e.device_type.name == "CPU" and "mm" in e.name)]

    return shapes, {"count": count, "total": total, "top": top}

eager_shapes, eager = foo(expert.act, n=5)
compiled_shapes, compiled = foo(act_c, n=5)

ic(eager_shapes)
ic(compiled_shapes)
print('\n')
ic(eager)
print('\n')
ic(compiled)







