import torch
from policies import MSEPolicy
from record_demos import record
## from X import Y, still runs the entire file "X", so the __name__ guard is still relevant
from icecream import ic
from pathlib import Path

demos, _ = record()
mse_policy = MSEPolicy()
device = "cuda"
# move norm to cuda
mse_policy.to(device)
# remember to fit the registered norm buffers
# fit after device move so norm is also on gpu
mse_policy.norm.fit(demos)
opt = torch.optim.Adam(params=mse_policy.parameters(), lr=3e-4)

B = 256
n_steps = 5000

for i in range(n_steps):
    obs_raw, chunk, pad = demos.sample(B)
    loss = mse_policy.loss(obs_raw, chunk, pad)
    opt.zero_grad()  # torch holds a bucket for each weight / bias
    # zero_grad empties the buckets
    # loss.backward() pours gradient into the buckets
    loss.backward()
    opt.step()

    if i % 500 == 0:
        ic(loss.item())

checkpoints = Path(__file__).parent / "checkpoints"
checkpoints.mkdir(exist_ok=True)  # make the folder, unless it already exists then don't overwrite

torch.save(mse_policy.state_dict(), checkpoints / "mse.pt")