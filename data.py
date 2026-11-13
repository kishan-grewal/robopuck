import torch
from dataclasses import dataclass, field

# we make a buffer, each row is a pair:
# obs, qdot
# that obs must be what the expert saw when it chose the action
# not the obs that comes back after that action is done

# we only want to keep the episodes that succeed
# obs (N, 7) should only have xvq and never the coin
# we dont want that extra bit of help from the coin info

@dataclass
class Buffer:
    # kwargs:
    T: float
    N: float

    device: str = "cuda"
    # device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # init tensor creation:
    # (T, N, 7)
    obs_buf: torch.Tensor = field(init=False)
    # (T, N, 3)
    act_buf: torch.Tensor = field(init=False)
    # obs, qdot, success are not fields, they are params on the methods

    def __post_init__(self):
        self.obs_buf = torch.zeros(self.T, self.N, 7, device=self.device)
        self.act_buf = torch.zeros(self.T, self.N, 3, device=self.device)

    def add(self, t, obs, qdot):
        self.obs_buf[t] = obs
        self.act_buf[t] = qdot

    def finish(self, success):
        # (T, M, 7)
        obs_filt = self.obs_buf[:, success]
        # (T, M, 3)
        act_filt = self.act_buf[:, success]

        flat_obs = obs_filt.flatten(start_dim=0, end_dim=1)
        flat_act = act_filt.flatten(start_dim=0, end_dim=1)
        return flat_obs, flat_act

    