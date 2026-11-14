import torch
from torch import nn
from data import Demos

class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()

        # found this example online
        # self.register_buffer('running_mean', torch.zeros(num_features))

        self.register_buffer('obs_mean', torch.zeros(7))
        self.register_buffer('obs_std', torch.ones(7))
        self.register_buffer('act_mean', torch.zeros(3))
        self.register_buffer('act_std', torch.ones(3))  
        self.eps = 1e-6

    def fit(self, demos: Demos):
        # mean and std need a tuple input (a shape) to reduce over
        self.obs_mean = torch.mean(demos.obs_dem, (0, 1))
        self.obs_std = torch.std(demos.obs_dem, (0, 1))
        self.act_mean = torch.mean(demos.act_dem, (0, 1))
        self.act_std = torch.std(demos.act_dem, (0, 1))

    def norm_obs(self, obs):
        assert(obs.shape[-1] == 7)
        return (obs - self.obs_mean) / (self.obs_std + self.eps)

    def norm_act(self, act):
        assert(act.shape[-1] == 3)
        return (act - self.act_mean) / (self.act_std + self.eps)

    def unnorm_act(self, act):
        assert(act.shape[-1] == 3)
        return act * (self.act_std + self.eps) + self.act_mean
    