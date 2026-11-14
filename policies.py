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

    # used in training, inference
    def norm_obs(self, obs):
        assert(obs.shape[-1] == 7)
        return (obs - self.obs_mean) / (self.obs_std + self.eps)

    # used in training only (for loss)
    def norm_act(self, act):
        assert(act.shape[-1] == 3)
        return (act - self.act_mean) / (self.act_std + self.eps)

    # used in inference only
    def unnorm_act(self, act):
        assert(act.shape[-1] == 3)
        return act * (self.act_std + self.eps) + self.act_mean

class MSEPolicy(nn.Module):
    def __init__(self, hidden_dims=(256, 256)):
        super().__init__()

        self.norm = Normalizer()
        # mse_policy.norm.fit() before training

        # obs in, chunk out
        # (B, 7) --> (B, 16, 3) where B=1 in inferencing time (torch handles the first dim seperately anyways)
        dims = [7, *hidden_dims, 48]
        layers = []

        # somehow dims[1:] doesn't cause an out of index error
        for i, (a, b) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(dims) - 2:  # i_a = 0..len-2
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)


    def forward(self, obs_raw):
        # obs_raw (B, 7) unnormalized
        obs_norm = self.norm.norm_obs(obs_raw)
        out = self.net(obs_norm)  # out (B, 48)
        # (B, 48) -> (unflatten) (B, 16, 3)
        out = torch.unflatten(out, -1, (16, 3))

        return out

    # inference, so prevent forward from building the DAG
    # using this decorator / context manager
    @torch.no_grad()
    def act(self, obs_raw):
        out = self.forward(obs_raw)
        return self.norm.unnorm_act(out)

    def loss(self, obs_raw, chunk, pad):
        pred = self.forward(obs_raw)  # (B, 7) --> (B, 16, 3)
        target = self.norm.norm_act(chunk)  # (B, 16, 3)
        joint_err = (pred - target) ** 2  # (B, 16, 3)
        mean_err = torch.mean(joint_err, dim=-1)  # (B, 16)
        keep = ~pad  # (B, 16)
        # pad = bool mask for keep or not keep in loss
        loss = torch.sum(mean_err * keep) / torch.sum(keep)

        return loss

