import torch
from dataclasses import dataclass, field

# buffer.finish(success) returns a Demos()

@dataclass
class Demos:
    # kwargs
    obs_dem: torch.Tensor
    act_dem: torch.Tensor
    
    # we read M and T from the tensors, so they aren't fields
    # no init tensor creation or post_init

    def sample(self, B, K=16):
        # for some random (e, t) in the (M, T, 7) obs_dem we have to get 
        # the corresponding qdot, and the next 15 qdots
        # for a chunk length of K=16 which is how much we will train the net to predict
        # from that 16 we can actually use 16 or less in inference, how ever many we want from testing
        # but the net itself would output 16 for every obs fed in at each timestep

        device = self.obs_dem.device
        M = self.obs_dem.shape[0]
        T = self.obs_dem.shape[1]
        # randint high is exclusive

        # B is literally us picking how many of these we want, how many batches do we want to train on
        env = torch.randint(M, (B,), device=device)  # random e
        start = torch.randint(T, (B,), device=device)  # random t

        # (B, 1) + (K,) -> (broadcast) (B, K)
        # kind of like a col vector plus a row vector, becomes a 2d matrix
        offsets = torch.arange(K, device=device)
        time_raw = start[:, None] + offsets
        assert(time_raw.shape == (B, K))

        # pad (B, K) bool
        pad = (time_raw >= T)  # (B, K) -> (B, K)
        # ic(pad.shape)
        time_clamped = torch.clamp(time_raw, max=T-1)

        # we want B observations (B, 7) from obs_dem (M, T, 7)
        # our target is a chunk of qdots (B, K, 3) from act_dem (M, T, 3)
        # our dem tensors are each 2d mats of size (M, T) row=env, col=step
        obs = self.obs_dem[env, start]
        chunk = self.act_dem[env[:, None], time_clamped]  # (B, 1) broadcast (B, K) -> (B, K) pairs picked from (M, T, 3)

        # When you index with two tensors, they're first broadcast against each other, right-aligned, into one common shape.

        return obs, chunk, pad



# we make a buffer, each row is a pair:
# obs, qdot
# that obs must be what the expert saw when it chose the action
# not the obs that comes back after that action is done

# we only want to keep the episodes that succeed
# obs (N, 7) should only have xvq and never the coin
# we dont want that extra bit of help from the coin info

@dataclass
class Buffer:
    # when you scale up the number of envs really high, 
    # T N D is much faster for buffers in the "hot loop"
    # meaning writing to them in the actual loop
    # this is because when you slice with [t] and add the obs / qdot
    # you are selecting a portion of memory
    # but with [:, t] slice (for env dim first) you are selecting ACROSS memory by adding a (N, D) in
    
    # the dim way of looking at tensors is helpful for understanding what to write, but
    # for memory's sake, it comes back to a tensor just being a 1D list of numbers
    # (ALSO in these two lines below each txny is a list of D numbers since these are vectors with feature dim D)
    # TND = t0n0, t0n1, t0n2 | t1n0, t1n1, t1n2
    # NTD = n0t0, n0t1, n0t2 | n1t0, n1t1, n1t2
    # so when you fill in a single timestep with an obs or qdot,
    # but you use NTD you are literally filling while skipping numbers
    # whereas TND just fills in however many together at a time
    # the pipe | is just to show a split for t and n, each is just the same type of 1D list of numbers
    # the dims of a shape therefore go L->R in terms of outermost to innermost in the memory list
    # L->R in terms of indexing
    # R->L in terms of broadcasting

    # a tensor = list of numbers T, shape (A, B, C..), stride (n1, n2, n3..)

    # kwargs:
    T: float
    N: float

    # optional kwargs:
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
        # integers, slices (:), None, ..., transpose, expand all give views - think of these as slicing or reshaping
        # tensor indexing, bool masks, give copies - think of these as querying / filtering / selecting
        # .contiguous gives a copy unless already contiguous

        # (T, N, 7) -> (M, T, 7)
        obs_filt = self.obs_buf.transpose(0, 1)[success]
        # (T, N, 3) -> (M, T, 3)
        act_filt = self.act_buf.transpose(0, 1)[success]

        # transpose -> mask = a view with a new stride then a copy that uses the strides readng order (contig)
        # mask -> transpose = a copy that you then view to change the stride / reading order of (not contig)
        # so we want a contig single copy not two because then our memory usage peaks lower
        # (outside this function's scope all the copies die anyway, the fields still use memory though)

        assert(obs_filt.is_contiguous() == True)
        assert(act_filt.is_contiguous() == True)

        # sum() waits on the GPU
        M = success.sum().item()  # .sum gives a () tensor
        # back to asserts
        assert(obs_filt.shape == (M, self.T, 7))
        assert(act_filt.shape == (M, self.T, 3))

        return Demos(obs_filt, act_filt)

    