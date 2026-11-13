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
        # .contigious gives a copy unless already contigious

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

        return obs_filt, act_filt

    