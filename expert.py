import torch
from dataclasses import dataclass, field

from kin import fk, jac, dls
from sim import Sim


def capped_prop(p, p_target, K=5.0, v_max=0.3):
    # (N, 3) and (N, 3) -> (N, 3)
    e = p_target - p
    e_mag = torch.norm(e, dim=-1, keepdim=True)  # (N, 3) -> (N, 1)
    return K * e * torch.clamp(v_max / (K * e_mag), max=1.0)

def park_spot(x, w, b=0.08, z_s=0.08):  # x + 0.05 is the edge of the puck
    # (N, 2), (N, 2), c -> (N, 3)
    dir = (w - x) / torch.norm(w - x, dim=-1, keepdim=True)
    s_no_z = x - b * dir
    first_col_from_x = x[:, :1]  # for (N, 1) we need :1 not 0
    z = torch.full_like(first_col_from_x, z_s) # const to (N, 1)
    s = torch.cat([s_no_z, z], dim=-1)  # cat (N, 2) with (N, 1)
    return s

def make_routes(side, route):
    # (N, 2) and (3, 2) -> (N, 3, 2)
    # entrywise product, order does not matter
    # we want that 2 dim multiplying the other 2 dim though, and the dims must match
    # the dims will stretch
    # therefore we want (1, 3, 2) * (N, 1, 2)
    return route[None] * side[:, None, :]  # imagine this with two fingers each from your left and right hands


# .expand makes a view (virtual memory)
# first we had a fixed v
# v_des = torch.tensor([0.0, 0.1, 0.0], device=sim.device).expand(sim.n, 3)
# then we had a fixed p_target
# example_p_target = torch.tensor([0.6, 0.3, 0.1], device=sim.device).expand(sim.n, 3)
# now we have a fixed waypoint
# example_w = torch.tensor([0.2, -0.4], device=sim.device).expand(sim.n, 2)
# now we have a list of waypoints from make_route

@dataclass
class Expert:
    sim: Sim
    # parking
    b: float = 0.08
    z_hi: float = 0.08
    z_lo: float = 0.02
    # tol 0.003 (was 0.01): the 1 cm slack let the hand start the drop already touching the puck
    # drift 0.03 (maybe tune): should happen after each waypoint and handle the puck slipping too
    tol: float = 0.003
    drift_max: float = 0.03
    # tracking
    K: float = 5
    v_max: float = 0.3
    # pushing
    v_push: float = 0.15
    # only push once the hand is down at the table (p z is the hand centre, r_e = 0.02, so 0.025 is "landed")
    z_landed: float = 0.025
    # ramp radius 0.03 within that is where it starts to slow down
    ramp: float = 0.03
    lam: float = 0.05

    # (N, 3, 2) waypoints per env
    W: torch.Tensor = field(init=False)
    # (N,) integers (torch.long), which waypoint each env is on
    idx: torch.Tensor = field(init=False)
    # (N,) bool, s_hi to s_lo
    dropped: torch.Tensor = field(init=False)
    # (N,) integers, True -> False changes of dropped
    lifts: torch.Tensor = field(init=False)

    def __post_init__(self):
        sim = self.sim
        self.W = torch.zeros(sim.n, 3, 2, device=sim.device)
        self.idx = torch.zeros(sim.n, dtype=torch.long, device=sim.device)
        self.dropped = torch.zeros(sim.n, dtype=torch.bool, device=sim.device)
        self.lifts = torch.zeros(sim.n, dtype=torch.long, device=sim.device)
        self.reset(torch.ones(sim.n, dtype=torch.bool, device=sim.device))

    def reset(self, mask: torch.Tensor):
        # flip a new coin for the masked envs, start them at waypoint 0, hand up
        sim = self.sim
        coin = (torch.rand(sim.n) < 0.5).to(sim.device)  # (N,) but bool
        sign = torch.where(coin, 1.0, -1.0) # (N,) float
        side = torch.stack([torch.ones(sim.n, device=sim.device), sign], dim=-1)  # (N,) and (N,) -> (N, 2)
        # cat cannot make a new axis, stack can
        self.W = torch.where(mask[:, None, None], make_routes(side, sim.route), self.W)  # (N, 3, 2)
        # we will index W (N, 3, 2) using an arange (N,) and the idx (N,)
        self.idx = torch.where(mask, 0, self.idx)
        self.dropped = self.dropped & ~mask
        self.lifts = torch.where(mask, 0, self.lifts)

    def act(self):
        # one control step: state -> qdot (N, 3), fed to sim.step
        sim = self.sim

        # arange is (N,) and 0 to N-1
        # but idx is (N,) and 0 1 or 2
        # we are indexing with two tensors so only the left two dims of W
        a = torch.arange(sim.n, dtype=torch.long, device=sim.device)
        w = self.W[a, self.idx]  # (N, 3, 2) -> (N, 2)

        # integers, slices (:), None, ... all give views - think of these as slicing or reshaping
        # tensors, bool masks give copies - think of these as querying / filtering / selecting

        # update waypoint
        passed = sim.x[:, 0] > w[:, 0]  # (N,) > (N,) -> (N,) bool
        self.idx = torch.clamp(self.idx + passed.long(), max=2)  # long() makes bool -> int

        # we did end up using fk for both (N, 3) case (p) and (3,) case (Jq)
        p = fk(sim.q, sim.l, sim.base_z)
        Jq = jac(sim.q, sim.l, sim.base_z)  # (N, 3, 3)

        # get tracking vel
        s_hi = park_spot(sim.x, w, b=self.b, z_s=self.z_hi)
        s_lo = park_spot(sim.x, w, b=self.b, z_s=self.z_lo)
        dist = (s_hi - p).norm(dim=-1)  # (N, 3) and (N, 3) -> (N,) (no keepdim true)
        drift = (p[:, :2] - s_lo[:, :2]).norm(dim=-1)  # (N, 2) and (N, 2) -> (N,)

        # initial / up <- -> dropped / needs to re-adjust
        # using OR gate in and AND gate back

        # DROP GATE
        self.dropped = self.dropped | (dist < self.tol)  # gate through into being dropped based on distance (currently dropping)
        # LIFT GATE
        was_dropped = self.dropped.clone()
        self.dropped = self.dropped & (drift < self.drift_max)  # escape gate with an AND instead of an OR, if drift > 0.03
        self.lifts = self.lifts + (was_dropped & ~self.dropped).long()

        parking_spot = torch.where(self.dropped[:, None], s_lo, s_hi)  # (N, 1) and (N, 3) and (N, 3)
        v_track = capped_prop(p, parking_spot, K=self.K, v_max=self.v_max)

        # get push vel
        push_norm = (w - sim.x) / torch.norm(w - sim.x, dim=-1, keepdim=True)  # (N, 2) and (N, 2) -> (N, 1)
        # dropped alone starts the push while the hand is still coming down, which is the initial bump
        # (N,) & (N,) -> (N,), then [:, None] to line up with (N, 2) like before
        landed = self.dropped & (p[:, 2] < self.z_landed)
        v_push = torch.where(landed[:, None], self.v_push * push_norm, torch.zeros_like(push_norm, device=sim.device))
        v_push = torch.cat([v_push, torch.zeros(sim.n, 1, device=sim.device)], dim=-1)

        # was overshooting last waypoint, scale down as a linear capped
        scale_down = torch.clamp((w - sim.x).norm(dim=-1, keepdim=True) / self.ramp, max=1.0)  # (N, 1)
        scale_down = torch.where((self.idx == 2)[:, None], scale_down, torch.ones_like(scale_down))
        v_push = v_push * scale_down

        # get desired vel
        v_des = v_track + v_push

        qdot = dls(Jq, v_des, lam=self.lam)
        return qdot
