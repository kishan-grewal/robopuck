import torch
from dataclasses import dataclass, field
import rerun as rr

# import lovely_tensors as lt
# lt.monkey_patch()  # lt override torch representation

def fk(q: torch.Tensor, L: float, base_z: float):
    # q could be (3,) or (N, 3)
    # unpack q with ... therefore
    q1, q2, q3 = [q[..., i] for i in range(3)]
    q23 = q2 + q3

    # EE_x before yaw
    rho = L * (torch.cos(q2) + torch.cos(q23))
    # EE_z
    z = base_z + L * (torch.sin(q2) + torch.sin(q23))
    # EE_x and EE_y
    px = rho * torch.cos(q1) # accounting for base
    py = rho * torch.sin(q1) # accounting for base

    # px py and z are (N,) or (,)
    return torch.stack([px, py, z], dim=-1)  # (N, 3) or (3,)

@dataclass
class Sim:
    # sim
    n: int = 16384
    dt: float = 0.005
    eps: float = 0.001
    g: float = 9.81
    # puck
    m: float = 0.1
    mu: float = 0.3
    a: float = 0.02
    r_p: float = 0.05
    h: float = 0.03
    # pillar
    r_P: float = 0.10
    k: float = 2000
    c: float = 10
    # arm
    r_e: float = 0.02
    l: float = 0.4
    base_z: float = 0.4
    u_clip: float = 2.0
    dt_ctrl: float = 0.02  # 4 sim steps per control step

    device: str = "cpu"
    # device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # (N, 2)
    x: torch.Tensor = field(init=False)
    # (N, 2)
    v: torch.Tensor = field(init=False)
    # (N, 3)
    q: torch.Tensor = field(init=False) # q1 yaw, q2 q3 pitch

    x0: torch.Tensor = field(init=False)
    q0: torch.Tensor = field(init=False)
    c_P: torch.Tensor = field(init=False)

    def __post_init__(self):
        # puck
        self.x = torch.zeros(self.n, 2, device=self.device)
        self.v = torch.zeros(self.n, 2, device=self.device)
        self.x0 = torch.tensor([0.2, 0.0], device=self.device)
        # pillar
        self.c_P = torch.tensor([0.4, 0.0], device=self.device)
        # arm
        self.q = torch.zeros(self.n, 3, device=self.device)
        self.q0 = torch.tensor([0.0, 0.0, 0.0], device=self.device)

    def step(self, u: torch.Tensor):
        # p = mv -> F deltat = m deltav
        # F_f = -mu mg . v / (|v| + eps)
        # F_P = k * min(0, r - R)

        # arm calcs
        assert(u.shape == (self.n, 3))
        u = torch.clamp(u, min=-self.u_clip, max=self.u_clip)
        p_old = fk(self.q, self.l, self.base_z)
        self.q = self.q + self.dt_ctrl * u
        p_new = fk(self.q, self.l, self.base_z)
        pdot = (p_new - p_old) / self.dt_ctrl

        for i in range(4):
            # friction force
            mag_v = torch.norm(self.v, dim=-1, keepdim=True)
            assert(mag_v.shape == (self.n, 1))
            F_f = -self.mu * self.m * self.g * self.v / (mag_v + self.eps)

            # pillar force
            r = torch.norm(self.x - self.c_P, dim=-1, keepdim=True)  # (N, 2) and (2,) broadcasts to (N, 2) then norm we want (N, 1)
            assert(r.shape == (self.n, 1))
            d_pP = torch.clamp(self.r_p + self.r_P - r, min=0.0)
            normal = (self.x - self.c_P) / (r + self.eps)  # (N, 2) and (N, 1) broadcasts to (N, 2)
            assert(normal.shape == (self.n, 2))
            # F_p = k * min(0, R - r)
            F_p =  self.k * d_pP * normal

            # arm stuff
            p = torch.lerp(p_old, p_new, (i + 1) / 4)
            pxy = p[:, :2]; pz = p[:, 2:]
            pdotxy = pdot[:, :2]
            # x, p, r_p, r_e
            # Q1: are they touching and how much
            dist = torch.norm(self.x - pxy, dim=-1, keepdim=True)  # x and pxy are both (N, 2)
            assert(dist.shape == (self.n, 1))
            d_pe = torch.clamp(self.r_p + self.r_e - dist, min=0.0)
            active_z = (pz - self.r_e < self.h)
            active_xy = (d_pe > 0.0)
            active = (active_z & active_xy).float()  # (N, 1)
            # Q2: which way does the puck get pushed
            n_e = (self.x - pxy) / (dist + self.eps)
            # Q3: how hard does the puck get pushed 
            # spring force and damping force
            s = -((self.v - pdotxy) * n_e).sum(dim=-1, keepdim=True) # (N, 2) product to (N, 2) then sum the 2 dim
            # norm and dot seem to both need dim-1 keepdimTrue]
            F_c = torch.clip(self.k * d_pe + self.c * s, min=0.0) * n_e * active  # (N, 2)

            F_resultant = F_f + F_p + F_c

            self.v = self.v + self.dt * F_resultant / self.m  # F_resultant = ma
            self.x = self.x + self.dt * self.v

    def reset(self, mask: torch.Tensor):
        fresh: torch.Tensor = torch.rand(self.n, 2, device=self.device)
        # random in [0, 1)
        fresh = (fresh * 2 - 1) * self.a + self.x0
        self.x = torch.where(mask[:, None], fresh, self.x)
        self.v = torch.where(mask[:, None], torch.zeros_like(self.v), self.v)     
        self.q = torch.where(mask[:, None], self.q0, self.q)


sim = Sim()

sim.reset(torch.ones(sim.n, dtype=torch.bool, device=sim.device))
# shove to see the spring
sim.v[:, 0] = 1.0
u = torch.zeros(sim.n, 3, device=sim.device)

rr.init("robopuck", spawn=True)
for i in range(400):
    sim.step(u)
    rr.set_time("step", sequence=i)
    rr.log("puck/x", rr.Scalars(sim.x[0, 0].item()))
    rr.log("puck/v", rr.Scalars(sim.v[0, 0].item()))
