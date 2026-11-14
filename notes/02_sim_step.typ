#let dark = true
#let fg = if dark { rgb("#e6e6e6") } else { black }
#let bg = if dark { rgb("#1b1b1f") } else { white }
#set page(width: 20cm, height: auto, margin: 2cm, fill: bg)
#set text(size: 12pt)
#set math.mat(delim: "[")
#set math.vec(delim: "[")
#let user-color = if dark { rgb("#7aa2ff") } else { rgb("#1a56db") }
#let claude(body) = text(fill: fg, body)
#set text(fill: user-color)
#import "@preview/lilaq:0.5.0" as lq
#import "@preview/cetz:0.4.2": canvas, draw
#import "@preview/fletcher:0.5.8": diagram, node, edge
#show: lq.set-spine(stroke: fg)
#show: lq.set-tick(stroke: fg)
#show: lq.set-grid(stroke: 0.5pt + (if dark { rgb("#ffffff30") } else { rgb("#00000020") }))
#show: lq.set-legend(fill: bg, stroke: 0.5pt + fg)
#show: lq.set-diagram(fill: bg)

#claude[ // claude:begin
#let api(name) = highlight(fill: if dark { rgb("#2a2f45") } else { rgb("#e6ecff") }, raw(name))

= robopuck: from `Sim(x, v)` to `Sim.step`

*Symbols.* $q in RR^3$ joint angles (yaw, pitch, pitch). $p in RR^3$ end-effector (EE) position, $dot(p)$ its velocity. $x, v in RR^2$ puck. $c_P = (0.4, 0)$ pillar center. $d$ penetration depth. $n$ unit vector from EE to puck. $s$ closing speed. $k = 2000$ spring, $c = 10$ damper. $r_p = 0.05$ puck, $r_e = 0.02$ EE, $r_P = 0.1$ pillar. $L = 0.4$ link. $Delta t_"ctrl" = 0.02$, $Delta t = 0.005$.

== Where `puck.py` is, and the target

You have the constants and `x, v` `(N, 2)`, built in `__post_init__`. Left to add: arm state `q`, EE pose `p`, the step counter, `reset`, forces, `step`, `obs`. Nothing leaves the `Sim` class except `fk`, which stays a plain function so `vmap` can reach it later.

#align(center, diagram(
  node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg, node-inset: 6pt, spacing: (3.6cm, 2.4cm),
  node((0, 0), text(size: 8pt)[*`Sim` fields* \ `x v` `(N,2)` \ `q p` `(N,3)` \ `step_i` `(N,)` int], name: <fields>, stroke: (dash: "dashed", paint: fg, thickness: 0.6pt)),
  node((1, 0), text(size: 8pt)[`reset(mask)`], name: <reset>),
  node((2, 0), text(size: 8pt)[`step(u) -> done` \ `u` `(N,3)`], name: <step>),
  node((0, 1), text(size: 8pt)[`obs() -> (N,7)`], name: <obs>),
  node((1, 1), text(size: 8pt)[`fk(q) -> (N,3)` \ _module-level_], name: <fk>),
  node((2, 1), text(size: 8pt)[`forces(p, pdot) -> (N,2)`], name: <forces>),
  edge(<fields>, <reset>, "<->", [`where`], label-side: left),
  edge(<fields>, <obs>, "->", [`q x v`], label-side: left),
  edge(<reset>, <fk>, "->", [`p = fk(q)`], label-side: left),
  edge(<step>, <fk>, "->", [×2], label-side: right),
  edge(<step>, <forces>, "->", [×4 substeps], label-side: left),
))

== One control step

The only loop is over _time_: 4 substeps per control step. The EE moves at constant velocity within a control step, so substep $i$ uses `lerp(p_old, p_new, (i + 1) / 4)` for its position and the same `pdot` for its velocity.

#align(center, diagram(
  node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg, node-inset: 6pt, spacing: (3.1cm, 2.2cm),
  node((0, 0), text(size: 8pt)[`u.clamp(-2, 2)`], name: <a>),
  node((1, 0), text(size: 8pt)[`p_old = fk(q)` \ `q += 0.02 * u` \ `p_new = fk(q)`], name: <b>),
  node((2, 0), text(size: 8pt)[`pdot = (p_new - p_old)` \ `/ 0.02` `(N,3)`], name: <c>),
  node((2, 1), text(size: 8pt)[×4: `F = forces(...)` \ `v += dt * F / m` \ `x += dt * v`], name: <d>, stroke: (dash: "dashed", paint: fg, thickness: 0.6pt)),
  node((1, 1), text(size: 8pt)[`step_i += 1` \ `done = step_i >= 400`], name: <e>),
  node((0, 1), text(size: 8pt)[`return done` \ caller resets], name: <f>),
  edge(<a>, <b>, "->"), edge(<b>, <c>, "->"), edge(<c>, <d>, "->"), edge(<d>, <e>, "->"), edge(<e>, <f>, "->"),
))

== Fields to add

#table(
  columns: (auto, auto, auto, 1fr),
  stroke: 0.5pt + fg, inset: 6pt,
  [*name*], [*type / shape*], [*init*], [*value or meaning*],
  [`k, c`], [`float`], [default], [2000, 10],
  [`r_p, r_e, puck_h`], [`float`], [default], [0.05, 0.02, 0.03],
  [`r_pillar, T`], [`float`, `int`], [default], [0.1, 400],
  [`L, dt_ctrl, u_max, substeps`], [`float`, `float`, `float`, `int`], [default], [0.4, 0.02, 2.0, 4],
  [`pillar_c`], [`Tensor (2,)`], [`init=False`], [`(0.4, 0.0)`, built in `__post_init__` on `device`],
  [`q, p`], [`Tensor (N,3)`], [`init=False`], [joint angles, EE position],
  [`step_i`], [`Tensor (N,)` `torch.long`], [`init=False`], [episode counter, a tensor not an int],
)

Pick a start pose `q0` yourself: `fk(q0)` must give $p_z > 0.08$ so the EE starts above the puck's top face (0.03) plus the sphere (0.02).

== Pieces, as math

Forward kinematics, $q_2$ measured from vertical, written so it works on `(3,)` and `(N, 3)` alike:
$ rho = L (sin q_2 + sin(q_2 + q_3)), quad z = L + L (cos q_2 + cos(q_2 + q_3)), quad p = vec(rho cos q_1, rho sin q_1, z) $
EE contact on the puck, active only while the sphere's lowest point is under the puck's top:
$ d = max(0, thin r_p + r_e - norm(x - p_"xy")), quad n = (x - p_"xy") / (norm(x - p_"xy") + epsilon), quad s = -(v - dot(p)_"xy") dot n $
$ F_c = max(0, thin k d + c s) thin n thin [p_z - r_e < 0.03] $
Pillar, same spring and no damper, puck center must stay outside $r_P + r_p = 0.15$:
$ F_P = k thin max(0, thin r_P + r_p - norm(x - c_P)) thin (x - c_P) / (norm(x - c_P) + epsilon) $
Friction, as in your loop, then the total:
$ F_f = -mu m g thin v / (norm(v) + epsilon), quad F = F_c + F_P + F_f $

== The pytorch picture of `forces`

The order of computation inside one substep. Every box is a tensor for all $N$ envs at once, with its shape. `p` is the lerped EE position, `pdot` its velocity.

#align(center, diagram(
  node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg, node-inset: 6pt, spacing: (3.2cm, 2.2cm),
  node((0, 0), text(size: 8pt)[`rel = x - p[:, :2]` \ `(N,2)`], name: <rel>),
  node((1, 0), text(size: 8pt)[`dist` `(N,1)` \ `rel.norm(dim=-1,` \ `keepdim=True)`], name: <dist>),
  node((2, 0), text(size: 8pt)[`n = rel / (dist + eps)` \ `(N,2)`], name: <n>),
  node((2, 1), text(size: 8pt)[`depth` `(N,1)` \ `(r_p + r_e - dist)` \ `.clamp(min=0)`], name: <depth>),
  node((1, 1), text(size: 8pt)[`s` `(N,1)` \ `-((v - pdot[:, :2]) * n)` \ `.sum(-1, keepdim=True)`], name: <s>),
  node((0, 1), text(size: 8pt)[`mag` `(N,1)` \ `(k*depth + c*s)` \ `.clamp(min=0)`], name: <mag>),
  node((0, 2), text(size: 8pt)[`active` `(N,1)` \ `p[:, 2:] - r_e < 0.03` \ `F_c = mag * n * active`], name: <act>),
  node((1, 2), text(size: 8pt)[`F = F_c + F_P + F_f` \ `(N,2)`], name: <tot>),
  edge(<rel>, <dist>, "->"), edge(<dist>, <n>, "->"), edge(<n>, <depth>, "->"),
  edge(<depth>, <s>, "->"), edge(<s>, <mag>, "->"), edge(<mag>, <act>, "->"), edge(<act>, <tot>, "->"),
))

`F_P` and `F_f` use the same pattern with `x - pillar_c` (a `(2,)` against `(N,2)` broadcasts with no `[:, None]`) and `v`. Slicing with `p[:, 2:]` keeps the column dim so `(N,1)` broadcasts against `(N,2)`; `p[:, 2]` would give `(N,)` and break it.

== Syntax hints

Marker means a call to look up, nothing more.

#table(
  columns: (4.2cm, 7cm, 4.8cm),
  stroke: 0.5pt + fg, inset: 6pt,
  [*need*], [*reach for*], [*watch out*],
  [uniform in $[-a, a]$, shape `(n, 2)`], [#api("torch.rand(n, 2, device=...)") then `* 2 - 1`, `* a`], [`rand` is $[0, 1)$],
  [per-env choice], [#api("torch.where(mask[:, None], new, old)")], [mask `(N,)` needs `[:, None]` against `(N, 2)` or `(N, 3)`],
  [per-env choice, 1-D state], [#api("torch.where(mask, new, old)")], [`step_i` is `(N,)`: no `[:, None]`],
  [mask times a float tensor], [`float_t * bool_t` works], [bool `(N,1)` or `(N,)` must broadcast],
  [build `(N, 3)` from three `(N,)`], [#api("torch.stack([a, b, c], dim=-1)")], [`dim=-1` so it also works for `(3,)`],
  [`fk` for one or many], [index with #api("q[..., 0]")], [`q[:, 0]` fails on a `(3,)`],
  [`(N, 7)` obs], [#api("torch.cat([q, x, v], dim=-1)")], [all three need the same leading `N`],
  [constant tensor on the right device], [#api("torch.tensor([0.4, 0.0], device=self.device)")], [built in `__post_init__`, not at class level],
  [int tensor counter], [#api("torch.zeros(n, dtype=torch.long, device=...)")], [`step_i += 1` works in place],
  [interpolate], [#api("torch.lerp(a, b, w)")], [`w` can be a float],
  [hold state past a call], [#api(".clone()")], [only matters once compiled],
  [all-true mask for first reset], [#api("torch.ones(n, dtype=torch.bool, device=...)")], [call `reset` once before the first `step`],
  [no `.item()` inside `step`], [keep everything a tensor], [every `.item()` syncs the GPU],
)

== Order to write it, and the check for each

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.5pt + fg, inset: 6pt,
  [*\#*], [*write*], [*print / check*],
  [1], [new fields and `__post_init__`], [`print(sim.q)`, `print(sim.p)`, `print(sim.step_i)`: shapes `(N,3) (N,3) (N,)`],
  [2], [`fk(q)`, module-level], [`fk(zeros(3))` is `(0, 0, 1.2)`; `(N, 3)` in gives `(N, 3)` out],
  [3], [`reset(mask)`: jitter $plus.minus 0.02$ around $(0.2, 0)$, `v = 0`, `q = q0`, `p = fk(q)`, `step_i = 0`], [`x.min(dim=0).values`, `x.max(dim=0).values` inside the box; an all-false `mask` changes nothing],
  [4], [friction-only substep, with `v[:, 0] = 1` after reset], [still stops near 0.17 m, as in stage 0],
  [5], [pillar force], [puck from $x = 0.2$ at 1 m/s: max `x[:, 0]` about 0.256, then it returns; never gets near 0.4],
  [6], [EE contact force], [EE held at table height moving $+x$ at 0.3 m/s into a puck at rest: puck speed climbs to about 0.3, no bounce-back],
  [7], [`step(u)`: clamp, move `q`, `pdot`, 4 substeps, `step_i`, `done`], [random `u` for 1000 steps: shapes constant, no NaN, `x` stays finite],
  [8], [`obs()`], [`(N, 7)`; no column derived from a waypoint or coin],
)

Do 1 to 4 first: they reuse your loop. 5 and 6 are the only new physics; 7 only wires pieces that already pass their checks.
] // claude:end
