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

#let api(s) = highlight(fill: rgb("#7aa2ff40"), raw(s))

#claude[ // claude:begin
= the expert, step 6: a coin flip and a route

*Symbols.*
- $sigma in {+1, -1}$: the side an env picked, one per env, `(N,)`.
- `route`: the 3 waypoints of the $+$ side as $(x, y)$ rows, `(3,2)`.
- $W$: every env's own route, `(N,3,2)`: env $i$, waypoint $j$, then $x$ or $y$.

== The idea

The scene is mirror symmetric across the $x$ axis, so each episode flips a coin and goes around the pillar on one side. The route is the same three points for everyone, except the $y$ of the first two flips sign with $sigma$. The goal is on the axis, so it never flips.

#align(center, canvas(length: 20cm, {
  import draw: *
  set-style(stroke: 0.6pt + fg)
  circle((0.4, 0), radius: 0.1, fill: fg.transparentize(85%))
  content((0.4, 0), [pillar])
  circle((0.2, 0), radius: 0.05)
  content((0.2, 0), [puck])
  circle((0.6, 0), radius: 0.015, fill: fg)
  content((0.6, 0), [goal], anchor: "west", padding: 0.025)
  for s in (1, -1) {
    line((0.2, 0), (0.3, 0.2 * s), (0.5, 0.2 * s), (0.6, 0), stroke: (dash: "dashed", paint: user-color, thickness: 0.6pt))
    circle((0.3, 0.2 * s), radius: 0.008, fill: user-color, stroke: none)
    circle((0.5, 0.2 * s), radius: 0.008, fill: user-color, stroke: none)
  }
  content((0.3, 0.2), text(fill: user-color)[$W_0$], anchor: "south-east", padding: 0.012)
  content((0.5, 0.2), text(fill: user-color)[$W_1$], anchor: "south-west", padding: 0.012)
}))

Top view. For now the loop only aims at $W_0$. Moving on to $W_1$ and the goal is the next step.

== The rule

$ W_(i j) = "route"_j dot.o mat(1; sigma_i) $

$dot.o$ is the entrywise product: each entry multiplied by the entry in the same place. The column $mat(1; sigma_i)$ multiplies the $x$ of every waypoint by 1 and the $y$ by $sigma_i$. The coin is $sigma = +1$ with chance one half, else $-1$.

== Shapes

#table(
  columns: (5.6cm, 5.6cm, 4.8cm),
  stroke: 0.5pt + fg, inset: 6pt,
  [*API*], [*does*], [*shapes*],
  [#api("torch.rand(N) < 0.5")], [a coin per env: True with chance one half], [`(N,)` bool],
  [#api("torch.where(coin, 1.0, -1.0)")], [True becomes $+1$, False becomes $-1$], [`(N,)`],
  [#api("torch.stack([ones, side], dim=-1)")], [pair up $(1, sigma)$ per env, for $(x, y)$], [`(N,)` twice $arrow.r$ `(N,2)`],
  [#api("route[None] * sign[:, None, :]")], [a size 1 axis is stretched to match the other: `route` is the same for every env, `sign` is the same for all 3 waypoints], [`(1,3,2)` times `(N,1,2)` $arrow.r$ `(N,3,2)`],
)

== Next edit

One function beside `park_spot`:

```
def make_route(side, route):   # (N,), (3,2) -> (N,3,2)
```

Hint: build `sign` first, then let the product stretch it. Before the loop: `route = torch.tensor([[0.3, 0.2], [0.5, 0.2], [0.6, 0.0]], device=sim.device)` and `side` from the coin. In the loop, replace `example_w` with `W[:, 0]`, where `W = make_route(side, route)` is built once, before the loop.

== Checks

`W[i]` for $sigma_i = -1$ is $[[0.3, -0.2], [0.5, -0.2], [0.6, 0]]$, and about half the envs have each sign. Aiming at $W_0$ for 6 s, the puck ends on its own side, mirrored:

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt + fg, inset: 6pt,
  [*envs*], [*puck $y$ (mean)*], [*puck $y$ has the sign of $sigma$*],
  [$sigma = +1$], [+0.297], [all of them],
  [$sigma = -1$], [-0.297], [all of them],
)

The puck is still pushed past $W_0$, because nothing advances the waypoint yet.
] // claude:end
