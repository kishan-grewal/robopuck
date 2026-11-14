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
= the expert, step 7: move on to the next waypoint

*Symbols.*
- $W$: every env's route, `(N,3,2)`, from step 6.
- `idx`: which waypoint each env is aiming at, $0$, $1$ or $2$, `(N,)` integers.
- $w$: the waypoint it is aiming at now, `(N,2)`. It replaces `W[:, 0]`.
- $x_1$: the puck's $x$ coordinate, `sim.x[:, 0]`.

== The idea

`idx` counts how many waypoints the env has passed. When the puck's $x$ gets past the current waypoint's $x$, `idx` goes up by one. The last waypoint is the goal, so it stops there.

#align(center, diagram(
  node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg, spacing: (4cm, 2cm),
  node((0, 0), [`idx = 0` \ aim at $W_0$], shape: rect),
  edge((0, 0), (1, 0), "->", [$x_1 > 0.3$], label-side: left),
  node((1, 0), [`idx = 1` \ aim at $W_1$], shape: rect),
  edge((1, 0), (2, 0), "->", [$x_1 > 0.5$], label-side: left),
  node((2, 0), [`idx = 2` \ aim at goal], shape: rect),
))

The edge labels are the $x$ of $W_0$ and $W_1$. There is no way back.

== The rule

$ "idx" arrow.l min("idx" + [x_1 > w_x], thin 2) $

$[x_1 > w_x]$ is 1 if the puck has passed the waypoint, else 0 (a comparison turned into a number). $w_x$ is the first entry of $w$.

== Shapes

#table(
  columns: (5.6cm, 5.6cm, 4.8cm),
  stroke: 0.5pt + fg, inset: 6pt,
  [*API*], [*does*], [*shapes*],
  [#api("W[torch.arange(N), idx]")], [pick one waypoint per env: the first index is "env number" (0, 1, 2, ...), the second is that env's own `idx`; they are read in pairs], [`(N,3,2)` $arrow.r$ `(N,2)`],
  [#api("sim.x[:, 0] > w[:, 0]")], [True where the puck has passed], [`(N,)` bool],
  [#api(".long()")], [True becomes 1, False becomes 0, as an integer], [`(N,)`],
  [#api(".clamp(max=2)")], [anything above 2 becomes 2], [`(N,)`],
)

== Next edit

No new function. Before the loop: `idx = torch.zeros(sim.n, dtype=torch.long, device=sim.device)`. At the top of the loop: pick `w` from `W` and `idx`, then update `idx`. Delete the old `w = W[:, 0]`.

== Checks

Across the first 64 envs, over time. Puck $x$ is a mean.

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt + fg, inset: 6pt,
  [*time*], [*mean idx*], [*puck $x$*],
  [3 s], [0], [0.202],
  [5 s], [0.03], [0.287],
  [6 s], [1.00], [0.386],
  [8 s], [2.00], [0.545],
)

Every env reaches `idx = 2` and stays there. At 8 s the puck is still on average 0.145 m short of the goal: the route is long for a 0.1 m/s push. That is a tuning job for later.
] // claude:end
