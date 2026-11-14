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
= the expert, step 4: park, then drop

*Symbols.* $s_"hi"$ parking spot at hover height ($z_s = 0.08$ m), from step 3. $s_"lo"$ same $(x, y)$ at table height, $z = r_e = 0.02$ m (hand centre one hand-radius above the table). $p$ hand position now. $"tol" = 0.01$ m. $"dropped"$ one bool per env: has this env finished parking?

== The idea

Two phases. Park: aim at $s_"hi"$. Once the hand is within tol of it, drop: aim at $s_"lo"$. The target jumps, so `capped_prop` just drives the hand down.

#align(center, diagram(
  node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg, spacing: (4cm, 2cm),
  node((0, 0), [park \ aim at $s_"hi"$], shape: rect),
  edge((0, 0), (1, 0), "->", [$norm(s_"hi" - p) < "tol"$], label-side: left),
  node((1, 0), [drop \ aim at $s_"lo"$], shape: rect),
))

There is no arrow back yet. Once an env is `dropped` it stays dropped (latched): after the drop, $p$ is far from $s_"hi"$ again, so the test alone would flip the env back to park.

== The rule

$ "dropped" arrow.l "dropped" or (norm(s_"hi" - p) < "tol"), quad p^* = cases(s_"lo" & "if dropped", s_"hi" & "otherwise") $

$or$ is "or": once true, it stays true. $p^*$ goes into `capped_prop` as before. The hand never reaches the puck during the drop: the spot is 0.08 m from the puck centre, and contact needs less than $r_p + r_e = 0.07$ m.

== Shapes

#table(
  columns: (5.6cm, 5.6cm, 4.8cm),
  stroke: 0.5pt + fg, inset: 6pt,
  [*API*], [*does*], [*shapes*],
  [#api("(s_hi - p).norm(dim=-1)")], [distance to the hover spot, per env], [`(N,3)` $arrow.r$ `(N,)`],
  [#api("dropped | (dist < tol)")], [latched flag, `|` is elementwise or on bools], [`(N,)`],
  [#api("torch.where(dropped[:, None], s_lo, s_hi)")], [pick per env; the mask needs `[:, None]` to line up with `(N,3)`], [`(N,3)`],
)

== Next edit

No new function. Before the loop: `dropped = torch.zeros(sim.n, dtype=torch.bool, device=sim.device)`. In the loop, build both spots (`park_spot` already takes `z_s`), update `dropped`, pick `p_target`, then `capped_prop` as before.

== Checks

Puck at $x = (0.2, 0)$, $w = (0.2, -0.4)$, so $d = (0, -1)$ and the spots are $s_"hi" = (0.2, 0.08, 0.08)$, $s_"lo" = (0.2, 0.08, 0.02)$.

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt + fg, inset: 6pt,
  [*time*], [*dropped*], [*hand $p$*],
  [0 s], [no], [$(0.660, 0, 0.400)$],
  [2 s], [no], [$(0.211, 0.078, 0.088)$],
  [2.4 s], [yes], [$(0.201, 0.080, 0.033)$],
  [8 s], [yes], [$(0.200, 0.080, 0.020)$],
)

The puck never moves: its speed stays 0 for the whole run.
] // claude:end
