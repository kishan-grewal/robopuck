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
= the expert, step 5: push

*Symbols.* $d$ unit vector from puck to waypoint $w$ (step 3), $(N,2)$. $v_"push" = 0.15$ m/s extra speed toward $w$. $v_"track"$ the `capped_prop` velocity from before, aimed at the spot behind the puck. $"dropped"$ the bool per env from step 4.

== The idea

The spot sits 0.08 m behind the puck centre and moves with the puck, because `park_spot` is recomputed from `sim.x` every step. Tracking alone just follows the puck. Adding a constant forward speed makes the hand lean into it: the hand presses the puck, the puck moves, the spot moves, the hand keeps following.

#align(center, diagram(
  node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg, spacing: (3.6cm, 2cm),
  node((0, 0), [`capped_prop` \ $v_"track"$ `(N,3)`], shape: rect),
  edge((0, 0), (1, 0), "->"),
  node((1, 0), [$+$], shape: circle),
  edge((1, 0), (2, 0), "->", [`v_des`]),
  node((2, 0), [`jac`, `dls`], shape: rect),
  node((1, -1), [$v_"push" [d, 0]$ `(N,3)` \ only if dropped], shape: rect),
  edge((1, -1), (1, 0), "->"),
))

== The rule

$ v_"des" = v_"track" + "dropped" dot v_"push" mat(d_x; d_y; 0) $

Push only after the drop: while parking, the hand is in the air and must not drift toward the puck. $"dropped"$ is 0 or 1 per env, so it switches the term on and off. The last entry is 0 because $d$ is flat: the push never changes the height.

== Shapes

#table(
  columns: (5.6cm, 5.6cm, 4.8cm),
  stroke: 0.5pt + fg, inset: 6pt,
  [*API*], [*does*], [*shapes*],
  [#api("torch.zeros_like(d[:, :1])")], [a column of zeros, same length as `d`], [`(N,1)`],
  [#api("torch.cat([d, zero], dim=-1)")], [make the direction 3D], [`(N,2)` + `(N,1)` $arrow.r$ `(N,3)`],
  [#api("dropped[:, None]")], [bool as a column, so each env's row of the push is kept or zeroed (a `(N,)` against `(N,3)` fails)], [`(N,)` $arrow.r$ `(N,1)`],
  [#api("push * dropped[:, None]")], [rows where `dropped` is False become 0 (False acts as 0, True as 1)], [`(N,3)`],
)

== Next edit

One function beside `park_spot`:

```
def push_dir(x, w):   # (N,2), (N,2) -> (N,3)   unit vector, height 0
```

Hint: it is the first line of `park_spot` plus a zero column. In the loop: `v_des = capped_prop(p, p_target) + v_push * push_dir(sim.x, w) * dropped[:, None]`, with `v_push = 0.15`.

== Checks

Puck at $(0.2, 0)$, $w = (0.2, -0.4)$: the push goes along $-y$. Puck $y$ and hand $y$ in metres:

#table(
  columns: (auto, auto, auto, 1fr),
  stroke: 0.5pt + fg, inset: 6pt,
  [*time*], [*puck $y$*], [*hand $y$*], [*gap hand to puck*],
  [2 s], [0], [0.078], [0.078 (parked)],
  [3 s], [-0.135], [-0.063], [0.072],
  [5 s], [-0.333], [-0.261], [0.072],
  [8 s], [-0.728], [-0.655], [0.073],
)

The hand stays about 0.072 m from the puck centre, just touching (contact starts at 0.07). Two things are expected: the puck passes $w$ at $y = -0.4$ and keeps going, because nothing switches the waypoint yet; and there is a small kick at the moment of the drop (puck $y$ about $-0.09$ at 2.4 s), since the hand lands 1 cm inside the tolerance.
] // claude:end
