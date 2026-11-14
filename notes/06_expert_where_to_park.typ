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
= the expert, step 3: where to park the hand

*Symbols.* $x in RR^2$ puck position (`sim.x`). $w in RR^2$ waypoint the puck should go toward. $d$ unit vector from puck to waypoint. $b = 0.08$ m how far behind the puck to park. $z_s = 0.08$ m hover height. $s in RR^3$ the parking spot, the new $p^*$ for `capped_prop`.

== The idea

To push the puck toward $w$, the hand must sit on the opposite side of it. So the target is no longer a constant: it is computed from where the puck _is_, every step.

#align(center, canvas(length: 30cm, {
  import draw: *
  set-style(stroke: 0.6pt + fg)
  circle((0.2, 0), radius: 0.05, fill: fg.transparentize(85%))
  content((0.2, 0), [$x$], anchor: "north-east", padding: 0.008)
  circle((0.3, 0.2), radius: 0.008, fill: fg)
  content((0.3, 0.2), [$w$], anchor: "west", padding: 0.012)
  line((0.2, 0), (0.2 + 0.447 * 0.1, 0.894 * 0.1), mark: (end: ">"), stroke: 0.6pt + user-color)
  content((0.2 + 0.447 * 0.05, 0.894 * 0.05), text(fill: user-color)[$d$], anchor: "west", padding: 0.012)
  line((0.2, 0), (0.164, -0.0716), stroke: (dash: "dashed", paint: fg, thickness: 0.6pt))
  circle((0.164, -0.0716), radius: 0.02, fill: user-color.transparentize(60%))
  content((0.164, -0.0716), [$s$], anchor: "east", padding: 0.03)
  content((0.182, -0.036), [$b$], anchor: "east", padding: 0.008)
}))

Top view. The dashed line is $b$ long and points opposite to $d$.

== The rule

$ d = (w - x) / norm(w - x), quad s = (x - b d, thick z_s) $

$x - b d$ walks from the puck centre backwards along $-d$ by $b$. The last entry $z_s$ is a height, so $s$ is `(N,3)` while $x$ and $w$ are `(N,2)`. The hand arrives at hover height, not on the table: the puck top is 0.03 m and the hand sphere has radius 0.02 m, so 0.08 m clears it with a 3 cm margin (contact only turns on when `pz - r_e < h`). Dropping to the table is a later step.

== Shapes

#table(
  columns: (5.6cm, 5.6cm, 4.8cm),
  stroke: 0.5pt + fg, inset: 6pt,
  [*API*], [*does*], [*shapes*],
  [#api("w - x")], [vector puck to waypoint], [`(N,2)`],
  [#api("d / d.norm(dim=-1, keepdim=True)")], [make it length 1], [`(N,2)` over `(N,1)`],
  [#api("torch.full_like(x[:, :1], z_s)")], [a column of heights, one per env], [`(N,1)`],
  [#api("torch.cat([xy, z], dim=-1)")], [glue the columns into one position], [`(N,2)` + `(N,1)` $arrow.r$ `(N,3)`],
)

== Next edit

One function beside `capped_prop`:

```
def park_spot(x, w, b=0.08, z_s=0.08):   # (N,2), (N,2) -> (N,3)
```

Hint: build the `(N,2)` xy part first, then add the height column. In the loop, replace `example_p_target` with `p_target = park_spot(sim.x, w)`. `w` is a made-up constant for now, e.g. `torch.tensor([0.3, 0.2], device=sim.device).expand(sim.n, 2)`.

== Checks

Puck at $x = (0.2, 0)$, $w = (0.3, 0.2)$, so $d = (0.447, 0.894)$ and $s = (0.164, -0.072, 0.08)$. The hand starts at $(0.660, 0, 0.400)$, so $norm(e) = 0.595$.

#table(
  columns: (auto, auto, auto, 1fr),
  stroke: 0.5pt + fg, inset: 6pt,
  [*time*], [*$norm(e)$*], [*$norm(v_"des")$*], [*hand $p$*],
  [0 s], [0.595], [0.300], [$(0.660, 0, 0.400)$],
  [1 s], [0.300], [0.300], [$(0.415, -0.035, 0.240)$],
  [2 s], [0.023], [0.115], [$(0.183, -0.068, 0.093)$],
  [8 s], [0], [0], [$(0.164, -0.072, 0.080)$],
)

The hand stays above 0.08 m the whole way, so the puck never moves.
] // claude:end
