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
= the expert, step 2: a target position becomes $v_"des"$

*Symbols.* $p in RR^3$ hand position now (`fk(sim.q, ...)`). $p^* in RR^3$ where we want it. $e = p^* - p$ error. $K = 5$ gain (1/s), my pick. $v_max = 0.3$ m/s cap, from the brief. $v_"des"$ wanted hand velocity, input of `dls`.

== The loop it closes

Step 1 took a made-up $v_"des"$. Now $v_"des"$ is computed from where the hand _is_, every control step. That feedback is what makes it a controller.

#align(center, diagram(
  node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg, spacing: (6mm, 9mm),
  node((0, 0), [`sim.q`], shape: rect),
  edge("->"),
  node((1, 0), [`fk` \ `p` `(N,3)`], shape: rect),
  edge("->"),
  node((2, 0), [$e = p^* - p$], shape: rect),
  edge("->"),
  node((3, 0), [capped $K e$ \ `v_des`], shape: rect, fill: user-color.transparentize(70%)),
  edge("->"),
  node((4, 0), [`jac`, `dls` \ `u`], shape: rect),
  edge("->"),
  node((5, 0), [`sim.step`], shape: rect),
  node((2, -1), [$p^*$ `(N,3)`], shape: rect),
  edge((2, -1), (2, 0), "->"),
  edge((5, 0), (5, 1), (0, 1), (0, 0), "->", label: [new `sim.q`], label-side: left),
))

== The rule

$ v_"des" = K e dot min(1, thin v_max / (K norm(e))) $

$norm(e)$ is the length of $e$ (straight-line distance to the target). The $min(1, dots)$ factor is 1 when the target is close, so $v_"des" = K e$: proportional, slowing as it arrives. When the target is far, the factor drops below 1 and shrinks $K e$ to length exactly $v_max$, same direction. `min` is the vectorized `if`.

#let ee = lq.linspace(0, 0.3, num: 200)
#align(center, lq.diagram(
  width: 9cm, height: 4.5cm,
  xlabel: $norm(e)$ + [ (m)], ylabel: $norm(v_"des")$ + [ (m/s)],
  ylim: (0, 1.6),
  lq.plot(ee, ee.map(x => 5 * x), mark: none, stroke: (dash: "dashed"), label: [no cap, $K norm(e)$]),
  lq.plot(ee, ee.map(x => calc.min(5 * x, 0.3)), mark: none, label: [capped]),
))

The kink sits at $norm(e) = v_max slash K = 0.06$ m. Past it the speed is constant, so a far target is approached in a straight line at 0.3 m/s. Without the cap, a target 0.27 m away asks for 1.4 m/s.

== Shapes

#table(
  columns: (5.6cm, 5.6cm, 4.8cm),
  stroke: 0.5pt + fg, inset: 6pt,
  [*API*], [*does*], [*shapes*],
  [#api("e.norm(dim=-1, keepdim=True)")], [length of $e$ per env], [`(N,3)` $arrow.r$ `(N,1)`],
  [#api("v_max / (K * n + eps)")], [the factor inside the `min`; `eps` avoids a divide by zero at $e = 0$], [`(N,1)`],
  [#api(".clamp(max=1.0)")], [the `min(1, ...)`], [`(N,1)`],
  [#api("K * e * scale")], [`(N,3)` times `(N,1)` broadcasts], [`(N,3)`],
)

== Next edit

One function beside `jac` and `dls`:

```
def capped_p(p, p_target, K=5.0, v_max=0.3):   # (N,3), (N,3) -> (N,3)
```

In the loop, replace the constant `v_des` with `p = fk(sim.q, sim.l, sim.base_z)` then `v_des = capped_p(p, p_target)`. `p_target` is a made-up constant for now, like `v_des` was, e.g. `torch.tensor([0.5, 0.2, 0.3]).expand(sim.n, 3)`.

== Checks

Start `q = (0, 0.6, -1.2)`, so $p = (0.660, 0, 0.400)$. Target $p^* = (0.5, 0.2, 0.3)$, so $norm(e) = 0.275$.

#table(
  columns: (auto, auto, auto, 1fr),
  stroke: 0.5pt + fg, inset: 6pt,
  [*time*], [*$norm(e)$*], [*$norm(v_"des")$*], [*hand $p$*],
  [0 s], [0.275], [0.300], [$(0.660, 0, 0.400)$],
  [0.5 s], [0.127], [0.300], [$(0.575, 0.108, 0.346)$],
  [1 s], [0.015], [0.072], [$(0.509, 0.189, 0.305)$],
  [2 s], [0.0001], [0], [$(0.500, 0.200, 0.300)$],
)

$norm(v_"des")$ never exceeds 0.3. The speed is flat while far, then falls off smoothly as the hand arrives, with no overshoot.
] // claude:end
