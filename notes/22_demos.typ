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
#let c-cpu = rgb("#6f8fcf")
#let c-flag = rgb("#c9a45c")
#let c-new = rgb("#8f6fb5")
#let nd(pos, c, body, name) = node(pos, text(size: 9pt, body), fill: c.transparentize(75%), stroke: 0.6pt + c, name: name, corner-radius: 3pt, shape: rect)
#show raw: set text(size: 8.5pt)

#align(center, block(fill: fg.transparentize(88%), inset: 10pt, radius: 3pt, text(size: 13pt, raw("demos = buffer.finish(success)"))))

= Step 5a: `finish` returns a `Demos`

== What we decided

- `finish` returns one object, `demos`, not two tensors.
- `Demos` is a small dataclass in `data.py` with two fields: `obs` and `act`.
- Nothing else changes yet. `sample` comes in 5b.

== The picture

#align(center, diagram(
  node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg, spacing: (1.2cm, 1.3cm),
  nd((0, 0), c-cpu, [`Buffer` \ `obs_buf` `(T, N, 7)` \ `act_buf` `(T, N, 3)`], <b>),
  nd((1.7, 0), c-flag, [`finish(success)` \ what you already wrote], <f>),
  nd((3.4, 0), c-new, [`Demos` \ `obs` `(M, T, 7)` \ `act` `(M, T, 3)`], <d>),
  edge(<b>, <f>, "->"), edge(<f>, <d>, "->"),
))

`finish` already builds the two tensors on the right. The only change is the last line: it wraps them in a `Demos` and returns that.

== The spec

`Demos` (a dataclass, like `Sim` and `Buffer`):

#table(
  columns: (2.6cm, 2.6cm, 1fr),
  stroke: 0.5pt + fg, inset: 6pt, align: horizon,
  [*field*], [*shape*], [*notes*],
  [`obs`], [`(M, T, 7)`], [the caller passes it in, so no `init=False`],
  [`act`], [`(M, T, 3)`], [same],
)

$M$ is the number of envs that succeeded. It's not a field, because it's already the first dimension of `obs`.

Two things that apply to any class:
- A dataclass with fields and no methods is fine. It's just a named bundle of the two tensors.
- A class can be defined above or below the one that creates it, because `finish` only looks up the name when it runs. Defining `Demos` first is the safest habit.

== Look at

- `type(demos)`, and `demos.obs.shape`, `demos.act.shape`.
- the asserts you already have in `finish` still pass.
- `run_expert.py`: the old line that unpacked two names from `finish` now gets one.

== Next, in order

- *5b*: `demos.sample(B)` returns random `obs` and a 16-step `chunk`.
- *5c*: the pad mask, then `del buffer` and the memory check.
] // claude:end
