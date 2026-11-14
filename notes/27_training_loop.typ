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
#let c-new = rgb("#8f6fb5")
#let c-warn = rgb("#b5655f")
#let c-flag = rgb("#c9a45c")
#let c-dim = rgb("#6b6b73")
#let note(c, title, body) = block(width: 100%, fill: c.transparentize(85%), stroke: 0.6pt + c, inset: 8pt, radius: 3pt)[#text(weight: "bold", fill: fg)[#title] \ #body]
#show raw: set text(size: 8.5pt)
#let t8(s) = text(size: 8pt, fill: fg, { show raw: set text(size: 7.5pt); s })
#let box-node(pos, label, c, name) = node(pos, text(fill: fg, size: 8.5pt, { show raw: set text(size: 8pt); label }), name: name, shape: rect, fill: c.transparentize(75%), stroke: 0.6pt + c, corner-radius: 3pt, inset: 6pt)

#align(center, block(fill: fg.transparentize(88%), inset: 10pt, radius: 3pt, text(size: 13pt, raw("train.py"))))

= Step 6c: the training loop

== Imports

#table(
  columns: (5.6cm, 1fr),
  stroke: 0.5pt + fg, inset: 5pt, align: horizon,
  [`import torch`], [tensors, `torch.optim`],
  [`from policies import MSEPolicy`], [the policy (it owns the `Normalizer`)],
  [`from icecream import ic`], [printing the loss],
  [`from utils import Timer`], [optional: time the loop],
  [the demos: from `run_expert`], [see the box in diagram 1],
)

== 1. Setup, once

#align(center)[
  #diagram(
    spacing: (0.8cm, 1.5cm),
    node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg,
    box-node((0, 0), [`demos` \ `Demos`], c-dim, <a>),
    box-node((1, 0), [`policy` \ `MSEPolicy()` \ on cuda], c-new, <b>),
    box-node((2, 0), [`policy.norm` \ `.fit(demos)`], c-new, <c>),
    box-node((3, 0), [`opt` \ `AdamW` \ `lr = 3e-4`], c-new, <d>),
    edge(<a>, <b>, "->"),
    edge(<b>, <c>, "->"),
    edge(<c>, <d>, "->"),
  )
]

#align(center, t8[`demos` comes from `run_expert.py`, a script that runs when imported. How it gets here is yours to decide.])

== 2. One step, repeated `n_steps` times

#align(center)[
  #diagram(
    spacing: (0.55cm, 1.6cm),
    node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg,
    box-node((0, 0), [`sample(B)`], c-dim, <s>),
    box-node((1, 0), [`policy.loss`], c-new, <l>),
    box-node((2, 0), [`1  zero_grad`], c-flag, <z>),
    box-node((3, 0), [`2  backward`], c-flag, <bk>),
    box-node((4, 0), [`3  step`], c-flag, <st>),
    edge(<s>, <l>, "->"),
    edge(<l>, <z>, "->"),
    edge(<z>, <bk>, "->"),
    edge(<bk>, <st>, "->"),
    edge(<st>, <s>, "->", bend: 40deg, t8[repeat], label-side: left),
  )
]

#align(center, t8[`sample` gives `obs (B,7)`, `chunk (B,16,3)`, `pad (B,16)`. `policy.loss` gives a scalar. The numbered three run in this order every step.])

== 3. What the optimizer updates

#align(center)[
  #diagram(
    spacing: (1.6cm, 1.2cm),
    node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg,
    box-node((0, 0), [`policy.net` \ weights, biases], c-dim, <w>),
    box-node((0, 1), [`policy.norm` \ 4 buffers], c-new, <n>),
    box-node((2, 0), [`opt`], c-flag, <o>),
    edge(<w>, <o>, "<->", t8[`policy.parameters()`], label-side: left),
    node((2, 1), t8[not in `parameters()`: \ never changed by training], stroke: none),
  )
]

== The few calls

#table(
  columns: (6.4cm, 1fr),
  stroke: 0.5pt + fg, inset: 5pt, align: horizon,
  [`torch.optim.AdamW(params, lr=...)`], [builds `opt`],
  [`opt.zero_grad()`], [clear old gradients],
  [`loss.backward()`], [fill the gradients],
  [`opt.step()`], [move the weights],
)

== Look at

#note(c-flag)[Look at][
- print `loss.item()` every few hundred steps. `.item()` waits on the GPU, so not every step.
- first a sanity run: train on one fixed batch only. the loss should go near 0. if not, something is wrong before the data is to blame.
- then real batches: the loss starts near 1 and falls.
]
] // claude:end
