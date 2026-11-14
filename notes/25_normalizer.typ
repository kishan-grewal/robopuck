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
#let box-node(pos, label, c, name) = node(pos, text(fill: fg, size: 9pt, label), name: name, fill: c.transparentize(75%), stroke: 0.6pt + c, corner-radius: 3pt, inset: 7pt)

#align(center, block(fill: fg.transparentize(88%), inset: 10pt, radius: 3pt, text(size: 13pt, raw("self.register_buffer(name, tensor)"))))

= Step 6a: the `Normalizer`

Goal: put every number the net sees on a similar scale (about $-1$ to $1$), and convert the net's outputs back into real units. The stats are fixed numbers computed once from the demos. They live in a small `nn.Module` that every policy will own, so they are saved and moved together with the policy.

== What we decided

- `Normalizer` is an `nn.Module`. The stats are *buffers*: tensors that are not trained, but are saved in `state_dict()` and move with `.to(device)`.
- `sample` stays raw. The policy normalizes inside its loss and its inference code, and un-normalizes only at rollout.
- Stats: per feature, over all `M * T` rows of the demos. Four of them: mean and std of `obs`, mean and std of the actions.
- The file is your call (`policies.py` is where the policies will live).

== The picture

Where the `Normalizer` sits. Purple is the part we are building:

#align(center)[
  #diagram(
    spacing: (3.1cm, 1.7cm),
    node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg,
    box-node((0, 0), [`sample`], c-dim, <s>),
    box-node((1, 0), [`norm_obs` \ `norm_act`], c-new, <n1>),
    box-node((2, 0), [net], c-dim, <net1>),
    box-node((3, 0), [loss], c-dim, <loss>),
    edge(<s>, <n1>, "->", text(size: 8pt, fill: fg)[raw `(B,7)`, `(B,16,3)`], label-side: left),
    edge(<n1>, <net1>, "->", text(size: 8pt, fill: fg)[normalized], label-side: left),
    edge(<net1>, <loss>, "->", text(size: 8pt, fill: fg)[normalized], label-side: left),
    box-node((0, 1), [sim obs], c-dim, <o>),
    box-node((1, 1), [`norm_obs`], c-new, <n2>),
    box-node((2, 1), [net], c-dim, <net2>),
    box-node((3, 1), [`unnorm_act`], c-new, <u>),
    edge(<o>, <n2>, "->", text(size: 8pt, fill: fg)[raw `(N,7)`], label-side: left),
    edge(<n2>, <net2>, "->", text(size: 8pt, fill: fg)[normalized], label-side: left),
    edge(<net2>, <u>, "->", text(size: 8pt, fill: fg)[normalized], label-side: left),
  )
]

Top row: training. Bottom row: rollout. Both rows use the same `norm_obs`, from the same object. `unnorm_act` is used only on the bottom row, where real `qdot` goes back to the sim.

== The spec

The buffers, created in `__init__`:

#table(
  columns: (2.4cm, 1.6cm, 1.8cm, 1fr),
  stroke: 0.5pt + fg, inset: 6pt, align: horizon,
  [*name*], [*shape*], [*start*], [*what it is*],
  [`obs_mean`], [`(7,)`], [zeros], [mean of each obs feature],
  [`obs_std`], [`(7,)`], [ones], [std of each obs feature],
  [`act_mean`], [`(3,)`], [zeros], [mean of each joint velocity],
  [`act_std`], [`(3,)`], [ones], [std of each joint velocity],
)

Starting at zeros and ones makes an unfitted `Normalizer` do nothing (subtract 0, divide by 1).

The methods:

#table(
  columns: (3.6cm, 1fr),
  stroke: 0.5pt + fg, inset: 6pt, align: horizon,
  [*signature*], [*what it does*],
  [`fit(self, demos)`], [fill the four buffers from `demos.obs_dem` `(M, T, 7)` and `demos.act_dem` `(M, T, 3)`. Reduce over the first two dims, so each stat is one number per feature],
  [`norm_obs(self, obs)`], [$("obs" - "mean") / ("std" + "eps")$, any shape ending in 7],
  [`norm_act(self, act)`], [the same with the action stats, any shape ending in 3, so `(B, 16, 3)` works],
  [`unnorm_act(self, act)`], [the inverse: `act * (std + eps) + mean`],
)

`eps` is a tiny number like `1e-6`, so a feature with zero spread cannot divide by zero. The stats have shape `(7,)` or `(3,)`, so they broadcast against the last axis with no reshaping.

Background: a Module's `__init__` must call `super().__init__()` first. `register_buffer("name", tensor)` stores the tensor on the module under that name. After that `self.name` reads it, and assigning a new tensor to `self.name` keeps it a buffer.

#note(c-warn)[Trap 1][
`fit` is separate from `__init__` on purpose. To load a saved policy in a fresh script you build the `Normalizer` first and then call `load_state_dict`. If `__init__` needed `demos`, you could not build it without the dataset. So `__init__` makes empty buffers of the right shapes, `fit` fills them, and `load_state_dict` overwrites them.
]

#note(c-warn)[Trap 2][
A plain tensor attribute (`self.mean = ...`) is not a buffer: it is not saved and `.to(device)` does not move it. Only `register_buffer` makes it one. Also, `fit` computes the stats on the demos' device, so build the `Normalizer` on that device too (`.to(device)`).
]

== Look at

#note(c-flag)[Look at][
- `list(norm.parameters())` is empty (nothing to train) but `norm.state_dict().keys()` shows all four names.
- after `fit`, `norm.norm_obs(demos.obs_dem)` has mean about 0 and std about 1 per feature (reduce over dims `(0, 1)`).
- round trip: `unnorm_act(norm_act(x))` equals `x` up to rounding.
- print the four stats once. The obs stds should range from about 0.04 to 0.5, which is the reason to normalize.
]

== Next

- 6b: the MSE policy (`obs` $->$ 48 numbers, reshaped to `(B, 16, 3)`).
] // claude:end
