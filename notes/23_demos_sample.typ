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
#let cell(label, c, w: 1.0cm) = box(width: w, height: 0.8cm, fill: c.transparentize(70%), stroke: 0.6pt + c, radius: 2pt, align(center + horizon, text(fill: fg, size: 9pt, label)))
#let strip(..cs) = stack(dir: ltr, spacing: 3pt, ..cs.pos())
#let lab(s) = text(size: 9pt, fill: fg, s)
#let endbar = box(width: 3pt, height: 0.8cm, fill: fg)
#let stepno(n) = box(fill: c-new.transparentize(60%), stroke: 0.6pt + c-new, radius: 2pt, inset: (x: 6pt, y: 3pt), text(fill: fg, weight: "bold", str(n)))

#align(center, block(fill: fg.transparentize(88%), inset: 10pt, radius: 3pt, text(size: 13pt, raw("obs, chunk, pad = demos.sample(B)"))))

= Step 5b: `Demos.sample` (with the pad mask)

Goal: a batch of `B` training pairs. The net learns `obs (B, 7)` $->$ `chunk (B, 16, 3)`. A sample is one env, one start step, the obs there, and that env's next 16 expert actions, plus a mask saying which of those 16 were padded.

Signature: `sample(self, B, K=16)`, returns `obs, chunk, pad`. Inside, build these in order. Every tensor is long unless noted, on the same device as the data.

#stepno(1) *Sizes.* `M` and `T` are the first two sizes of `obs_dem`. Nothing to store.

#stepno(2) *Pick the samples.* Two random index tensors. Start with shape `(B,)`:
- `env`: random env, $0 .. M - 1$
- `start`: random step, $0 .. T - 1$ (any step may start a chunk)
Keep both as variables. Row `i` is one sample: env `env[i]`, starting at `start[i]`. `(B,)` is all `obs` needs (step 4). The chunk needs a size-1 axis added in step 5, to broadcast against `(B, K)`.

#stepno(3) *Which steps does each chunk cover.* `offsets` is `(K,)`, $0 .. K - 1$. `time = start + offsets` needs `start` as `(B, 1)` to broadcast with `(K,)` to `(B, K)`. Keep this unclamped `(B, K)` tensor in its own variable (name it, the mask needs it in step 6), then clamp a result at $T - 1$ to get `time`. Example with $T = 10$, $K = 4$, start 8: steps 8, 9, 10, 11 don't all exist (the bar is the end of the episode), so 10 and 11 become 9.

#align(center)[
  #grid(
    columns: (5.2cm, auto), column-gutter: 14pt, row-gutter: 12pt, align: horizon + left,
    lab[start 8, before clamping], strip(cell("8", c-new), cell("9", c-new), endbar, cell("10", c-warn), cell("11", c-warn)),
    lab[start 8, clamped at $T - 1$], strip(cell("8", c-new), cell("9", c-new), endbar, cell("9", c-flag), cell("9", c-flag)),
  )
]

#stepno(4) *The input.* `obs` `(B, 7)`, float32: `obs_dem` at `(env, start)`, paired index tensors of shape `(B,)`.

#note(c-warn)[Trap][
The obs comes from `start`, the step you began at. Not from `time`, which has been clamped.
]

#stepno(5) *The target.* `chunk` `(B, K, 3)`, float32: `act_dem` at `(env, time)`. `env` needs shape `(B, 1)` so it broadcasts against `time` `(B, K)`: the same env for all `K` steps of row `i`. The last axis of `act_dem` rides along.

Toy: $M = 4$ envs, $T = 5$ steps, each cell is `(env, step)`. Env 1, start 1, chunk of 3 is the purple run:

#align(center)[
  #grid(
    columns: (1.6cm,) + (1.3cm,) * 5,
    column-gutter: 3pt, row-gutter: 3pt, align: center + horizon,
    [], ..range(5).map(t => lab[step #t]),
    ..range(4).map(e => (
      lab[env #e],
      ..range(5).map(t => cell(str(e) + "," + str(t), if e == 1 and t >= 1 and t <= 3 { c-new } else { c-dim }, w: 1.3cm)),
    )).flatten(),
  )
]

#stepno(6) *The pad mask.* `pad` `(B, K)`, bool: `True` where the unclamped step is past the end ($>= T$), so that entry is a repeat of the last action. In the picture above these are the two yellow cells. It must come from the unclamped tensor: after clamping nothing is past the end any more. The loss will skip `True` entries later. This expert's actions are nearly zero at the end (about 0.006 at step 599), so the mask isn't needed for correctness here. We keep it because real pipelines do, and it costs one line.

#stepno(7) *Return* `obs, chunk, pad`.

== Look at

#note(c-flag)[Look at][
- shapes: `obs` `(B, 7)`, `chunk` `(B, 16, 3)`, `pad` `(B, 16)` bool.
- `chunk[:, 0]` equals `act_dem[env[:, 0], start[:, 0]]`: the first action is the action at the start step.
- force `start` to $T - 1$ for a moment: the chunk should be the last action repeated 16 times, and `pad[:, 0]` all `False`, `pad[:, 1:]` all `True`.
- `pad.float().mean()` over uniform starts is about 1.25% ($0 + 1 + ... + 15$ over $16 times 600$).
]

== Next

- *5c*: `del buffer` and the memory check.
] // claude:end
