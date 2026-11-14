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
#let blk(label, c, w: 4.6cm) = box(width: w, height: 0.9cm, fill: c.transparentize(70%), stroke: 0.6pt + c, radius: 2pt, align(center + horizon, text(fill: fg, size: 9pt, label)))
#let strip(..cs) = stack(dir: ltr, spacing: 3pt, ..cs.pos())
#let lab(s) = text(size: 9pt, fill: fg, s)

#align(center, block(fill: fg.transparentize(88%), inset: 10pt, radius: 3pt, text(size: 13pt, raw("torch.cuda.memory_allocated()"))))

= Step 5c: free the buffer

Goal: after `finish`, the `Buffer` has done its job. `Demos` holds its own copies of the data, so the buffer's two big tensors are only taking up GPU memory. Delete the buffer and check that the memory really comes back.

== What we decided

- `del buffer` right after `demos = buffer.finish(success)`.
- Measure with `torch.cuda.memory_allocated()` at three moments and look at the differences.

== The picture

GPU memory in use, at three moments. Each block is one dataset: 375 MiB (the buffer's two tensors, and the `Demos` copy of them, since all envs succeed):

#align(center)[
  #grid(
    columns: (5.4cm, auto), column-gutter: 14pt, row-gutter: 12pt, align: horizon + left,
    lab[`mem0`: after the loop, before `finish`], strip(blk("buffer", c-dim)),
    lab[`mem1`: after `finish`], strip(blk("buffer", c-warn), blk("demos", c-new)),
    lab[`mem2`: after `del buffer`], strip(blk("demos", c-new)),
  )
]

The red block in `mem1` is the one to remove. It is a copy you no longer read.

== The spec

#table(
  columns: (1.6cm, 1fr, 2.2cm),
  stroke: 0.5pt + fg, inset: 6pt, align: horizon,
  [*name*], [*what it is*], [*type*],
  [`mem0`], [`memory_allocated()` after the recording loop, before `finish`], [int, bytes],
  [`mem1`], [the same call, right after `finish`], [int, bytes],
  [`mem2`], [the same call, right after `del buffer`], [int, bytes],
)

`memory_allocated()` returns the bytes held by live tensors on the GPU, as a plain Python int. Divide by `1024 ** 2` for MiB.

What `del buffer` does: a name is a label stuck on an object. `del` removes the label. The object (and its tensors) is freed when no label is left on it. A tensor held inside another object (a field of a dataclass) is a label too.

#note(c-warn)[Trap 1][
`del buffer` frees memory only because `finish` returned *copies*. If `Demos` held views of the buffer's tensors, the views would keep the whole original memory alive and `del buffer` would free nothing. This is why the bool mask in `finish` (a copy) matters.
]

#note(c-warn)[Trap 2][
After `del buffer`, the name is gone. Anything later in `run_expert.py` that reads `buffer` (like `buffer.T` in the final `ic`) fails with a `NameError`. Read what you need from it first, or take it from `demos`.
]

== Look at

#note(c-flag)[Look at][
- `mem1 - mem0` is about 375 MiB: $600 times 16384 times (7 + 3) times 4$ bytes.
- `mem2 - mem1` is about $-375$ MiB, and `mem2` is back near `mem0`.
- `nvidia-smi` does not drop: torch keeps freed memory in a cache for reuse. `torch.cuda.memory_reserved()` stays high while `memory_allocated()` falls.
]

== Next

- standardizing: mean and std of `obs` and `chunk` from the demos, then the MSE policy.
] // claude:end
