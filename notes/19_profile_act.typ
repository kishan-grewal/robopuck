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
#let c-gpu = rgb("#4f8f8b")
#let c-warn = rgb("#b5655f")
#let c-flag = rgb("#c9a45c")
#let c-new = rgb("#8f6fb5")
#let chip(c, s) = box(fill: c.transparentize(65%), inset: (x: 4pt, y: 2pt), radius: 2pt, text(fill: fg, s))
#let nd(pos, c, body, name) = node(pos, text(size: 9pt, body), fill: c.transparentize(75%), stroke: 0.6pt + c, name: name, corner-radius: 3pt, shape: rect)
#let note(c, title, body) = block(width: 100%, fill: c.transparentize(85%), stroke: 0.6pt + c, inset: 8pt, radius: 3pt)[#text(weight: "bold", fill: fg)[#title] \ #body]
#show raw: set text(size: 8.5pt)

= Profiling `act`

== What we decided

- Profile `act` only, not `step`.
- Two runs, each warmed up first: eager `expert.act` and `torch.compile(expert.act)`.
- Look only. No changes to `expert.py` or `kin.py` in this step.
- Which file it lives in is yours: a new script in the repo root is the obvious place.

== The idea

`Timer` gave you one number for a whole block. The profiler records every GPU kernel that ran inside a block, with its name and how long it took on the GPU. A _kernel_ is one small program that the GPU runs, e.g. one multiply over a tensor. You get a list, and you read it after the block ends, the same way you read `tm.ms`.

#align(center, diagram(
  node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg, spacing: (0.6cm, 1cm),
  nd((0, 0), c-new, [warm up \ the function], <a>),
  nd((1, 0), c-cpu, [`with profile(...)` \ `as p:`], <b>),
  nd((2, 0), c-gpu, [call `act` \ once, then \ `synchronize()`], <c>),
  nd((3, 0), c-cpu, [block ends: \ `p` holds the \ events], <d>),
  nd((4, 0), c-flag, [keep GPU \ kernels, count, \ add, sort], <e>),
  edge(<a>, <b>, "->"), edge(<b>, <c>, "->"), edge(<c>, <d>, "->"), edge(<d>, <e>, "->"),
))

== The spec

#table(
  columns: (8.4cm, 1fr),
  stroke: 0.5pt + fg, inset: 6pt, align: horizon,
  [*piece*], [*what it does*],
  [`from torch.profiler import profile, ProfilerActivity`], [the recorder and the list of things it can watch],
  [`with profile(activities=[ProfilerActivity.CUDA]) as p:`], [records every GPU kernel launched inside the block. A context manager, like `Timer`],
  [`p.events()`], [list of recorded events. Read it after the block],
  [`e.device_type.name == "CUDA"`], [true for GPU kernels. Other events are CPU-side bookkeeping],
  [`e.device_time`], [microseconds the kernel ran on the GPU],
  [`e.name`], [the kernel's name. Compiled ones start with `triton_` and name what they fused],
  [`profile(..., record_shapes=True)`], [also add `ProfilerActivity.CPU` to `activities`. Then each torch operation records its input shapes: `e.input_shapes`. For when a kernel's name doesn't say where it came from],
  [`p.key_averages(group_by_input_shape=True)`], [a ready-made summary. Call `.table(sort_by="device_time_total", row_limit=10)` on it to print it sorted by GPU time],
)

#note(c-warn)[The traps][
- warm up before the profiler block, or you record the compile.
- keep `torch.cuda.synchronize()` _inside_ the block, after the call. The kernels run in the background, and the profiler stops recording when the block ends.
- one `with profile(...)` block per version (eager, then compiled), so each list is separate.
]

#note(c-flag)[Look at][
- how many GPU kernels in eager, and in compiled. Eager is near the 147 launches we counted before.
- the total GPU time in each, next to the 0.44 ms per call you measured for compiled `act`.
- the single biggest kernel in each: its name, how long, and what share of the total.
- which line of `act` or `dls` produces it. `record_shapes` and the table are for finding this.
]
] // claude:end
