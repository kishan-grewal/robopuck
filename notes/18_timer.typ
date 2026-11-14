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

= Timer: a `with` block that times the GPU

== What we decided

- `Timer` is a class. The result, `ms`, is stored on the object, so `tm.ms` is readable after the block.
- Used for the warm-up and for the loop in `run_expert.py`, as two separate `with` blocks with different names (e.g. `warm` and `loop`). The two copies of the event code in that file go away.
- Where the class lives is yours: the top of `run_expert.py`, or a small module you import.

== What `with` does

`with Timer() as tm:` does three things, in this order:
+ `Timer()` makes the object. Python runs its `__init__` first (an empty default if you don't write one).
+ calls `__enter__` on it and stores what that returns in `tm`.
+ runs the block, then calls `__exit__`, even if the block raised an error.

`__enter__` and `__exit__` are separate functions, so a local variable made in one is gone in the other. Whatever has to survive from one to the other lives on `self`.

== Picture: what exists, and when

#align(center, diagram(
  node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg, spacing: (0.7cm, 1cm),
  nd((0, 0), c-new, [`__init__` \ object made], <a>),
  nd((1, 0), c-new, [`__enter__` \ start the clock], <b>),
  nd((2, 0), c-gpu, [the block \ GPU works], <c>),
  nd((3, 0), c-new, [`__exit__` \ stop the clock], <d>),
  nd((4, 0), c-cpu, [after: \ read `tm.ms`], <e>),
  edge(<a>, <b>, "->"), edge(<b>, <c>, "->"), edge(<c>, <d>, "->"), edge(<d>, <e>, "->"),
))

An `Event` is a time marker. `ev.record()` puts it into the GPU's queue, and once the GPU reaches it, it holds the time it passed that point. "Recorded" below means `.record()` has been called on it. What each name on `self` holds at each moment:

#let empty = table.cell(fill: fg.transparentize(90%))[#text(fill: fg.transparentize(50%))[nothing yet]]
#let either = table.cell(fill: c-new.transparentize(75%))[#text(fill: fg)[nothing, or a blank `Event` if made here]]
#let blank = table.cell(fill: c-new.transparentize(60%))[#text(fill: fg)[`Event`, not recorded yet]]
#let recorded = table.cell(fill: c-gpu.transparentize(55%))[#text(fill: fg)[`Event`, recorded]]
#let number = table.cell(fill: c-flag.transparentize(55%))[#text(fill: fg)[a number, in ms]]

#table(
  columns: (4.6cm, 1fr, 1fr, 1fr),
  stroke: 0.5pt + fg, inset: 7pt, align: horizon,
  [], [*start*], [*end*], [*ms*],
  [`__init__` runs], either, either, empty,
  [`__enter__` runs], recorded, blank, empty,
  [the block runs], recorded, blank, empty,
  [`__exit__` runs], recorded, recorded, number,
  [after the block], recorded, recorded, number,
)

== The spec

#table(
  columns: (3.3cm, 2.6cm, 1fr, 3.0cm),
  stroke: 0.5pt + fg, inset: 6pt, align: horizon,
  [*method*], [*runs when*], [*job*], [*leaves on `self`*],
  [`__init__(self)`], [`Timer()` is evaluated], [optional. May make the two blank events], [`start`, `end` (if made here)],
  [`__enter__(self)`], [the block begins], [wait until the GPU is idle, then record the start event. Return `self`, so `as tm` gets the object], [`start`, `end` (if not made in `__init__`)],
  [`__exit__(self, exc_type, exc, tb)`], [the block ends], [record the end event, wait until the GPU has reached it, then read the time between the two events], [`ms`],
)

`exc_type, exc, tb` are filled in by Python and must be in the signature. You don't use them here. `__exit__` returns nothing.

Pieces you already know: `torch.cuda.Event(enable_timing=True)`, `ev.record()`, `torch.cuda.synchronize()`, `start.elapsed_time(end)` (milliseconds, a float). The finished class is used as `with Timer() as tm:`, then `tm.ms`.

#note(c-warn)[The trap: two orders matter][
- `start.record()` goes in `__enter__` even if you make the events in `__init__`: the clock must start when the block starts, not when the object is built.
- in `__exit__`, `end.record()` comes first, then `synchronize()`, then `elapsed_time`. Before that `synchronize()` the end event may not have happened yet, so there is no time to read.
]

#note(c-flag)[Look at][
- With `Timer` in `run_expert.py`: warm-up about 2 s, the loop about 1.0 s, as before. `tm.ms` is milliseconds, so divide by 1000 where you print seconds.
- The three `ic` lines still print at steps 599, 1199, 1799 with success 1.0.
]
] // claude:end
