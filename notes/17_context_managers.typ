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

= a tidy timer: context managers

Your timing block appears twice in `run_expert.py` (warm-up and loop), identical apart from what sits in the middle. Python has a tool for "run some setup, then my code, then some cleanup": the `with` statement. You've already used it: `with profile(...)`, `with torch.no_grad():`, `with open(...)`.

#align(center, diagram(
  node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg, spacing: (0.6cm, 1cm),
  nd((0, 0), c-cpu, [`with timer()` \ `as tm:`], <a>),
  nd((1, 0), c-new, [`__enter__` \ setup], <b>),
  nd((2, 0), c-gpu, [the body \ your code], <c>),
  nd((3, 0), c-new, [`__exit__` \ cleanup], <d>),
  nd((4, 0), c-cpu, [next line \ after the block], <e>),
  edge(<a>, <b>, "->"), edge(<b>, <c>, "->"), edge(<c>, <d>, "->"), edge(<d>, <e>, "->"),
))

The cleanup runs even if the body raises an error. That guarantee is the real reason to use `with` rather than just writing the lines before and after.

*What `with A() as x:` does, in order:*
+ makes the object `A()` and calls its `__enter__()` method. Whatever that returns is stored in `x`.
+ runs the body.
+ calls its `__exit__(exc_type, exc, tb)`. The three arguments are all `None` if the body finished normally; if it raised an error they describe the error. Returning `True` from `__exit__` would hide the error, so normally return `False` or nothing.

#line(length: 100%, stroke: 0.5pt + fg)

== Ways to write one

#table(
  columns: (3.8cm, 1fr, 1fr),
  stroke: 0.5pt + fg, inset: 6pt,
  [*way*], [*shape*], [*good for*],
  [#chip(c-gpu, [A]) a generator with `@contextmanager`], [one function with a single `yield` in the middle. Code above it is the setup, code below it is the cleanup], [short and quick. Nothing to hold on to between the start and the end],
  [#chip(c-new, [B]) a class], [two methods on a class: `__enter__` and `__exit__`], [when the object must keep something, for example a result you read after the block ends],
  [#chip(c-cpu, [C]) ready-made ones], [`contextlib.suppress`, `contextlib.nullcontext`, `contextlib.ExitStack`, `contextlib.redirect_stdout`, and `with a, b:` to use two at once], [common small jobs. Worth knowing they exist],
)

So there are two ways to write your own (A and B), and they do the same thing. A is shorter and B is more flexible.

A toy so you can see the shape (it prints, no timing). Way A:

```
from contextlib import contextmanager

@contextmanager
def announce(name):
    print("start", name)       # setup: runs when the block begins
    try:
        yield                  # the body of the with block runs here
    finally:
        print("done", name)    # cleanup: runs when it ends, error or not

with announce("warm-up"):
    print("inside")            # prints: start warm-up / inside / done warm-up
```

The same thing, way B:

```
class Announce:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        print("start", self.name)
        return self             # this is what `as x` receives
    def __exit__(self, exc_type, exc, tb):
        print("done", self.name)
        return False            # False: don't hide an error

with Announce("warm-up") as a:
    print("inside")
```

In A, `yield something` hands `something` to `as x`. In B, `__enter__` returns it. Without the `try` and `finally` in A, the cleanup is skipped if the body raises.

*Decorators.* A manager written with `@contextmanager` can also be put above a function, `@announce("slow")`, and then it wraps every call of that function. That's the decorator idea. A decorator wraps a function you call. `with` wraps a block in the middle of a script, which is the shape of your timing.

#line(length: 100%, stroke: 0.5pt + fg)

== The timer: what exists, and when

The code is yours to write. This picture is the design: which names exist at each moment of a `with Timer() as tm:` block, and which method creates them. Read it left to right, one row at a time.

#let empty = table.cell(fill: fg.transparentize(90%))[#text(fill: fg.transparentize(50%))[nothing yet]]
#let blank = table.cell(fill: c-new.transparentize(60%))[#text(fill: fg)[an `Event`, not stamped]]
#let stamped = table.cell(fill: c-gpu.transparentize(55%))[#text(fill: fg)[an `Event`, stamped]]
#let number = table.cell(fill: c-flag.transparentize(55%))[#text(fill: fg)[a number, in ms]]
#let either = table.cell(fill: c-new.transparentize(30%).transparentize(50%))[#text(fill: fg)[nothing yet, or a blank `Event` if you make it here. Your choice]]

#table(
  columns: (4.6cm, 1fr, 1fr, 1fr),
  stroke: 0.5pt + fg, inset: 7pt, align: horizon,
  [], [*start*], [*end*], [*ms*],
  [`Timer()` is written: `__init__` runs], either, either, empty,
  [`__enter__` runs], stamped, blank, empty,
  [the body runs (the GPU works)], stamped, blank, empty,
  [`__exit__` runs], stamped, stamped, number,
  [after the block, you read `tm`], stamped, stamped, number,
)

What the picture tells you:
- Each column is a name that has to exist in more than one row. `__enter__` and `__exit__` are two separate functions, and a variable made inside one is gone when the other runs. So anything that must survive from the `__enter__` row to the `__exit__` row has to be stored on the object, which is `self`.
- `as tm` receives whatever `__enter__` returns. For `tm.ms` to work after the block, `tm` must be the object that holds `ms`.
- The `ms` column only fills in on the `__exit__` row, after the end stamp is on the GPU. That is the same reason the `synchronize()` was needed before you read the time in your earlier code.
- Python also passes `__exit__` three error arguments (`exc_type, exc, tb`). They have to be in its signature. You don't use them here.
- `__init__` always runs first, when `Timer()` is evaluated. If you don't write one, Python runs an empty default, so skipping it is allowed. It also takes any options you pass, e.g. `Timer("loop")`. You have none here.
- So the blank events can be made in `__init__` or in `__enter__`. The stamp (`start.record()`) must stay in `__enter__`, because the clock has to start when the block starts, not when the object is built. If you ever reuse one object for two blocks (`t = Timer()`, then `with t:` twice), making the events in `__enter__` gives each block fresh ones.

Pieces you already know: `torch.cuda.Event(enable_timing=True)`, `ev.record()`, `torch.cuda.synchronize()`, `start.elapsed_time(end)` (milliseconds). You call the finished class as `with Timer() as tm:`.

#note(c-flag)[Look at][
- Use it for the warm-up and for the loop in `run_expert.py` (two `with` blocks with different names, e.g. `warm` and `loop`), then delete the two copies of the event code. The printed times should be about what you had: warm-up roughly 2 s, the loop about 1.0 s. If they changed a lot, check which side of `synchronize()` your `record()` calls ended up on.
- Try the toy `announce` with a `raise` inside the body, once with the `try` and `finally` and once without. Which one still prints "done"?
]
] // claude:end
