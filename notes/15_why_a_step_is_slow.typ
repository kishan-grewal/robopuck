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
#let chip(c, s) = box(fill: c.transparentize(65%), inset: (x: 4pt, y: 2pt), radius: 2pt, text(fill: fg, s))
#let nd(pos, c, body, name) = node(pos, body, fill: c.transparentize(75%), stroke: 0.6pt + c, name: name, corner-radius: 3pt)

= step 3: why a step is slow, and what compile does

You know what a graph is, so this page uses that. The story has three layers, and each one fixes the problem the one before leaves:

+ *eager*: how your code runs today. The GPU waits on Python.
+ *`torch.compile`*: turns your function into a graph of operations and merges neighbours.
+ *CUDA graphs* (`mode="reduce-overhead"`): records the GPU's to-do list once and replays it.

== What I measured on your sim

One control step, eager, on the 4090, 16384 envs:

#table(
  columns: (auto, auto, auto, auto, auto),
  stroke: 0.5pt + fg, inset: 6pt,
  [], [*time per call*], [*GPU kernels launched*], [*GPU actually busy*], [*busy fraction*],
  [`sim.step`], [1.13 ms], [226], [0.30 ms], [27%],
  [`expert.act`], [1.66 ms], [147], [0.50 ms], [30%],
)

A *kernel* is one job handed to the GPU. Your matmul was one kernel. So `step` is 226 jobs in a row, and the GPU spends about 70% of the time doing nothing. The rest of this page is why, and how to fix it.

== Part 1: two workers, one queue

The CPU runs your Python. The GPU runs the maths. They are separate chips. When you write `a * b`, the CPU does not multiply anything. It puts a message in a queue (called a *stream*): "run the multiply kernel on these two tensors". Then it moves straight on to your next line. The GPU takes jobs off the queue and runs them, one after another.

Each message costs the CPU a few microseconds: Python looks up the operation, checks shapes and types, finds the GPU memory, and posts the job. In this sim it is about 5 µs each (1.13 ms ÷ 226 operations).

Now the size of the work. A tensor of shape `(16384, 3)` is about 49 thousand numbers. The 4090 has 16 thousand cores, so one multiply over that finishes in a couple of microseconds. The CPU takes about 5 µs to post the next job, but the GPU needs only about 1 to 2 µs to finish this one. The GPU is faster than the CPU can feed it.

#align(center, canvas({
  import draw: *
  set-style(stroke: none)
  let cpu = c-cpu.transparentize(30%)
  let gpu = c-gpu.transparentize(10%)
  content((-1.1, 1.2), text(size: 10pt, fill: fg)[CPU])
  content((-1.1, 0.3), text(size: 10pt, fill: fg)[GPU])
  content((7.3, 2.3), text(size: 10pt, fill: fg)[*eager*: 8 operations])
  for k in range(8) {
    rect((k * 1.0, 1.0), (k * 1.0 + 0.9, 1.4), fill: cpu)
    rect((k * 1.0 + 0.9, 0.1), (k * 1.0 + 1.15, 0.5), fill: gpu)
  }
  content((4.2, -0.5), text(size: 9pt, fill: fg)[the GPU is idle in the gaps, waiting for the next post])
  // reduce-overhead
  content((7.3, -1.6), text(size: 10pt, fill: fg)[*CUDA graph replay*: same 8 operations])
  rect((0, -2.3), (0.5, -1.9), fill: cpu)
  content((-1.1, -2.1), text(size: 10pt, fill: fg)[CPU])
  content((-1.1, -3.0), text(size: 10pt, fill: fg)[GPU])
  for k in range(8) {
    rect((0.5 + k * 0.25, -3.2), (0.5 + k * 0.25 + 0.25, -2.8), fill: gpu)
  }
  content((3.0, -3.8), text(size: 9pt, fill: fg)[one post, then the GPU runs without gaps])
  line((0, -4.2), (9.6, -4.2), stroke: 0.5pt + fg, mark: (end: ">"))
  content((9.9, -4.2), text(size: 9pt, fill: fg)[time])
}))

Blue is CPU work, teal is GPU work, drawn to the same time scale. Both rows do the same 8 jobs. The top one is as slow as the CPU's posting. The bottom one is as slow as the GPU's real work.

A second cost: every separate kernel reads its inputs from the GPU's main memory and writes its result back there, even if the next kernel reads it straight away. `a * b + c` as three kernels is three trips to memory. As one kernel it is one trip.

*Waiting.* The CPU does not wait for the GPU. Some operations force it to: turning a GPU value into a Python value (`.item()`, `print(x)`, `if x.any():`) cannot be posted and forgotten, because Python needs the answer now. The CPU stops and waits for the whole queue to drain. This is called a _sync_. It is also why timing needs `torch.cuda.Event`: a normal stopwatch measures how fast the CPU posted jobs, not how long the GPU took.

== Part 2: `torch.compile` and the operation graph

Take one line of your friction force, $v / (norm(v) + epsilon)$. Draw each operation as a node and each tensor passed between them as an edge. You get a directed graph with no cycles, a DAG. Note that $v$ has two outgoing edges, which is why it is a graph and not a list:

#align(center, diagram(
  node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg, spacing: (2.6cm, 1.3cm),
  nd((0, 0), fg, [$v$], <v>),
  nd((1, 0), c-gpu, [`norm`], <n>),
  nd((2, 0), c-gpu, [`+ eps`], <e>),
  nd((3, 0), c-gpu, [`divide`], <d>),
  edge(<v>, <n>, "->"),
  edge(<n>, <e>, "->"),
  edge(<e>, <d>, "->"),
  edge(<v>, <d>, "->", bend: -45deg),
))

Eager mode launches one kernel per teal box, so three kernels here, with a memory trip between each. `torch.compile` works in two stages:

#chip(c-cpu, [stage 1: Dynamo]) records the graph. It runs your Python function once, and watches which tensor operations happen. It records those into a graph like the one above. Python itself is not recorded. Loops and plain numbers are worked out while recording: your `for i in range(4)` substep loop becomes four copies of its body in the graph. The graph is saved together with some cheap checks ("were the shapes the same as when I recorded this?"). On the next call, if the checks pass, it skips Python and runs the saved graph.

#chip(c-gpu, [stage 2: Inductor]) takes the graph and merges neighbouring nodes into one kernel where it can. It writes new GPU code for each merged group. The three boxes above become one kernel: the intermediate numbers stay in the GPU's fast on-chip storage and never go back to main memory. That is the "fused" in the logs.

*A graph break* is when Dynamo hits something it cannot record, usually one that needs a sync. For example `if done.any():` means Python must see the answer to choose a branch. Dynamo cuts there. You get: graph A, then plain Python, then graph B.

#align(center, diagram(
  node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg, spacing: (3.4cm, 1.2cm),
  nd((0, 0), c-gpu, [graph A \ (compiled, fused)], <a>),
  nd((1, 0), c-warn, [Python runs \ `if done.any():`], <p>),
  nd((2, 0), c-gpu, [graph B \ (compiled, fused)], <b>),
  edge(<a>, <p>, "->", [sync], label-side: left),
  edge(<p>, <b>, "->"),
))

Merging only happens _inside_ one graph. That answers your question from earlier: two separately compiled functions are two graphs, and nothing merges across the gap between them. The cost is small (a handful of kernels), which is why compiling `step` and `act` separately is fine.

== Part 3: CUDA graphs

This is a different idea at a different layer, and the word "graph" is a coincidence. A *CUDA graph* is also a DAG, but its nodes are GPU kernels, with their exact arguments, and its edges mean "this one must finish first". You build it by recording: you tell CUDA to start capturing, run your code once, and CUDA stores the list of every kernel that was launched instead of just running them. Then to run it again, you make one call, "replay", and the GPU driver runs the whole stored list. The CPU does not post 226 jobs. It posts one. That is the bottom row of the timeline picture.

The price of this is that the recording is frozen, and two consequences follow:

+ *Fixed memory.* The recording stores _where in GPU memory_ each kernel reads and writes, not the tensors' names. So on every replay the inputs must sit at the same addresses, and the results come out at the same addresses too. Torch handles this by giving the graph its own private memory. New inputs get copied into it first. Results live in it and are _overwritten on the next replay_. If you are still holding last step's `obs` after the next replay, it has been silently replaced, and torch raises an error to tell you. The fix is `.clone()` the output you want to keep. This is the thing to watch for with `self.x`, `self.idx` and the other state you reassign every step.
+ *No decisions.* Replay runs the same kernels in the same order every time. Nothing can look at a value and choose another path. That is why Python `if`s on GPU values are not allowed inside the compiled region, and why your code uses `torch.where` for everything (both branches are always computed, and `where` picks).

== Putting the three together

#table(
  columns: (auto, 1fr, 1fr, 1fr),
  stroke: 0.5pt + fg, inset: 6pt,
  [], [*eager*], [*`torch.compile`*], [*`mode="reduce-overhead"`*],
  [*kernels*], [one per operation (226)], [merged groups, far fewer], [same merged groups],
  [*CPU posts*], [one job per kernel], [one job per kernel], [one replay call],
  [*memory trips*], [one per operation], [one per merged group], [one per merged group],
  [*what it needs from you*], [nothing], [no data-dependent Python branches (or breaks happen)], [that, plus: do not hold on to old outputs],
)

== How fast can it get?

There is a floor. Replaying a graph removes the idle gaps, but it cannot go below the time the kernels themselves need, and in the table at the top that is 0.30 ms of the 1.13 ms for `step`. Fusion is what can lower the floor: 226 small kernels with a memory trip each are mostly overhead, so merging them should cut the kernel time too. How far, I don't know. Guess before you run it, then see how close you were.

== To do and look at

+ Wrap `sim.step` with `torch.compile(..., mode="reduce-overhead")` and run it. Where you put the wrap is your call.
+ Time eager and compiled with `torch.cuda.Event` (start event, end event, `torch.cuda.synchronize()`, then `start.elapsed_time(end)` in ms). Discard the first few calls: the first one is compiling.
+ Set `TORCH_LOGS="graph_breaks,recompiles"` in the environment and run again. Silence is good. A `graph_breaks` line names the exact line that cut the graph.
+ Expect an error about CUDA graph outputs being overwritten. When it appears, work out which tensor it is, and why.
] // claude:end
