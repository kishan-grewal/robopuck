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
#show raw: set text(size: 8.5pt)
#let engage(s) = chip(c-gpu, [ENGAGE · #s])
#let touch(s) = chip(c-new, [TOUCH ONCE · #s])
#let case(n, c, title) = block(width: 100%, fill: c.transparentize(78%), stroke: (left: 3pt + c), inset: 9pt, radius: 2pt, above: 1.6em)[#text(size: 14pt, weight: "bold", fill: fg)[Case #n. #title]]
#let note(c, title, body) = block(width: 100%, fill: c.transparentize(85%), stroke: 0.6pt + c, inset: 8pt, radius: 3pt)[#text(weight: "bold", fill: fg)[#title] \ #body]

= step 3: the case of the slow step

#note(c-flag)[The case][
`sim.step` takes about 1.1 ms. The GPU is busy for only 0.30 ms of that. The other 70% is lost somewhere between Python and the GPU. Your job: find where, win it back with `torch.compile`, and work out what each layer of the fix actually buys. Each case ends with something to look at.
]

#note(c-gpu)[Already solved: the bench and the baseline][
`bench_step.py` exists and works. Eager `step` measured *1.106 ms* with CUDA events and *1.080 ms* with `perf_counter`, both with `synchronize()` on each side, so the two clocks agree. That is the number everything below is compared against. Keep the file: from here on you add to it.
]

*Two speeds.* Not everything here deserves the same attention. Some cases are the skill itself: go slowly, write the code, understand each line. Others you only need to touch once: run it, look at the output, move on, and let the background of your head work on it. Every case is marked:

#chip(c-gpu, [ENGAGE]) the next hour: write it, get it right. \
#chip(c-new, [TOUCH ONCE]) run it, look, don't chase it. Come back whenever.

#align(center, diagram(
  node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg, spacing: (0.5cm, 1cm),
  nd((0, 0), c-gpu, text(size: 9pt)[1 \ wrap], <a>),
  nd((1, 0), c-gpu, text(size: 9pt)[2 \ logs], <b>),
  nd((2, 0), c-gpu, text(size: 9pt)[3 \ `act`], <c>),
  nd((3, 0), c-new, text(size: 9pt)[4 \ ladder], <d>),
  nd((4, 0), c-new, text(size: 9pt)[5 \ crime], <e>),
  nd((5, 0), c-new, text(size: 9pt)[6 \ numbers], <f>),
  nd((6, 0), c-flag, text(size: 9pt)[7 \ close], <g>),
  edge(<a>, <b>, "->"), edge(<b>, <c>, "->"), edge(<c>, <d>, "->"),
  edge(<d>, <e>, "->"), edge(<e>, <f>, "->"), edge(<f>, <g>, "->"),
))

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt + fg, inset: 6pt,
  [*case*], [*speed*], [*why*],
  [1 wrap `step`], [#chip(c-gpu, [engage])], [the core skill: one line gives about 10x, and you decide where the code lives],
  [2 logs], [#chip(c-gpu, [engage])], [how you know a compile is healthy. The planted crime inside it is touch-once],
  [3 `act`], [#chip(c-gpu, [engage])], [doing it a second time on your own is what makes it stick. The combined version is touch-once],
  [4 the ladder], [#chip(c-new, [touch once])], [the idea behind `reduce-overhead`: read it, change one argument, see what happens],
  [5 the crime scene], [#chip(c-new, [touch once])], [see the "overwritten" error once, apply the one-line flag, move on],
  [6 the numbers], [#chip(c-new, [touch once])], [the profiler: run it, read the kernel names],
  [7 close], [#chip(c-flag, [engage])], [five minutes: commit and write down what you chose],
)

Colours on this page: #chip(c-cpu, [blue]) CPU and Python side, #chip(c-gpu, [teal]) GPU and compiled side, #chip(c-warn, [rose]) trouble, #chip(c-flag, [amber]) things to look at, #chip(c-new, [purple]) new ideas. If a word here is new (kernel, stream, Dynamo, Inductor, CUDA graph), the earlier explainer is saved in `archive/old_typ/step3_how_it_works.typ`. Or ask.


#case(1, c-gpu)[Wrap `step`]

#engage[about 20 min]

`torch.compile(f)` takes a function and returns a faster version of the same function. Your timing block in `bench_step.py` stays as it is; only the function it calls changes. Which function, and where you put the call, is your decision. First, what ends up inside and outside:

#align(center, diagram(
  node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg, spacing: (0.45cm, 1.5cm),
  nd((0, 0), c-cpu, [the loop: \ `act`, `ic`, \ timing, \ `expert.reset`, \ `if done.any()`], <loop>),
  nd((1, 0), c-gpu, [`step` \ `(qdot)`], <s1>),
  nd((2, 0), c-gpu, [4 substeps: \ friction, \ pillar, \ contact], <s2>),
  nd((3, 0), c-gpu, [`done`, \ `success`], <s3>),
  nd((4, 0), c-gpu, [`reset` \ `(done)`], <s4>),
  nd((5, 0), c-gpu, [`obs`], <s5>),
  node(enclose: (<s1>, <s2>, <s3>, <s4>, <s5>), stroke: (dash: "dashed", paint: c-gpu), fill: none, snap: false, inset: 10pt),
  edge(<loop>, <s1>, "->"),
  edge(<s1>, <s2>, "->"), edge(<s2>, <s3>, "->"), edge(<s3>, <s4>, "->"), edge(<s4>, <s5>, "->"),
  edge(<s5>, <loop>, "->", bend: -40deg, label-side: left, label-pos: 50%),
))

The arrow out of the loop carries `qdot`. The one back carries `obs, done, success`. The dashed box is what Dynamo records.

Everything inside the dashed box is compiled. The methods `step` calls (`friction_force`, `pillar_force`, `ee_force`, `reset`) are pulled in automatically. You don't wrap them one by one. Everything outside stays ordinary Python. That is why the `if done.any()` in your loop is fine: it sits outside.

*Four places to put the wrap:*

#table(
  columns: (3.6cm, 1fr, 1fr),
  stroke: 0.5pt + fg, inset: 6pt,
  [*choice*], [*what it means*], [*cost*],
  [#chip(c-gpu, [A]) in the script, a new name], [you make a compiled copy of `sim.step` under a new name and call that. `sim.py` is untouched], [none. Eager and compiled sit side by side in one file. *Start here.*],
  [#chip(c-flag, [B]) in the script, replacing the method], [the compiled version takes the place of `sim.step` on that object], [anyone calling `sim.step` now gets the compiled one, and you lose the eager one. Hard to compare],
  [#chip(c-new, [C]) decorator in `sim.py`], [a line above `def step` in the class compiles it for every `Sim`], [compiled forever. Good as a final state, bad for experiments, since you can't turn it off from outside],
  [#chip(c-warn, [D]) rewrite as a function of tensors], [state goes in as arguments and the new state comes out. No hidden state on `self`], [the cleanest for compile, and the end of the "overwritten output" problem in Case 5. But it's a rewrite of `Sim`. Not now],
)

The four spellings:

```
A   step_c = torch.compile(sim.step)          # in bench_step.py, then call step_c(...)
B   sim.step = torch.compile(sim.step)        # replaces the method on that object
C   @torch.compile                            # on the line above def step in class Sim
D   x, v, q, t, ... = step(x, v, q, t, ..., qdot)   # a plain function, no self
```

A bound method is tied to its object, so `step_c` and `sim.step` act on the same `sim`. For timing that's fine: one `Sim` is enough, and `qdot` is zeros so the cost per step doesn't depend on the state. You need *two* `Sim` objects only when comparing results (e.g. the same 300 steps from the same start, then checking `x`, `t`, `episode` agree), because otherwise the two runs advance the same state.

#note(c-flag)[Look at][
- The first call takes seconds. Time it by itself with a plain stopwatch and write it down.
- After warm-up, the timed number. Is it smaller than eager? By how much?
- Does anything print? (Case 2 tells you how to ask the compiler properly.)
- Run a few hundred steps and print `t.unique()` and `episode.unique()` at the end for the compiled sim. Does it still behave like the eager one?
]


#case(2, c-gpu)[Ask the compiler what it saw]

#engage[about 10 min: the logs, and reading silence]

By default `torch.compile` is quiet. You turn on its notes with an environment variable. An _environment variable_ is a setting you put in front of a command, and the program reads it when it starts:

```
TORCH_LOGS="graph_breaks,recompiles" uv run python bench_step.py
```

#table(
  columns: (auto, 1fr),
  stroke: 0.5pt + fg, inset: 6pt,
  [*value*], [*what it prints*],
  [`graph_breaks`], [every place Dynamo had to cut the graph, with the reason and the line of your code],
  [`recompiles`], [every time a compiled function was thrown away and compiled again, with the check that failed (for example, a shape changed)],
  [`output_code`], [the code Inductor generated for each compiled graph. Long. Worth reading once to see the fused kernels],
  [`cudagraphs`], [what the CUDA graph machinery is doing: recording, replaying],
  [`TORCH_LOGS_OUT=file.txt`], [a separate variable: send the logs to a file instead of the terminal],
)

Separate values with commas, no spaces. You can also turn them on from inside Python with `torch._logging.set_logs(graph_breaks=True, recompiles=True)`.

*Silence is a result.* If you turn on `graph_breaks,recompiles` and nothing prints, then your function compiled as one piece and was never recompiled.

*Plant a crime* #touch[optional, 10 min]. To see what a graph break looks like, cause one on purpose. In the script (not in `sim.py`), write a throwaway function that calls `sim.step`, then does `if done.any():` with something trivial inside, then returns. Compile _that_ and run it with the logs on. This is what it looks like (trimmed):

```
Graph break in user code at bench_step.py:10
Graph Break Reason: Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`).
               Dynamo does not support tracing dynamic control flow.
  Hint: ... consider making the code branchless by always applying the transform.
        Operations like torch.clamp, torch.nan_to_num, and torch.where ...
User code traceback:
    if done.any():
```

The log names the exact line and the reason. It is also where the answer lives: `torch.where` instead of `if`, which is what your `Sim.step` already does.

#note(c-flag)[Look at][
Time the planted-crime version against the clean compiled one. The graph is now two pieces with Python in the middle. What happened to the number?
]


#case(3, c-cpu)[The other suspect: `act`]

#engage[about 20 min, the separate version]

`step` is only half of a control step. `expert.act()` took 1.66 ms and launched 147 kernels, more than `step` when we started. So do for `act` exactly what you did for `step`: wrap it in default mode (no `mode=`), warm it up, time it, and run it once with `TORCH_LOGS="graph_breaks,recompiles"` to check for silence. `expert.reset(done)` stays outside, as before.

Two ways to put the pieces together:

#align(center, diagram(
  node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg, spacing: (1.0cm, 1.5cm),
  nd((0, 0), c-gpu, [*separate* \ compile `act` \ compile `step`], <p1>),
  nd((1, 0), c-new, [*combined* \ one function: \ `act`, then `step`], <p2>),
  nd((0, 1), fg, [two graphs, \ `qdot` crosses the gap \ simple to debug], <q1>),
  nd((1, 1), fg, [one graph, one fused region \ more to write, \ errors are harder to place], <q2>),
  edge(<p1>, <q1>, "->"), edge(<p2>, <q2>, "->"),
))

Do the left one now. The right one is #touch[10 min, after the left works]: a small function in `run_expert.py` that calls `expert.act()` then `sim.step(qdot)`, compiled as a single unit. Write it once, time it against the separate version, and see whether fusing across the `qdot` handoff bought anything. I haven't measured that one, so your number is new.

`Expert` also reassigns its own state every call: `self.idx`, `self.dropped`, `self.lifts`, `self.W`. If you put `act` on the `reduce-overhead` rung later (Cases 4 and 5), these are the same kind of suspect as `self.x` was.

Two parts of `act` to look at first if anything surprises you: `jac` (it's `vmap(jacrev(fk))`) and `dls` (it solves a small linear system per env with `torch.linalg.solve`). Neither is data-dependent branching, but they are heavier than anything in `step`.

#note(c-flag)[Look at][
Time a full control step (`act` then `step`, the loop body) in these setups: both eager, only `step` compiled, both compiled separately, and (touch once) both as one function. Which of the two is the bigger share now? Check the end state again (`t`, `episode`).
]


#case(4, c-new)[The ladder: one rung at a time]

#touch[read, then change one argument]

You asked: first without `mode="reduce-overhead"`, or straight to it? *First without.* Each rung adds one good thing and one new way to fail, and you want to know which rung broke:

#align(center, diagram(
  node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg, spacing: (1.7cm, 2.0cm),
  nd((0, 0), c-warn, [*eager* \ \ buys: nothing \ breaks: nothing \ cost: slow, 226 kernels], <r1>),
  nd((1, 0), c-gpu, [*`torch.compile`* \ default mode \ \ buys: fused kernels \ breaks: graph breaks, \ recompiles], <r2>),
  nd((2, 0), c-new, [*`+ reduce-overhead`* \ \ buys: one replay call \ breaks: overwritten \ outputs, no `if`s], <r3>),
  edge(<r1>, <r2>, "->", [wrap], label-side: left),
  edge(<r2>, <r3>, "->", [add `mode=`], label-side: left),
))

Three reasons to go in order:
+ If something fails on rung 2, it's a compile problem. If it fails on rung 3 only, it's a memory problem. Jump straight to rung 3 and you can't tell which.
+ You get three numbers that say what each rung bought: eager, then fused, then replayed.
+ Rung 2 might already be most of the win. Then rung 3 has to earn its place.

Adding the mode is one argument: `torch.compile(f, mode="reduce-overhead")`. (There's also `mode="max-autotune"`: slower to compile, tries more kernel variants. Ignore it for now.)

Do rung 2 fully (Cases 1 to 3) before you touch rung 3.

#note(c-flag)[Look at][
Print the three timings in one table at the end of the script: eager, default, reduce-overhead. Rung 3 will not run as is. Go to Case 5.
]


#case(5, c-warn)[The crime scene: a tensor that was overwritten]

#touch[reproduce the error, apply fix 3]

Add `mode="reduce-overhead"` to your wrap and run a few calls. Within two or three calls you get this (trimmed; your line numbers are from your own file):

```
RuntimeError: Error: accessing tensor output of CUDAGraphs that has been
overwritten by a subsequent run. Stack trace:
  File "sim.py", line 140, in step
    self.reset(done)
  File "sim.py", line 156, in reset
    self.t = torch.where(mask, 0, self.t)
. To prevent overwriting, call torch.compiler.cudagraph_mark_step_begin()
before each model invocation. If a tensor is a user-visible output that must
remain live across generations, manually clone the tensor outside
torch.compile(), or set
torch._inductor.config.triton.cudagraph_trees_generation_cloning = 'user_visible'
```

*Reading the evidence.* The stack trace points at the line that _created_ the tensor that later went bad (`self.t = torch.where(...)` in `reset`). That's the victim, not the culprit. The culprit is a later call that tried to use it.

*The mechanism.* A CUDA graph records memory addresses. So its results are written into a private pool of memory that the graph owns and reuses on every replay:

#align(center, canvas({
  import draw: *
  set-style(stroke: none)
  content((-1.3, 2.7), anchor: "west", text(size: 9pt, fill: fg)[time #sym.arrow.r])
  rect((0, 1.9), (3.2, 2.5), fill: c-gpu.transparentize(55%))
  content((1.6, 2.2), text(size: 10pt, fill: fg)[replay 1])
  rect((5.0, 1.9), (8.2, 2.5), fill: c-gpu.transparentize(55%))
  content((6.6, 2.2), text(size: 10pt, fill: fg)[replay 2])
  rect((0.4, 0), (7.8, 0.7), fill: c-new.transparentize(70%), stroke: 0.6pt + c-new)
  content((4.1, 0.35), text(size: 9pt, fill: fg)[the graph's private memory pool, slot for `t`])
  line((2.4, 1.9), (2.4, 0.7), stroke: 0.8pt + fg, mark: (end: ">"))
  content((2.55, 1.3), anchor: "west", text(size: 9pt, fill: fg)[writes `t`])
  line((5.6, 0.7), (5.6, 1.9), stroke: 0.8pt + fg, mark: (end: ">"))
  content((5.45, 1.3), anchor: "east", text(size: 9pt, fill: fg)[reads `self.t`])
  line((7.4, 1.9), (7.4, 0.7), stroke: 0.8pt + c-warn, mark: (end: ">"))
  content((7.55, 1.3), anchor: "west", text(size: 9pt, fill: c-warn)[writes the new `t` \ into the same pool])
  content((4.1, -0.7), text(size: 9pt, fill: fg)[`self.t` is a Python name pointing into the pool])
}))

Between replays, `self.t` is a normal-looking tensor. But it lives in the pool, and the pool belongs to whichever replay runs next. Torch notices that you're using an output from a previous generation of the graph, and refuses, because it can no longer promise that the numbers are still what you stored.

*The suspects.* Which tensors does `step` reassign every call and read again on the next one?

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt + fg, inset: 6pt,
  [*tensor*], [*outlives the call?*], [*notes*],
  [`self.x`, `self.v`, `self.q`], [yes], [the sim's state: output of step $k$ is input of step $k + 1$],
  [`self.t`, `self.episode`], [yes], [counters: same story. The error named `t` first],
  [`obs`, `done`, `success`], [returned], [used right away by the loop, and not needed after the next call. Different risk],
)

*Four fixes, and what each does:*

#table(
  columns: (3.4cm, 1fr, 1fr),
  stroke: 0.5pt + fg, inset: 6pt,
  [*fix*], [*what it does*], [*the catch*],
  [#chip(c-flag, [1]) mark the start of each step], [tells torch "a new iteration starts, old outputs are dead"], [the error message suggests it, but your state is carried _across_ iterations. Try it and watch what happens],
  [#chip(c-gpu, [2]) copy the state after each call], [a copy lives outside the pool, so the pool can be reused freely], [five extra copies per step. Works. You choose where: in the loop, or in a small wrapper around the compiled function],
  [#chip(c-cpu, [3]) the config flag from the error message], [makes torch do the cloning for you], [one line, set before the first call. Torch decides what counts as user-visible],
  [#chip(c-new, [4]) update the state in place], [the tensors you keep are never graph outputs], [rewrites parts of `step` and `reset`. It's the real fix, and it's close to choice D from Case 1],
)

The spellings:

```
1   torch.compiler.cudagraph_mark_step_begin()     # call before each step
2   t = t.clone()                                  # each state tensor, after the call
3   torch._inductor.config.triton.cudagraph_trees_generation_cloning = "user_visible"
4   self.t.copy_(new_t)                            # instead of  self.t = new_t
```

#note(c-flag)[Look at][
Touch it once: reproduce the error, then apply fix 3 (the one-line flag). Does it run? Is the end state right (`t.unique()` all zeros, `episode.unique()` equal to `[3]` after three episodes)? What is the timing next to your eager 1.1 ms? Fixes 1 and 2 are there if you want to try them. Fix 4 is for later.
]


#case(6, c-gpu)[Interrogate the numbers]

#touch[run the snippet, read the kernel names]

You now have timings for all three rungs. The detective question: *where does the time go now?*

Tools for counting kernels and the time they take:

#table(
  columns: (8.4cm, 1fr),
  stroke: 0.5pt + fg, inset: 6pt,
  [`from torch.profiler import profile, ProfilerActivity`], [the profiler],
  [`with profile(activities=[ProfilerActivity.CUDA]) as p:`], [everything the GPU runs inside the block is recorded. Call `step` once inside it, then `torch.cuda.synchronize()`],
  [`p.events()`], [a list of everything recorded],
  [`e.device_type.name == "CUDA"`], [keeps only the GPU kernels],
  [`e.device_time`], [microseconds the kernel took on the GPU],
  [`e.name`], [the kernel's name. Compiled ones start with `triton_`, and say what they fuse],
)

Count the kernels and add their durations for eager, and then for compiled in default mode. Do this inside the profiler block with a warmed-up function, so you aren't profiling the compile.

*Questions the evidence should answer:*
+ How many kernels now, compared with 226? What do the names of a few say?
+ The kernels' total time versus the timed number per call. If they differ by a lot, the GPU is idle again. Whose fault is it this time? (Remember what a timer without `synchronize()` measures: the CPU's side.)
+ Is `reduce-overhead` faster than the default mode here? Include the cost of whatever fix you chose in Case 5. If not, what does that say about the CUDA graph's job?
+ Keep or drop `reduce-overhead`? Your call, with your numbers.


#case(7, c-flag)[Close the case]

#engage[about 5 min]

What to leave in the repo:
- `bench_step.py` with the timings, so the numbers can be reproduced.
- Whatever wrap you decided on, in a place where `run_expert.py` (and later the demo collection) uses it.
- A line in `CLAUDE.md` for the future: what you chose and why, plus the numbers.

When it works, commit. Then the next job is the demo buffer: run the compiled loop for 600 steps, record `obs` and `qdot` at every step into big GPU tensors, and drop the failed envs.


#pagebreak()

#case("X", c-new)[Sealed envelope: my notes]

Open after you have your own numbers. All measured on this machine (RTX 4090, torch 2.14.1, 16384 envs, `step` alone with a fixed `qdot`, 200 timed calls after warm-up). Yours will differ a little.

#align(center, canvas({
  import draw: *
  set-style(stroke: none)
  let bars = (
    ("eager", 1.13, c-warn),
    ("compile, default mode", 0.10, c-gpu),
    ("compile + reduce-overhead + state cloned", 0.106, c-cpu),
  )
  for (k, (name, v, c)) in bars.enumerate() {
    let y = -k * 1.25
    content((0, y + 0.55), anchor: "west", text(size: 9pt, fill: fg)[#name])
    rect((0, y - 0.25), (v * 10, y + 0.25), fill: c.transparentize(25%))
    content((v * 10 + 0.15, y), anchor: "west", text(size: 9pt, fill: fg)[#v ms])
  }
  content((7, -4.0), text(size: 9pt, fill: fg)[milliseconds per `step` call, same scale])
}))

#table(
  columns: (1fr, auto, auto),
  stroke: 0.5pt + fg, inset: 6pt,
  [], [*kernels per call*], [*kernel time*],
  [eager `step`], [226], [0.30 ms],
  [compiled, default], [18], [0.027 ms],
)

- *Default compile worked first time.* No graph breaks, no recompiles: `TORCH_LOGS="graph_breaks,recompiles"` printed nothing. First calls (the compile) took about 4.6 s. Per call: 0.09 to 0.11 ms across runs, about 10x faster than eager. 226 kernels became 18, with names like `triton_poi_fused_add_clamp_cos_mul_select_sin_stack_1`.
- *`reduce-overhead` as it comes:* the "overwritten" error, within three calls, naming `self.t` in `reset`.
- *Fix 1 alone (`cudagraph_mark_step_begin`):* still fails, in a loop with the expert, because the state is carried across calls.
- *Fix 2 (clone the five state tensors):* runs, end state correct, 0.106 ms per call with the clones included.
- *Fix 3 (the config flag):* runs, end state correct. In the full control-step loop it was 1.89 ms against 1.82 ms for default compile (act eager in both), so no gain.
- *So here `reduce-overhead` didn't beat the default mode.* The compiled step is only 18 kernels and 0.027 ms of GPU time, but it still takes about 0.1 ms per call. The remaining time is the CPU, not the GPU: 18 launches plus Dynamo's checks before each call. A CUDA graph removes the launches, but the clones (or the config flag) bring cost back. It might win after fix 4 (updating in place), which I didn't test.
- *`act` compiled in default mode:* also worked, no graph breaks. Both compiled: 0.559 ms per control step, against 2.82 ms eager, so about 5x for the whole loop. `act` is now the bigger share (about 0.45 ms of the 0.56). End state correct.

The ranking, for what it's worth: default compile gave most of the speedup for one line of change. The rest of the effort is in `act`, and in whether in-place state updates would make `reduce-overhead` worth it.
] // claude:end
