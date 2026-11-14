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
#let c-state = rgb("#4f8f8b")
#let c-flag = rgb("#c9a45c")
#let c-warn = rgb("#b5655f")
#let chip(c, s) = box(fill: c.transparentize(65%), inset: (x: 4pt, y: 2pt), radius: 2pt, text(fill: fg, s))
#let nd(pos, c, body, name) = node(pos, body, fill: c.transparentize(75%), stroke: 0.6pt + c, name: name, corner-radius: 3pt)

= step 2: episodes

== What we decided

- `Sim` knows nothing about who drives it. Expert now, MSE and flow policies later, all plug in the same way.
- `Sim` owns the clock: a counter for each env, an episode count, and the success of the episode that just ended.
- `step` resets finished envs inside itself (Isaac Lab style) and tells the caller who finished.
- The expert stays separate, so the _caller_ passes the news on with `expert.reset(done)`.
- Demos get collected in step 4, not here. Nothing in this step records `(obs, action)`.

== Colours

#chip(c-state, [state]) per-env numbers that live between steps. #chip(c-flag, [signal]) a flag produced by one step and read by the caller. #chip(c-warn, [order matters]) a place where swapping two lines gives the wrong answer silently.

== Inside `Sim.step`

The old body (arm, 4 substeps) stays as it is. The new part comes after it.

#align(center, diagram(
  node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg, spacing: (2.2cm, 1.9cm),
  nd((0, 0), fg, [physics \ (already there)], <phys>),
  nd((1, 0), c-state, [`t` + 1], <tick>),
  nd((2, 0), c-flag, [`done` \ `t` = 600], <done>),
  nd((3, 0), c-flag, [`success` \ puck within 3 cm], <succ>),
  nd((3, 1), c-warn, [reset only \ the `done` envs], <reset>),
  nd((2, 1), c-state, [`t` back to 0 \ `episode` + 1], <zero>),
  nd((1, 1), c-flag, [build `obs` \ `(N, 7)`], <obs>),
  edge(<phys>, <tick>, "->"),
  edge(<tick>, <done>, "->"),
  edge(<done>, <succ>, "->"),
  edge(<succ>, <reset>, "->", [only after this], label-side: left),
  edge(<reset>, <zero>, "->"),
  edge(<zero>, <obs>, "->"),
))

The one red box is the trap: `success` has to read `x` _before_ `reset` moves the puck back to the start. The envs that did not finish must come out of the reset untouched.

== The caller

#align(center, diagram(
  node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg, spacing: (3.4cm, 1.8cm),
  nd((0, 0), fg, [`expert.act()`], <act>),
  nd((1, 0), fg, [`sim.step(qdot)`], <step>),
  nd((2, 0), c-flag, [`obs, done, success`], <out>),
  nd((2, 1), c-warn, [`expert.reset(done)`], <er>),
  edge(<act>, <step>, "->", [`qdot`]),
  edge(<step>, <out>, "->"),
  edge(<out>, <er>, "->", [any `done`], label-side: left),
  edge(<er>, <act>, "->", [next step], bend: -40deg, label-side: right),
))

Forgetting `expert.reset(done)` crashes nothing. The expert drags last episode's `dropped` and `idx` into the new one. If success falls on episode 2, look here first.

== You write it

Two files, in this order. After each, run and look at the numbers.

*A. `sim.py`*

#chip(c-state, [1. new state]) in `__post_init__`

#table(
  columns: (auto, auto, auto, auto),
  stroke: 0.5pt + fg, inset: 6pt,
  [*name*], [*shape*], [*dtype*], [*starts at*],
  [`t`], [`(N,)`], [`torch.long`], [0],
  [`episode`], [`(N,)`], [`torch.long`], [0],
  [`last_success`], [`(N,)`], [`torch.bool`], [False],
)

`episode` is per env, not one number, because envs drift apart later and sit on different episodes.

#chip(c-flag, [2. new code]) at the end of `step`, after the substeps

One line of code per box of the first diagram, in the same order: `t` + 1, `done`, `success`, reset, zero `t` and bump `episode`, build `obs`. Optional: also write `last_success`.

#chip(c-warn, [3. what `step` returns])

#table(
  columns: (auto, auto, auto, 1fr),
  stroke: 0.5pt + fg, inset: 6pt,
  [*name*], [*shape*], [*dtype*], [*meaning*],
  [`obs`], [`(N, 7)`], [float], [`q` (3), then `x` (2), then `v` (2)],
  [`done`], [`(N,)`], [bool], [True for envs that just hit step 600],
  [`success`], [`(N,)`], [bool], [puck within 3 cm of the goal at that last step],
)

Tools: `torch.where(cond, a, b)`, bool indexing, `sim.reset(mask)` (already written, it resets only the masked envs), `torch.cat(list, dim=-1)`.

*B. `run_expert.py`*
- loop for 3 episodes' worth of steps, call `expert.reset(done)` on the right step, print one line when any env finishes: step number, how many finished, success rate among them.

== Look at

- three lines at steps 599, 1199, 1799: `finished 16384   success 100.0%`
- `sim.t.unique()` at the end is all zeros, and `sim.episode.unique()` is `tensor([3])`.
- the trap check: print `sim.x[0]` on the step of the first finish, before and after your reset. Before is near the goal `(0.6, 0.0)`. After is near the start `(0.2, 0.0)`. If `success` came out 0%, you read it after.
- the silent bug: comment out `expert.reset(done)` once and see what happens to episodes 2 and 3.

If you get stuck on one box, tell me which one and what you tried.
] // claude:end
