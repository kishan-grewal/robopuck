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
#let nd(pos, c, body, name) = node(pos, text(size: 9pt, body), fill: c.transparentize(75%), stroke: 0.6pt + c, name: name, corner-radius: 3pt, shape: rect)
#let note(c, title, body) = block(width: 100%, fill: c.transparentize(85%), stroke: 0.6pt + c, inset: 8pt, radius: 3pt)[#text(weight: "bold", fill: fg)[#title] \ #body]
#show raw: set text(size: 8.5pt)

= Step 4: the demo buffer

== What we decided

- A row is a pair: the obs the expert *saw* when it chose an action, and the `qdot` it chose. Not the obs that comes back after the action.
- `Buffer` is a dataclass in `data.py`. It owns `obs_buf` and `act_buf` and nothing else. The 600-step loop lives in your script, like `run_expert.py`.
- The tensors are `(T, N, ...)` because the loop writes one step for all envs at a time. After filtering they are flattened to rows.
- Record one episode per env first. Keep only the envs whose episode succeeded.
- `obs` stays 7 numbers (`x`, `v`, `q`). Nothing in it says which side the coin picked.

== Done, and still to write

Done in your `data.py`: `T`, `N` and `device` as arguments, `obs_buf` `(T, N, 7)` and `act_buf` `(T, N, 3)` made in `__post_init__`.

Still to write, two methods on `Buffer` and the loop that calls them:

#table(
  columns: (4.6cm, 1fr),
  stroke: 0.5pt + fg, inset: 6pt, align: horizon,
  [*signature*], [*what it does*],
  [`add(self, t, obs, qdot)`], [fills row `t` of both tensors. It returns nothing. `obs` is `(N, 7)` and `qdot` is `(N, 3)`, both for all envs at step `t`.],
  [`finish(self, success)`], [`success` is `(N,)` bool. Drops the envs that failed, then merges the time and env axes into one. Returns `obs` `(T*M, 7)` and `qdot` `(T*M, 3)`, where $M$ = number of envs that succeeded.],
)

A method gets `self` first, and through it the two tensors. The values `add` and `finish` receive as arguments (`t`, `obs`, `qdot`, `success`) are the loop's own variables, so they are not fields.

== The picture

One step $t$ in the loop. The buffer gets its two inputs from *before* `step`.

#align(center, diagram(
  node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg, spacing: (1.0cm, 1.5cm),
  nd((0, 0), c-cpu, [`obs` (before) \ `(N, 7)`], <o>),
  nd((1.3, 0), c-gpu, [`expert.act()` \ reads the sim itself], <a>),
  nd((2.6, 0), c-gpu, [`qdot` \ `(N, 3)`], <q>),
  nd((3.9, 0), c-gpu, [`sim.step(qdot)`], <s>),
  nd((5.2, 0), c-flag, [`obs` (after) \ `(N, 7)`], <n>),
  nd((1.3, 1.4), c-new, [`buffer.add(t, obs, qdot)`], <b>),
  edge(<o>, <a>, "->"), edge(<a>, <q>, "->"), edge(<q>, <s>, "->"), edge(<s>, <n>, "->"),
  edge(<o>, <b>, "->", bend: -20deg), edge(<q>, <b>, "->", bend: 20deg),
  edge(<n>, <o>, "->", bend: -35deg, label: [becomes `obs` next step], label-side: left),
))

After the last step, `finish` filters on the env axis, then merges time and env:

#align(center, diagram(
  node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg, spacing: (1.1cm, 1.2cm),
  nd((0, 0), c-new, [`obs_buf` \ `(T, N, 7)`], <a>),
  nd((1.4, 0), c-flag, [keep envs with \ `success` \ `(T, M, 7)`], <b>),
  nd((2.8, 0), c-gpu, [merge \ `(T*M, 7)`], <c>),
  edge(<a>, <b>, "->"), edge(<b>, <c>, "->"),
))

Same for `act_buf`, ending at `(T*M, 3)`. Row $r$ of one still belongs with row $r$ of the other.

== The loop (in your script)

The loop uses these, all made before it starts:

#table(
  columns: (2.2cm, 2.0cm, 1fr),
  stroke: 0.5pt + fg, inset: 6pt, align: horizon,
  [*name*], [*shape*], [*notes*],
  [`buffer`], [], [one `Buffer(T=600, N=sim.n)`, created once],
  [`obs`], [`(N, 7)`], [the obs the expert is about to act on. First value: the sim's state after the post-warm-up reset, in the order `x`, `v`, `q`. After that: what `step` returned.],
  [`qdot`], [`(N, 3)`], [what `act()` returned this step],
  [`success`], [`(N,)`], [from the *last* `step` call (where $t$ reaches 600): the verdict for the episode],
)

Order inside one step: `act`, `add`, `step`, then the returned obs replaces `obs`. `finish` runs once after all 600 steps.

#note(c-warn)[Trap 1: the obs after the step][
`step` returns the obs *after* the action. If that goes into `add`, every pair is shifted by one step. Shapes are right and numbers look fine, and only training goes badly.
]

#note(c-warn)[Trap 2: the last step already reset][
`step` calls `reset` inside, so on step 600 the obs it returns is the start of the *next* episode. Don't store it. `success` is worked out before that reset, so it is still the verdict for the episode that just ended.
]

#note(c-warn)[Trap 3: warm-up is not data][
The compile warm-up moves the sim and the expert on. Reset both (all-true mask) after it, and make the first `obs` after that reset.
]

== Two small things to look at

#note(c-flag)[Hints][
- In `add`: print the shape of `obs_buf[t]` and compare it with the shape of `obs`.
- In `finish`: `success` has one entry per env. Which axis of `obs_buf` is the env axis?
- In `data.py`: `T` and `N` are typed `float`. `torch.zeros` wants integer sizes. Which type should they be?
]

#note(c-flag)[Look at][
- shapes before and after `finish`, and $M$ against $N$. Target: $M = N$, because the expert succeeds everywhere.
- `obs_buf[0]` across envs: all near the start pose, `v` all 0.
- one env's `x` over `t`: continuous, no jumps. A jump means the obs was shifted or reset.
- `act_buf` min and max against the `u_clip` of 2.0. `step` clamps `qdot` to that range inside, but `act()` does not, so the buffer can hold values outside it.
]
] // claude:end
