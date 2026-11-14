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
= the expert, check: who reaches the goal, when, and how many lifts

*Symbols.*
- $d_"goal"$: distance from the puck to the goal, `(N,)`.
- `reached`: has this env ever been within 3 cm of the goal, `(N,)` bool.
- `first_step`: the step it first got there, `(N,)` integers, $-1$ if never.
- `closest`: the smallest $d_"goal"$ seen so far, `(N,)`.
- `lifts`: how many times this env lifted, `(N,)` integers.
- `i`: the loop step.

== The idea

Four tensors with one number per env, each updated every step, printed once after the loop. Nothing is logged to rerun and nothing is stored per step, so it stays cheap.

#align(center, diagram(
  node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg, spacing: (2.2cm, 1.6cm),
  node((0, 0), [`sim.x`], shape: rect),
  edge((0, 0), (1, 0), "->"),
  node((1, 0), [$d_"goal"$], shape: rect),
  edge((1, 0), (2, 0), "->"),
  node((2, 0), [`reached` \ `first_step` \ `closest`], shape: rect),
  node((0, 1), [`dropped` \ before, after], shape: rect),
  edge((0, 1), (2, 1), "->"),
  node((2, 1), [`lifts`], shape: rect),
  edge((2, 0), (3, 0.5), "->"),
  edge((2, 1), (3, 0.5), "->"),
  node((3, 0.5), [`print`], shape: rect),
))

Every arrow runs once per step; `print` runs once, after the loop.

== The rule

$ "now" = d_"goal" < 0.03, quad "first_step" arrow.l cases(i & "if now and not reached", "first_step" & "otherwise"), quad "reached" arrow.l "reached" or "now" $

$ "closest" arrow.l min("closest", d_"goal"), quad "lifts" arrow.l "lifts" + ("dropped before" and not "dropped after") $

`first_step` must be written _before_ `reached` is updated; once `reached` is True the test "now and not reached" is never True again, which is what freezes the first value. A lift is the moment `dropped` goes from True to False, so the two reads of `dropped` are either side of the line `dropped = dropped & (drift < 0.03)`.

== Shapes

#table(
  columns: (5.6cm, 5.6cm, 4.8cm),
  stroke: 0.5pt + fg, inset: 6pt,
  [*API*], [*does*], [*shapes*],
  [#api("(sim.x - goal).norm(dim=-1)")], [`goal` is `(2,)`, stretched against every env], [`(N,2)` $arrow.r$ `(N,)`],
  [#api("torch.where(c, i, first_step)")], [where `c` is True write the number `i`, else keep the old value], [`(N,)`],
  [#api("torch.minimum(a, b)")], [smaller of the two, entry by entry], [`(N,)`],
  [#api("was & ~dropped")], [`~` flips True and False; True only where it was dropped and no longer is], [`(N,)` bool],
  [#api("first_step[reached]")], [keep only the envs where `reached` is True (a copy)], [`(N,)` $arrow.r$ `(M,)`],
  [#api("torch.bincount(lifts)")], [how many envs have 0 lifts, 1 lift, 2 lifts, ...], [`(N,)` $arrow.r$ `(max+1,)`],
)

== Next edit

No new function. Before the loop: `goal`, plus `reached` (zeros, bool), `first_step` (all $-1$, long), `closest` (all `inf`) and `lifts` (zeros, long), each with `device=sim.device`. `torch.full((sim.n,), value, ...)` makes the constant ones. In the loop: two reads of `dropped` around the AND line for `lifts`, and the other three updates at the end of the loop.

After the loop, print: the percent that reached (`reached.float().mean()`); the mean, median, min and max of `first_step[reached]` (`.float()` first for mean and median); the mean and max of `closest`; the last $d_"goal"$ averaged, and the fraction still within 3 cm; the mean and max of `lifts`, and `bincount`.

== Checks

All 16384 envs, 600 steps, with the push ramp on the last waypoint:

#table(
  columns: (auto, auto),
  stroke: 0.5pt + fg, inset: 6pt,
  [*reached*], [100%],
  [*first step*], [mean 544, median 544, min 527, max 559],
  [*closest*], [mean 0.0100 m, max 0.0103 m],
  [*final distance*], [mean 0.0100 m, 100% within 3 cm],
  [*lifts*], [mean 2.00, max 2, counts `[0, 0, 16384]`],
)

Two ways to read it. `reached` and the final check differ when the puck overshoots: with the old push (no ramp) the first was 100% and the second 0%. And one mean can hide a spread, which is why min and max are printed: here every env arrives within 32 steps of the others.
] // claude:end
