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
= step 1: check the expert

Map: step 1 of 8 (`map.typ`). Afterwards you will know, for every env, whether the expert got the puck to the goal, when it first got there, and how many times it lifted.

== Where you are

The code is in four files now. You only edit `run_expert.py` this session.

#table(
  columns: (auto, 1fr),
  stroke: 0.5pt + fg, inset: 6pt,
  [`kin.py`], [`fk`, `jac`, `dls`],
  [`sim.py`], [`Sim`: the puck, the pillar, the arm. `sim.step(qdot)` moves everything one control step],
  [`expert.py`], [`Expert`: park high behind the puck, drop, push, lift and re-park if the hand drifts. `expert.act()` returns `qdot`. It counts lifts in `expert.lifts`],
  [`run_expert.py`], [builds both, runs 600 steps, logs env 0 to rerun],
)

`run_expert.py` with the three places your code goes marked:

```
sim = Sim()
sim.reset(torch.ones(sim.n, dtype=torch.bool, device=sim.device))
expert = Expert(sim)
# (1) BEFORE THE LOOP: make your tensors
rr.init("robopuck", spawn=True)
sim.log_scene()
for i in range(600):
    qdot = expert.act()
    sim.step(qdot)
    # (2) IN THE LOOP: update them
    sim.log(i)
# (3) AFTER THE LOOP: print
```

(2) comes after `sim.step`, so it sees the puck after this step's push. `i` is the step number, 0 to 599.

== The numbers you will keep

Each is one number per env, so shape `(N,)` with N = 16384.

- $d_"goal"$: distance from the puck to the goal _at this step_.
- `now`: is $d_"goal"$ under 3 cm at this step. bool.
- `reached`: has it _ever_ been under 3 cm. bool. Once an entry is True it stays True: `[F,T,F] | [F,F,F]` = `[F,T,F]`.
- `first_step`: the step when it first got under 3 cm. integer. $-1$ means never.
- `closest`: the smallest $d_"goal"$ so far. float.

Do one part at a time. After each, run and print the new tensor's first few entries (`x[:5]`) to see it work.

== Part 1: distance to the goal

(1) Make `goal`, shape `(2,)`. The goal is the last waypoint, the last row of `sim.route` (shape `(3, 2)`).

(2) Two lines:

```
d_goal = ______   # (N, 2) minus (2,), then length -> (N,)
now    = ______   # (N,) bool
```

Look at: `d_goal[:5]` printed every 100 steps goes down from about 0.4 to about 0.01.

== Part 2: did it reach, and when

(1) Make `reached` and `first_step`, both `(N,)`, with the dtype and starting value from the list above. Signatures:
- `torch.zeros(size, dtype=, device=)`: all zeros. For bool that means all False.
- `torch.full(size, fill_value, dtype=, device=)`: every entry is `fill_value`.
- dtypes: `torch.bool`, `torch.long` (integers). device: `sim.device`.

(2) Two lines, in this order:

```
first_step = torch.where(______, i, first_step)
reached    = reached | now
```

`torch.where(cond, a, b)` picks `a` where `cond` is True and `b` elsewhere. `a` can be a plain python number like `i`. You want `cond` True only for envs that are close now _and_ were never close before. At this point `reached` still means "before this step", which is why the `first_step` line comes first.

Look at: `first_step[:5]` sits at $-1$, then each entry jumps to a step number once and never changes again.

== Part 3: closest approach

(1) Make `closest`, `(N,)` float, starting at a value every real distance beats. `float("inf")` is infinity.

(2) One line:

```
closest = torch.minimum(______, ______)
```

`torch.minimum(a, b)` takes the smaller of each pair of entries: `minimum([3, 1], [2, 5])` = `[2, 1]`. Both arguments `(N,)`.

Look at: `closest[:5]` only ever goes down.

== Part 4: print

(3) After the loop, fill in these five lines:

```
print(f"reached       {reached.float().mean():.1%}")
print(f"first step    mean ___   min ___   max ___")
print(f"closest       mean ___ m")
print(f"final dist    mean ___ m   within 3 cm ___")
print(f"lifts         mean ___   max ___")
```

What you need:
- `x[mask]` with a `(N,)` bool `mask` gives a 1-D tensor of only the entries where `mask` is True. Use it so the $-1$ entries of `first_step` don't count.
- `.mean()` needs floats: `.float()` first. `.min()` and `.max()` work on integers.
- after the loop, `d_goal` and `now` still hold the last step's values: that is the "final dist" line.
- formats: `f"{x:.1%}"` for a fraction as a percent, `f"{x:.4f}"` for a length. A one-number tensor formats fine.
- `torch.bincount(expert.lifts)` counts how many envs had 0, 1, 2 lifts.

== Target

What it printed when I ran it (CUDA, seeds 0 and 1 gave the same):

#table(
  columns: (auto, auto),
  stroke: 0.5pt + fg, inset: 6pt,
  [*reached*], [100.0%],
  [*first step*], [mean 544, min 529, max 559],
  [*closest*], [mean 0.0100 m],
  [*final dist*], [mean 0.0100 m, within 3 cm 100.0%],
  [*lifts*], [mean 2.00, max 2],
)

Invariant to look at: `first_step >= 0` is True exactly where `reached` is True.

When this matches, step 1 is done: move "here" to step 2 in `map.typ`.
] // claude:end
