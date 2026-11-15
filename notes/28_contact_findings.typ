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
#let t8(s) = text(size: 8pt, fill: fg, { show raw: set text(size: 7.5pt); s })
#let box-node(pos, label, c, name) = node(pos, text(fill: fg, size: 8.5pt, { show raw: set text(size: 8pt); label }), name: name, shape: rect, fill: c.transparentize(75%), stroke: 0.6pt + c, corner-radius: 3pt, inset: 6pt)

#align(center, block(fill: fg.transparentize(88%), inset: 10pt, radius: 3pt, text(size: 13pt, raw("obs (E, T, 7)  ->  offset_y (E',),  max_speed (E,)"))))

= Step 6e: where does the hand first touch the puck?

`E` is the number of envs in whatever you pass in: `N` for the rollout, `M` for the expert demos. `E'` is how many of them ever touched the puck.

== What we decided

- plot the hand's `y` *relative to the puck* at the first step they touch. the expert sits on one side of the puck to push it left or right, so that is where the choice shows up.
- one function for both. the rollout and `demos.obs_dem` are both env-first `(E, T, 7)` in the order x, v, q, so the same code reads either.
- the rollout saves `obs_raw` each step, not `sim.x`. the hand position is not stored anywhere, but `q` is in the obs and `fk` gives the hand from it.
- "touch" is the sim's own rule from `ee_force`, not a new one.
- print for each of expert, MSE and (later) flow: the fraction of envs that touch the puck, the contact `offset_y` histogram, the mean max puck speed *and* the share of envs whose max is over 1 m/s (a few huge launches drag a mean), and the success rate.
- the expert is the reference, and its numbers cover only the demos that succeeded. that is fine: it is exactly what the policy is trained to imitate.
- *not* plotting the puck's `y` at the pillar's x (see "Findings"): it is misleading.

== Calls

#table(
  columns: (7.2cm, 1fr),
  stroke: 0.5pt + fg, inset: 5pt, align: horizon,
  [`fk(q, sim.l, sim.base_z)`], [`(E, T, 3)` in, hand position `(E, T, 3)` out],
  [`mask.any(dim=1)`], [`(E, T)` bool to `(E,)` bool: did it ever happen],
  [`mask.int().argmax(dim=1)`], [index of the first True per row `(E,)`],
  [`torch.norm(v, dim=-1)`], [length of each vector, last dim removed],
  [`plt.subplots(2, 1, sharex=True)`], [two plots stacked, same x axis],
  [`ax.hist(arr, bins=40)`], [`arr` must be a 1D numpy array, not a CUDA tensor],
)

== 1. What goes in, what comes out

#align(center)[
  #diagram(
    spacing: (0.5cm, 1.2cm),
    node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg,
    box-node((0, 0), [`obs_raw` each step \ list of `T` `(N,7)`], c-dim, <a>),
    box-node((1, 0), [`torch.stack` \ `dim=1`], c-new, <b>),
    box-node((0, 1), [`record()`], c-dim, <c>),
    box-node((1, 1), [`demos.obs_dem` \ `(M,T,7)`], c-dim, <d>),
    box-node((2, 0.5), [the \ function], c-flag, <f>),
    box-node((3, 0), [`offset_y` \ `(E',)`], c-new, <g>),
    box-node((3, 1), [`max_speed` \ `(E,)`], c-new, <h>),
    edge(<a>, <b>, "->"),
    edge(<c>, <d>, "->"),
    edge(<b>, <f>, "->"),
    edge(<d>, <f>, "->"),
    edge(<f>, <g>, "->"),
    edge(<f>, <h>, "->"),
  )
]

#align(center, t8[the stack gives `(N, T, 7)`, about 275 MB at `N = 16384`. keep it on the GPU as torch until plotting. `offset_y` goes into the histograms, `max_speed` only into a mean to print.])

== 2. Inside the function

#align(center)[
  #diagram(
    spacing: (0.5cm, 1.2cm),
    node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg,
    box-node((0, 0.5), [`obs` \ `(E,T,7)`], c-dim, <o>),
    box-node((1, 0), [`x` puck xy \ `(E,T,2)`], c-dim, <x>),
    box-node((1, 1), [`q` \ `(E,T,3)`], c-dim, <q>),
    box-node((2, 1), [`p` hand \ `(E,T,3)`], c-new, <p>),
    box-node((3, 0.5), [`contact` \ `(E,T)` bool], c-flag, <m>),
    box-node((4, 0), [`valid` \ `(E,)` bool], c-new, <v>),
    box-node((4, 1), [`first` \ `(E,)` long], c-new, <fi>),
    box-node((5, 0.5), [`offset_y` \ `(E,)`], c-new, <r>),
    edge(<o>, <x>, "->"),
    edge(<o>, <q>, "->"),
    edge(<q>, <p>, "->", t8[`fk`], label-side: left),
    edge(<x>, <m>, "->"),
    edge(<p>, <m>, "->"),
    edge(<m>, <v>, "->"),
    edge(<m>, <fi>, "->"),
    edge(<fi>, <r>, "->"),
    edge(<v>, <r>, "->"),
  )
]

#align(center, t8[`x = obs[..., 0:2]`, `v = obs[..., 2:4]`, `q = obs[..., 4:7]`. `offset_y` is the hand's `y` minus the puck's `y`, both read at step `first`. `valid` is applied last, so only envs that touched are kept.])

== 3. The touch rule, from `sim.py`

#table(
  columns: (1fr, 1fr, 1fr),
  stroke: 0.5pt + fg, inset: 5pt, align: horizon,
  [*overlap in the plane*], [*low enough*], [*constants*],
  [`norm(x - p_xy, dim=-1) < r_p + r_e + tol`], [`p_z - r_e < h`], [`sim.r_p` 0.05, `sim.r_e` 0.02, `sim.h` 0.03, `tol` 0.005],
)

#align(center, t8[both are `(E, T)` bool; `&` them. read the constants from `sim`, do not copy the numbers. `tol` is 5 mm and is yours to choose.])

== 4. First True in each row, on a different subject

`passed[s, d]` is whether student `s` had passed by day `d`. which day did each pass first, and how did they score that day?

```python
passed = torch.tensor([[0,0,1,1], [1,1,1,1], [0,0,0,0]], dtype=torch.bool)  # (3, 4)
score  = torch.tensor([[5,6,7,8], [1,2,3,4], [9,9,9,9]])                     # (3, 4)

passed.any(dim=1)                              # [True, True, False]
first = passed.int().argmax(dim=1)             # [2, 0, 0]
score[torch.arange(3), first]                  # [7, 1, 9]
score[torch.arange(3), first][passed.any(dim=1)]  # [7, 1]
```

#note(c-warn)[Trap][
- *without `tol` the expert never seems to touch the puck until step 558.* it pushes with almost no overlap, so when the obs is sampled the hand is at about `r_p + r_e`, sometimes a hair over, and a strict `<` is true on only a few steps. measured: the puck first moves at step 112, the strict rule says 558, with `tol = 0.005` it says 109.
- the third student never passed, but `argmax` still returned `0` and raised no error. the `9` is meaningless. that is what `valid` is for, and it must be applied last.
- `argmax` raises an error on a bool tensor (checked on this torch), so make the mask `.int()` first.
- `torch.arange(E)` needs `device=` the same as the tensors, or the paired index errors.
- append `obs_raw` *before* `sim.step`. the step that makes `t == T` resets the env, so the obs after it is the start pose again.
- the expert's `obs` was recorded before each action, the same convention as the rollout append, so the touch step lines up.
]

== 5. Names

#table(
  columns: (3cm, 3cm, 2.4cm, 1fr),
  stroke: 0.5pt + fg, inset: 4pt, align: horizon,
  [*name*], [*shape*], [*dtype*], [*what*],
  [`obs_list`], [`[]`, then length `T`], [list], [one `obs_raw` per step, appended before the step],
  [`obs_all`], [`(N, T, 7)`], [float32], [`torch.stack(obs_list, dim=1)`],
  [`p`], [`(E, T, 3)`], [float32], [hand position from `fk`],
  [`contact`], [`(E, T)`], [bool], [the two conditions, `&`'d],
  [`valid`], [`(E,)`], [bool], [env ever touched],
  [`first`], [`(E,)`], [long], [first touch step per env],
  [`offset_y`], [`(E',)`], [float32], [hand `y` minus puck `y` at `first`],
  [`speed`], [`(E, T)`], [float32], [`norm(v, dim=-1)`],
  [`max_speed`], [`(E,)`], [float32], [max over time, then `.mean()` to print],
)

== 6. The plot

two rows sharing one x axis: expert on top, policy below, titled. `.cpu().numpy()` each `offset_y` only here. a vertical line at `0` (hand level with the puck centre). save with `fig.savefig(plots / "...")`. `plots/` already exists in `rollout.py`.

== Findings so far (measured 2026-11-15)

all 16384 envs, expert demos vs the `mse.pt` from the 5000-step run. a retrained policy will move the MSE column. the page rules say measurements do not belong here; they are here because you asked for everything to be kept.

#table(
  columns: (4.4cm, 1fr, 1fr, 1fr),
  stroke: 0.5pt + fg, inset: 4pt, align: horizon,
  [*metric*], [*expert*], [*MSE, k = 1*], [*MSE, k = 8*],
  [envs that touch the puck], [100%], [83%], [79%],
  [first touch step (median)], [109], [90], [90],
  [`offset_y` within 2 cm of 0], [0%], [50%], [78%],
  [hand `x` behind puck at touch], [about 3.3 cm], [about 5.9 cm], [about 6.0 cm],
  [mean max puck speed (m/s)], [0.35], [2.12], [2.49],
  [envs with max speed over 1 m/s], [0%], [96%], [100%],
  [success rate], [demos are successes only], [0.01%], [0.3%],
)

- *contact `offset_y`*: the expert is two narrow spikes at `+0.067` and `-0.067`. MSE at `k = 8` is one blob around 0. MSE at `k = 1` is wider and skewed negative.
- the puck's `y` at the pillar's x is *bad*: only 11 to 18 percent of MSE envs reach that x at all, and those pucks have been deflected to `y` about `±0.2`, so it looks two-sided without showing any choice by the policy (survivorship plus deflection). dropped.
- the policy's hand first touches the puck at about step 90, earlier than the expert's 109. whether that is because it has no "park high, then drop" stage is a guess, not measured.
- earlier, at `k = 4`, 25 percent of envs had the puck go beyond 1 m from the origin at some point, first at step 100 and median 114. the sim clamps `qdot` to `u_clip = 2.0`, so the policy's raw outputs (up to 34) are not the cause.
- you watched env 0 in rerun: the hand squashes the puck into the pillar, and the penalty contact throws it out fast.
- the cause you named: the puck starts at `(0.2, 0)`, the pillar is at `(0.4, 0)` with radius 0.10, the goal is at `(0.6, 0)`, so the straight line hits the pillar. the expert's coin sends it around `+0.2` or `-0.2`, and the coin is not in the obs. MSE averages left and right into straight ahead. the data fits; no flow run has tested the cure.
- the touch rule needs a 5 mm tolerance (see Trap), or the expert looks as if it first touches at step 558.

what this cannot show:
- it is a snapshot at one step. a flow policy could pick a side at contact and switch between chunks later; this histogram would miss that.
- it counts only envs that touch. if few do, the histogram is thin and may be accidental sweeps, so always print `E'` out of `E`.

== Loose ends

- `.gitignore`: `checkpoints/`, `plots/`, `rollout.png`.
- take the rerun lines (`rr.init`, `sim.log_scene`, `sim.log`) out of `rollout.py` when done, each run opens a viewer.
- `train.py` has an unused `Timer` import. `K` is hard-coded as 16 and 48 in `MSEPolicy`. `record_demos.py` comments still say `train.py` and `!= "check"`. `map.typ` step 4 row still says flat `(T*M, 7)`.
- the expert's `if done.any(): expert.reset(done)` in `record()` does nothing useful, since `done` only fires on the last step, and it forces a GPU wait each step (600 times). delete it or keep it for a longer loop; your call.
- `record_demos.py` and `train.py` have uncommitted changes; `rollout.py` and `checkpoints/` are untracked.
- this page and the 6d page (`archive/old_typ/step6d_save_and_rollout.typ`) are not yet in `notes/`; add as `28` and `29` when finished.
- later: flow policy with the same four numbers, and a transformer.

== Look at

#note(c-flag)[Look at][
- print `valid.sum()` out of `E` for the expert and for the policy. if almost no policy env touches, that is a result, and the histogram will be thin.
- the expert's histogram: two narrow spikes near `+0.067` and `-0.067`, nothing within 2 cm of `0`. the first touch step should be about 109, close to the step the puck first moves.
- `success.float().mean()`, and the two mean max speeds, one line each.
- the shapes in section 5 at each step. if `argmax` returns `(E,)` and the paired index gives `(E,)`, the rest is bookkeeping.
]
] // claude:end
