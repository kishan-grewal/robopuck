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
= the expert, step 1: replace `u` with `dls(jac(q), v_des)`

*Symbols.* $q in RR^3$ joint angles. $p(q) in RR^3$ hand position (`fk`). $v_"des" in RR^3$ the hand velocity we want. $dot(q) in RR^3$ joint velocity, which is `u`. $J(q) in RR^(3 times 3)$ Jacobian. $lambda = 0.05$ damping.

== What $J$ is, in your words

You are right that $p(q)$ bends space. Equal steps in joint angle do not give equal steps of the hand. $J$ is the _straight-line stand-in_ for that bend at one $q$: for a tiny change $d q$, the hand moves $d p approx J thin d q$. A different $q$ bends differently, so $J$ is different. That is why every env, every control step, gets its own $J$.

Planar slice (shoulder $q_2$, elbow $q_3$, hand $(rho, z)$): the grid on the left is joint space, the bent grid on the right is where those same joint values put the hand. At the dot, the small joint square becomes (almost) a parallelogram. Its two sides are the two columns of $J$.

#let L = 0.4
#let s = 6
#let hp(a, b) = (s * L * (calc.cos(a) + calc.cos(a + b)), s * L * (calc.sin(a) + calc.sin(a + b)))
#let c2(a, b) = (-s * L * (calc.sin(a) + calc.sin(a + b)), s * L * (calc.cos(a) + calc.cos(a + b)))
#let c3(a, b) = (-s * L * calc.sin(a + b), s * L * calc.cos(a + b))
#let grid-col = fg.transparentize(55%)
#let acc = rgb("#e8a23a")
#align(center, grid(
  columns: (6.8cm, 8cm), column-gutter: 6mm,
  canvas(length: 1cm, {
    import draw: *
    set-style(stroke: fg)
    for i in range(0, 5) {
      line((i * 0.3 * 3, 0), (i * 0.3 * 3, 4.8), stroke: 0.5pt + grid-col)
      line((0, i * 1.2), (3.6, i * 1.2), stroke: 0.5pt + grid-col)
    }
    // dot at q2 = 0.6, q3 = -1.2
    circle((0.6 * 3, (-1.2 + 2.0) * 3), radius: 0.07, fill: acc, stroke: none)
    let o = (0.6 * 3, (-1.2 + 2.0) * 3)
    line(o, (o.at(0) + 0.4 * 3, o.at(1)), stroke: 1.2pt + acc, mark: (end: ">", scale: 0.5))
    line(o, (o.at(0), o.at(1) + 0.4 * 3), stroke: 1.2pt + acc, mark: (end: ">", scale: 0.5))
    content((3.0, -0.35), text(size: 9pt)[$q_2$ $arrow.r$])
    content((-0.45, 2.4), text(size: 9pt)[$q_3$ $arrow.t$])
    content((1.8, 5.2), text(size: 9pt)[joint space])
  }),
  canvas(length: 1cm, {
    import draw: *
    set-style(stroke: fg)
    for i in range(0, 5) {
      let a = i * 0.3
      line(..range(0, 13).map(k => hp(a, -2.0 + k * 0.1333)), stroke: 0.5pt + grid-col)
      let b = -2.0 + i * 0.4
      line(..range(0, 13).map(k => hp(k * 0.1, b)), stroke: 0.5pt + grid-col)
    }
    let o = hp(0.6, -1.2)
    circle(o, radius: 0.07, fill: acc, stroke: none)
    let u = c2(0.6, -1.2)
    let w = c3(0.6, -1.2)
    line(o, (o.at(0) + 0.4 * u.at(0), o.at(1) + 0.4 * u.at(1)), stroke: 1.2pt + acc, mark: (end: ">", scale: 0.5))
    line(o, (o.at(0) + 0.4 * w.at(0), o.at(1) + 0.4 * w.at(1)), stroke: 1.2pt + acc, mark: (end: ">", scale: 0.5))
    content((o.at(0) + 0.4 * u.at(0) - 0.55, o.at(1) + 0.4 * u.at(1) + 0.15), text(size: 8pt, fill: acc)[col 2])
    content((o.at(0) + 0.4 * w.at(0) - 0.1, o.at(1) + 0.4 * w.at(1) + 0.3), text(size: 8pt, fill: acc)[col 3])
    content((2.0, 5.2), text(size: 9pt)[hand space $(rho, z)$])
  }),
))

== The two tools are separate

*`jacrev` turns a function into its Jacobian function.* Give it `f` that takes one `q` of shape `(3,)` and returns a hand position `(3,)`. You get back a new function that takes one `q` and returns `(3,3)`. It computes nothing until you call it.

*`vmap` turns a one-example function into a batch function.* It has nothing to do with derivatives. `vmap(g)` calls `g` on every row of the leading axis, all at once, and stacks the results.

#align(center, diagram(
  node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg, spacing: (16mm, 8mm),
  node((0, 0), [`f` \ `(3,)` $arrow.r$ `(3,)`], shape: rect),
  edge("->", [`jacrev`]),
  node((1, 0), [`Jf` \ `(3,)` $arrow.r$ `(3,3)`], shape: rect),
  edge("->", [`vmap`]),
  node((2, 0), [`jac` \ `(N,3)` $arrow.r$ `(N,3,3)`], shape: rect, fill: user-color.transparentize(70%)),
))

Why both: `jacrev` only knows how to do _one_ arm. Hand it a whole `(N,3)` batch and it differentiates every output with respect to every input, giving `(N,3,N,3)` that is almost all zeros. `vmap` is what keeps the envs apart.

`jacrev` differentiates with respect to the first argument only, and `fk(q, L, base_z)` has two more. Wrap it:

#table(
  columns: (5.2cm, 6.4cm, 4.4cm),
  stroke: 0.5pt + fg, inset: 6pt,
  [*API*], [*does*], [*shapes*],
  [#api("f = lambda q: fk(q, L, base_z)")], [one-input version of `fk`], [`(3,)` $arrow.r$ `(3,)`],
  [#api("torch.func.jacrev(f)")], [function that returns $J$ for one `q`], [`(3,)` $arrow.r$ `(3,3)`],
  [#api("torch.func.vmap(Jf)")], [same function, for every row], [`(N,3)` $arrow.r$ `(N,3,3)`],
)

== Replacing `u` in your file

Right now the bottom of `puck.py` has `u = torch.zeros(sim.n, 3)` and `sim.step(u)`. That `u` is what `dls` takes over. $v_"des"$ is a made-up constant for now, so you can see the converter work before building the rest of the expert.

#align(center, diagram(
  node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg, spacing: (9mm, 7mm),
  node((0, 0), [`sim.q` \ `(N,3)`], shape: rect),
  edge("->", [`jac`]),
  node((1, 0), [`J` \ `(N,3,3)`], shape: rect),
  edge("->"),
  node((2, 0), [`dls`], shape: rect, fill: user-color.transparentize(70%)),
  edge("->"),
  node((3, 0), [`u` \ `(N,3)`], shape: rect),
  edge("->"),
  node((4, 0), [`sim.step`], shape: rect),
  node((2, 1), [`v_des` \ `(N,3)` constant], shape: rect),
  edge((2, 1), (2, 0), "->"),
))

Two new functions beside `fk`, and one change in the loop:

```
def jac(q, L, base_z):          # (N,3) -> (N,3,3)      uses jacrev, vmap
def dls(J, v_des, lam=0.05):    # (N,3,3), (N,3) -> (N,3)

for i in range(400):
    # u = jac(...) then dls(...), computed fresh from sim.q every step
    sim.step(u)
```

`u` is no longer defined above the loop. `v_des` is, for example, `torch.tensor([0, 0.1, 0.]).expand(sim.n, 3)`.

== Checks

Set the arm to `q = (0, 0.6, -1.2)` first (`sim.q[:] = ...` or change `q0`). Its `fk` is $(0.660, 0, 0.400)$.

#table(
  columns: (auto, auto, auto, 1fr),
  stroke: 0.5pt + fg, inset: 6pt,
  [*$v_"des"$*], [*$dot(q)$*], [*$J dot(q)$*], [*note*],
  [$(0, 0.1, 0)$], [$(0.151, 0, 0)$], [$(0, 0.099, 0)$], [only the base yaws],
  [$(0.1, 0, 0)$], [$(0, -0.207, 0.417)$], [$(0.094, 0, 0.001)$], [elbow and shoulder; 6% short from the damping],
  [$(0, 0, -0.1)$], [$(0, -0.149, -0.003)$], [$(-0.001, 0, -0.099)$], [straight down],
)

`J.shape == (N,3,3)`. Then the loop test: run 50 control steps (1 s) with $v_"des" = (0, 0.1, 0)$. `fk(sim.q, ...)` should end near $(0.660, 0.099, 0.400)$: the hand slid 0.1 m sideways and did not drift in $x$ or $z$.
] // claude:end

#claude[ // claude:begin
= dls: from $v_"des"$ to $dot(q)$

*Symbols.* $J in RR^(3 times 3)$ Jacobian at the current $q$. $v_"des" in RR^3$ wanted hand velocity. $dot(q) in RR^3$ joint velocity to send. $sigma_i$ singular values of $J$ (how strongly $J$ stretches each direction). $lambda = 0.05$ damping. $I$ identity. $J^T$ rows and columns of $J$ swapped.

== The problem with $J^(-1)$

$J dot(q) = v_"des"$ looks solved by $dot(q) = J^(-1) v_"des"$. In a direction where $sigma_i$ is tiny (arm nearly stretched out), $J^(-1)$ multiplies by $1 slash sigma_i$, so a small hand request becomes a huge joint speed.

== The trade

Choose $dot(q)$ to minimise tracking error plus a price on joint speed:

$ min_(dot(q)) thin norm(J dot(q) - v_"des")^2 + lambda^2 norm(dot(q))^2 $

Set the gradient to zero:

$ (J^T J + lambda^2 I) thin dot(q) = J^T v_"des" $

Call the left matrix $A$ and the right side $b$. $lambda^2 I$ keeps $A$ invertible even when $J$ is not, and you solve $A dot(q) = b$ instead of inverting.

== What it does per direction

Exact inversion scales direction $i$ by $1 slash sigma_i$. DLS scales it by $sigma_i slash (sigma_i^2 + lambda^2)$:

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt + fg, inset: 6pt,
  [*$sigma_i$*], [*exact $1 slash sigma_i$*], [*DLS*],
  [0.66], [1.5], [1.5],
  [0.20], [5.0], [4.6],
  [0.01], [100], [3.8],
  [0], [$infinity$], [0],
)

Healthy directions are nearly unchanged. Near-singular ones are capped at $1 slash (2 lambda) = 10$.

== Shapes

#align(center, diagram(
  node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg, spacing: (9mm, 12mm),
  node((0, 0), [`J` \ `(N,3,3)`], shape: rect),
  edge("->"),
  node((1, 0), [`A = J.mT @ J` \ `+ lam**2 * eye(3)`], shape: rect),
  edge("->"),
  node((2, 0), [`solve(A, b)` \ `(N,3,1)`], shape: rect),
  edge("->"),
  node((3, 0), [`[..., 0]` \ `qdot (N,3)`], shape: rect, fill: user-color.transparentize(70%)),
  node((0, 1), [`v_des` \ `(N,3)`], shape: rect),
  edge("->"),
  node((1, 1), [`b = J.mT @` \ `v_des[..., None]`], shape: rect),
  edge((1, 1), (2, 0), "->"),
  edge((0, 0), (1, 1), "->"),
))

#table(
  columns: (5.2cm, 6cm, 4.8cm),
  stroke: 0.5pt + fg, inset: 6pt,
  [*API*], [*does*], [*shapes*],
  [#api("J.mT @ J")], [$J^T J$ for every env], [`(N,3,3)`],
  [#api("lam**2 * torch.eye(3)")], [adds to the diagonal; broadcasts], [`(3,3)` onto `(N,3,3)`],
  [#api("v_des[..., None]")], [makes $v_"des"$ a column], [`(N,3)` $arrow.r$ `(N,3,1)`],
  [#api("torch.linalg.solve(A, b)")], [solves $A x = b$ per env], [`(N,3,1)`],
  [#api("x[..., 0]")], [drops the column axis], [`(N,3)`],
)

== Check

At $q = (0, 0.6, -1.2)$:

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt + fg, inset: 6pt,
  [*$v_"des"$*], [*$dot(q)$*], [*$J dot(q)$*],
  [$(0, 0.1, 0)$], [$(0.151, 0, 0)$], [$(0, 0.099, 0)$],
  [$(0.1, 0, 0)$], [$(0, -0.207, 0.417)$], [$(0.094, 0, 0.001)$],
)
] // claude:end
