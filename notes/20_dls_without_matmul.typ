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

= Replacing `J.mT @ J` in `dls`

== What we decided

- The profile pointed at one line: `J.mT @ J` in `dls`, a batch of 16384 products of two 3×3 matrices. The GPU's matrix-multiply library is built for big matrices, so it handles these tiny ones badly.
- Write the same product as a multiply and a sum instead, with broadcasting. Compile can fuse those into one small kernel.
- Change only that line in `kin.py`. The matrix-times-vector `@` in `dls` stays, and so does the `lam**2 * eye` term.
- The new line must give the same numbers as the old one.

== The idea

Symbols:
- $n$: which env (one of 16384)
- $k$: the row of $J$. This is the index that gets added up.
- $i, j$: the column of $J$. They become the row and column of the result.

$ (J^top J)_(i j) = sum_k J_(k i) dot J_(k j) $

So it is: for every pair of columns $(i, j)$, multiply the two columns entry by entry down the rows $k$, then add up. In shapes:

#align(center, diagram(
  node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg, spacing: (0.8cm, 1.3cm),
  nd((0, 0), c-cpu, [`J` \ $(n, k, i)$ \ `(N, 3, 3)`], <j>),
  nd((1, -0.7), c-new, [view A \ $(n, k, i, 1)$ \ `(N, 3, 3, 1)`], <a>),
  nd((1, 0.7), c-new, [view B \ $(n, k, 1, j)$ \ `(N, 3, 1, 3)`], <b>),
  nd((2, 0), c-gpu, [product \ $(n, k, i, j)$ \ `(N, 3, 3, 3)`], <p>),
  nd((3, 0), c-flag, [sum over $k$ \ $(n, i, j)$ \ `(N, 3, 3)`], <s>),
  edge(<j>, <a>, "->"), edge(<j>, <b>, "->"), edge(<a>, <p>, "->"), edge(<b>, <p>, "->"), edge(<p>, <s>, "->"),
))

Both views come from the same `J`. They differ only in where the column axis sits: in A it is $i$ (axis 2) and in B it is $j$ (axis 3). Broadcasting then pairs every $i$ with every $j$ for the same $n$ and $k$.

== Why rows give $J^top J$, and columns would give $J J^top$

A small $J$ with numbers, so you can check it by hand. Each coloured block is one *table* (a column times a row, entry by entry). Top: one table per *row* of $J$, then add. Bottom: one table per *column* of $J$, then add.

#let M(c, b) = text(fill: c, $display(#b)$)
#let J-rows = math.mat(
  (text(fill: c-cpu, $1$), text(fill: c-cpu, $2$)),
  (text(fill: c-new, $3$), text(fill: c-new, $4$)),
)
#let J-cols = math.mat(
  (text(fill: c-gpu, $1$), text(fill: c-flag, $2$)),
  (text(fill: c-gpu, $3$), text(fill: c-flag, $4$)),
)

#align(center)[
  #grid(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    column-gutter: 7pt, row-gutter: 14pt, align: horizon + center,
    M(fg, J-rows), [$arrow.r$],
    M(c-cpu, math.mat((1, 2), (2, 4))), [$+$],
    M(c-new, math.mat((9, 12), (12, 16))), [$=$],
    M(c-flag, math.mat((10, 14), (14, 20))),
    text(size: 9pt, fill: fg)[row 1 of $J$ is $(1, 2)$], [],
    text(size: 9pt, fill: c-cpu)[$vec(1, 2) mat(1, 2)$], [],
    text(size: 9pt, fill: c-new)[row 2 is $(3, 4)$: $vec(3, 4) mat(3, 4)$], [],
    text(size: 9pt, fill: fg)[$= J^top J$],
  )
  #v(14pt)
  #grid(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    column-gutter: 7pt, row-gutter: 14pt, align: horizon + center,
    M(fg, J-cols), [$arrow.r$],
    M(c-gpu, math.mat((1, 3), (3, 9))), [$+$],
    M(c-flag, math.mat((4, 8), (8, 16))), [$=$],
    M(fg, math.mat((5, 11), (11, 25))),
    text(size: 9pt, fill: fg)[column 1 is $(1, 3)$], [],
    text(size: 9pt, fill: c-gpu)[$vec(1, 3) mat(1, 3)$], [],
    text(size: 9pt, fill: c-flag)[column 2 is $(2, 4)$: $vec(2, 4) mat(2, 4)$], [],
    text(size: 9pt, fill: fg)[$= J J^top$],
  )
]

The two results are different, so the choice matters:
- *Top, $J^top J$*: entry $(i, j)$ pairs *column* $i$ with *column* $j$, adding down the rows. So you add up one table per *row*: the row index is the one that gets summed away, and the column index is what is left.
- *Bottom, $J J^top$*: entry $(a, b)$ pairs *row* $a$ with *row* $b$, adding across the columns. So you add up one table per *column*, and the column index is the one summed away.

In your `J` of shape `(N, 3, 3)`, axis 1 is the row axis. For $J^top J$ you sum axis 1 and keep axis 2 (columns) as the table's two axes. That is why both views keep the row axis in the same place and only move the column axis, as in the shape diagram above.

== The spec

#table(
  columns: (3.2cm, 1fr),
  stroke: 0.5pt + fg, inset: 6pt, align: horizon,
  [*step*], [*what it does*],
  [two views of `J`], [insert a new axis of size 1 with `None` (you're solid on this): one view puts it last, the other puts it just before the last],
  [multiply], [the two views broadcast against each other. Size-1 axes stretch to size 3, giving `(N, 3, 3, 3)`],
  [sum], [`.sum(dim=...)` over the axis that is $k$. That axis disappears (no `keepdim`), leaving `(N, 3, 3)`],
  [the rest of `dls`], [unchanged: add `lam**2 * eye`, solve, as before],
)

The `(N, 3, 3, 3)` in the middle is 27 numbers per env, small, and compile is likely to fuse it away so it never exists as a full tensor.

#note(c-warn)[The trap][
Summing the wrong axis gives $J J^top$ instead of $J^top J$. Both are 3×3 and symmetric, so nothing crashes and the shapes still print right. Only the numbers are wrong. The comparison against the old line is what catches it.
]

== Where to try it

Before touching `kin.py`, test the idea in a scratch script on a random `J` of shape `(N, 3, 3)`, float32, on `cuda`. Compare it with `J.mT @ J` there. When they match, change the line in `dls`.

#note(c-flag)[Look at][
- the largest absolute difference between the new version and `J.mT @ J` on the same random `J`. Target: around 1e-6 (float rounding).
- `profile_expert.py` again: the biggest kernel, the kernel count, and the total GPU time, eager and compiled.
- the `Timer` number for compiled `act` against the 0.44 ms from before.
- `run_expert.py` still prints success 1.0 at steps 599, 1199, 1799.
]
] // claude:end
