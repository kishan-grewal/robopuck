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

#claude[ // claude:begin
= the contact force, from scratch

*Symbols.* $p = (p_x, p_y, p_z)$ end-effector (EE) center, $p_"xy"$ its horizontal part, $p_z$ its height. $dot(p)$ EE velocity. $x in RR^2$ puck center, $v in RR^2$ puck velocity. $r_e = 0.02$ EE radius, $r_p = 0.05$ puck radius, $h = 0.03$ puck height. $k = 2000$ spring, $c = 10$ damper. $epsilon = 0.001$.

== The one idea

The EE is a ball, the puck is a short disk. They are not allowed to overlap, but the sim has no hard walls, so overlap is allowed and _punished_: a stiff spring pushes the puck away, with a damper so it doesn't bounce. Finding the force is answering three questions, and each answer is one variable:

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.5pt + fg, inset: 6pt,
  [*question*], [*answer*], [*variable*],
  [are they touching, and how much?], [overlap of the two circles seen from above, and is the ball low enough], [depth $d$, switch $"active"$],
  [which way does the puck get pushed?], [straight away from the ball's center], [direction $n$],
  [how hard?], [spring on the overlap plus damper on the closing speed], [$k d + c s$],
)

== Question 1: touching?

*Top view.* Two circles. They touch when the distance between their centers is less than the sum of their radii. How far inside each other they are is the depth $d$.

#align(center, canvas(length: 1.2cm, {
  import draw: *
  set-style(stroke: fg)
  circle((3.2, 0), radius: 2.5, stroke: 1pt + blue, fill: blue.lighten(75%))
  circle((0, 0), radius: 1, stroke: 1.4pt + fg)
  circle((0, 0), radius: 0.05, fill: fg)
  circle((3.2, 0), radius: 0.05, fill: fg)
  content((-0.9, -1.45), text(size: 8pt)[EE, radius $r_e$, center $p_"xy"$])
  content((3.2, -1.7), text(size: 8pt, fill: black)[puck, radius $r_p$, center $x$])
  line((0, 0.0), (3.2, 0.0), stroke: 0.8pt + black, mark: (end: ">", scale: 0.5))
  content((2.1, 0.3), text(size: 8pt, fill: black)[`dist`])
  line((0.7, 0.9), (1.0, 0.9), stroke: 0.6pt + fg, mark: (start: ">", end: ">", scale: 0.4))
  line((0.7, 0.9), (0.7, 0.1), stroke: (dash: "dotted", paint: gray, thickness: 0.4pt))
  line((1.0, 0.9), (1.0, 0.1), stroke: (dash: "dotted", paint: gray, thickness: 0.4pt))
  content((0.85, 1.2), text(size: 9pt)[$d$])
}))

$ "dist" = norm(x - p_"xy"), quad d = max(0, thin r_p + r_e - "dist") $

$"dist"$ is the center-to-center length. If it is bigger than $r_p + r_e$ the circles are apart and $d = 0$. The picture exaggerates $d$; in the sim it is a few millimetres.

*Side view.* The puck is only $h = 0.03$ tall, so a ball that is high enough passes over it. The ball's lowest point is $p_z - r_e$.

#align(center, canvas(length: 1.5cm, {
  import draw: *
  set-style(stroke: fg)
  line((-4, 0), (6, 0), stroke: 0.8pt)
  rect((0, 0), (5, 1.5), stroke: 1pt + blue, fill: blue.lighten(75%))
  content((2.5, 0.75), text(size: 8pt, fill: black)[puck, height $h$])
  circle((-0.6, 1.0), radius: 1, stroke: 1pt + fg)
  content((-2.2, 1.0), text(size: 8pt)[low: hits])
  circle((2.5, 3.2), radius: 1, stroke: 1pt + fg)
  content((4.5, 3.2), text(size: 8pt)[high: passes over])
  line((-5, 1.5), (6.2, 1.5), stroke: (dash: "dashed", paint: gray, thickness: 0.5pt))
  content((-4.0, 1.7), text(size: 8pt)[top of puck])
  line((2.5, 2.2), (2.5, 1.5), stroke: 0.6pt + fg, mark: (start: ">", end: ">", scale: 0.4))
  content((3.0, 1.85), text(size: 8pt)[gap])
}))

$ "active" = [thin p_z - r_e < h thin] $

$[dot]$ is 1 if true, else 0. The low ball has its lowest point under the dashed line, so it can hit; the high ball clears it.

== Question 2: which way?

The force on the puck points from the EE center toward the puck center. Take the vector between them and divide by its length so only the direction is left:

$ n = (x - p_"xy") / ("dist" + epsilon) $

$n$ is a unit vector, length 1. $epsilon$ only stops a divide by zero when the centers coincide.

== Question 3: how hard?

*Spring.* Pushing harder the deeper the overlap: $k d$. Zero when $d = 0$.

*Damper.* Without it the puck bounces off like a rubber ball. It needs a force that resists the closing speed. Relative velocity of the puck as seen from the ball is $v - dot(p)_"xy"$. Its component along $n$ (the direction from ball to puck) is how fast they separate. We want how fast they _close_, so flip the sign:

$ s = -(v - dot(p)_"xy") dot n $

$dot$ is the dot product: multiply matching components and add. It picks out how much of a vector lies along $n$. $s > 0$ means they are approaching, $s < 0$ means separating.

Put together, and then clamp so contact can never _pull_ (when the puck leaves fast, $s$ is very negative and could flip the sign):

$ F_c = max(0, thin k d + c s) thin n thin "active" $

== Every variable, as code

#table(
  columns: (auto, 1fr, auto, auto),
  stroke: 0.5pt + fg, inset: 6pt,
  [*math*], [*meaning*], [*in code*], [*shape*],
  [$p_"xy"$], [EE center, horizontal], [`p[:, :2]`], [`(N,2)`],
  [$p_z$], [EE center, height], [`p[:, 2:]`], [`(N,1)`],
  [$dot(p)_"xy"$], [EE velocity, horizontal], [`pdot[:, :2]`], [`(N,2)`],
  [$x$, $v$], [puck position, velocity], [`self.x`, `self.v`], [`(N,2)`],
  [$r_p, r_e, h$], [radii and height], [`self.r_p`, `self.r_e`, `self.h`], [floats],
  [$"dist"$], [center-to-center length], [`rel.norm(dim=-1, keepdim=True)`], [`(N,1)`],
  [$d$], [overlap depth], [`(r_p + r_e - dist).clamp(min=0)`], [`(N,1)`],
  [$n$], [unit push direction], [`rel / (dist + eps)`], [`(N,2)`],
  [$s$], [closing speed], [`-((v - pdot_xy) * n).sum(-1, keepdim=True)`], [`(N,1)`],
  [$"active"$], [ball low enough], [`p_z - r_e < h`], [`(N,1)` bool],
  [$F_c$], [force on the puck], [the final line], [`(N,2)`],
)

Define `rel = x - p[:, :2]` first: it feeds `dist` and `n`. You need one new field, `r_e: float = 0.02`.

== A worked case

Puck at rest at $(0.2, 0)$. EE at $p = (0.14, 0, 0.02)$ moving $dot(p) = (0.3, 0, 0)$ toward it.

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt + fg, inset: 6pt,
  [*step*], [*value*], [*how*],
  [`rel`], [$(0.06, 0)$], [$x - p_"xy" = (0.2 - 0.14, 0 - 0)$],
  [`dist`], [$0.06$], [length of `rel`],
  [$d$], [$0.01$], [$0.05 + 0.02 - 0.06$],
  [$n$], [$(1, 0)$], [`rel` over its length: pointing from EE toward puck],
  [$s$], [$0.3$], [$v - dot(p)_"xy" = (-0.3, 0)$, dot $n$ is $-0.3$, flip sign],
  [$k d$, $c s$], [$20$, $3$], [$2000 dot 0.01$, $10 dot 0.3$],
  [$"active"$], [true], [$p_z - r_e = 0 < 0.03$],
  [$F_c$], [$(23, 0)$], [$(20 + 3) dot (1, 0)$],
)

With $p_z = 0.1$ instead: $p_z - r_e = 0.08$, not less than $0.03$, so $"active"$ is false and $F_c = (0, 0)$ exactly. These are your two checks.

== How to find a force like this yourself

Do not start from the formula. Ask in order: when does it act (a condition, becomes a mask), which way (a direction, becomes a unit vector), how much (a size that grows with the thing that is wrong, here overlap, plus something that resists motion, here the damper). Each question gives one tensor, and the force is their product.
] // claude:end
