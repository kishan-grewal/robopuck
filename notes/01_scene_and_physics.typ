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
= robopuck, part 1: scene and reach

*Symbols.* $q = (q_1, q_2, q_3)$ joint angles (yaw, pitch, pitch). $L = 0.4$ link length. $p in RR^3$ end-effector (EE) position. $x in RR^2$ puck center. $r_p = 0.05$ puck radius, $r_e = 0.02$ EE sphere radius. $norm(dot)$ Euclidean length.

== Top view

Base at the origin, puck at $x = 0.2$ (jitter box $plus.minus 2$ cm), pillar at $x = 0.4$, goal at $x = 0.6$. Black dots are the waypoints, dashed lines the two mirror-image routes. The dotted circle around the pillar has radius $0.1 + 0.05 = 0.15$: the puck _center_ must stay outside it. The thin arc is the reach limit.

#align(center, canvas(length: 11cm, {
  import draw: *
  set-style(stroke: fg)
  let reach = 0.693
  arc((reach * calc.cos(-40deg), reach * calc.sin(-40deg)), start: -40deg, stop: 40deg, radius: reach, stroke: 0.4pt + gray)
  circle((0.4, 0), radius: 0.15, stroke: (dash: "dotted", paint: gray, thickness: 0.6pt))
  circle((0.4, 0), radius: 0.1, fill: if dark { gray.darken(40%) } else { gray.lighten(50%) }, stroke: none)
  rect((0.18, -0.02), (0.22, 0.02), stroke: 0.4pt + blue)
  circle((0.2, 0), radius: 0.05, fill: blue.lighten(60%), stroke: none)
  circle((0.6, 0), radius: 0.03, stroke: green.darken(20%) + 1pt)
  circle((0, 0), radius: 0.015, fill: fg)
  for s in (1, -1) {
    line((0.2, 0), (0.3, s * 0.2), (0.5, s * 0.2), (0.6, 0), stroke: (dash: "dashed", thickness: 0.6pt))
    circle((0.3, s * 0.2), radius: 0.008, fill: fg)
    circle((0.5, s * 0.2), radius: 0.008, fill: fg)
  }
  content((0, -0.05), text(size: 8pt)[base])
  content((0.2, -0.1), text(size: 8pt)[puck])
  content((0.4, 0), text(size: 8pt)[pillar])
  content((0.6, -0.07), text(size: 8pt)[goal])
  content((0.3, 0.24), text(size: 8pt)[$w_1$])
  content((0.5, 0.24), text(size: 8pt)[$w_2$])
  line((-0.02, 0), (0.7, 0), stroke: 0.3pt + gray)
}))

== Side view and reach

Shoulder at height $L$. The grey line is the fully extended chain, length $2 L$, from shoulder to the table. Blue is one bent configuration reaching $r = 0.5$.

#align(center, canvas(length: 11cm, {
  import draw: *
  set-style(stroke: fg)
  let S = (0, 0.4)
  let reach = calc.sqrt(3 * 0.16)
  let T = (0.5, 0)
  let d = calc.sqrt(0.25 + 0.16)
  let h = calc.sqrt(0.16 - d * d / 4)
  let ux = 0.5 / d
  let uy = -0.4 / d
  let E = (0.25 - h * uy, 0.2 + h * ux)
  line((-0.1, 0), (0.78, 0), stroke: 0.6pt)
  line((0, 0), S, stroke: 2pt)
  line(S, (reach, 0), stroke: (dash: "dashed", paint: gray, thickness: 1pt))
  line(S, E, T, stroke: 2pt + blue)
  circle(S, radius: 0.012, fill: bg)
  circle(E, radius: 0.012, fill: bg)
  circle(T, radius: 0.012, fill: blue)
  content((0.05, 0.2), text(size: 8pt)[$L$])
  content((E.at(0) - 0.04, E.at(1) + 0.03), text(size: 8pt)[$q_3$])
  content((0.04, 0.44), text(size: 8pt)[$q_2$])
  content((reach, -0.04), text(size: 8pt)[$r = L sqrt(3) = 0.693$])
}))

Fully extended: shoulder to table-contact is a straight line of length $2 L$ with vertical drop $L$, so
$ sqrt(r^2 + L^2) <= 2 L quad arrow.r.double quad r <= L sqrt(3) = 0.693 "m" $
One forward-kinematics convention (the text doesn't fix one), $q_2$ measured from vertical:
$ rho = L (sin q_2 + sin(q_2 + q_3)), quad z = L + L (cos q_2 + cos(q_2 + q_3)), quad p = vec(rho cos q_1, rho sin q_1, z) $
$q_1$ rotates the whole picture about the vertical axis; $q_2, q_3$ set the radius $rho$ and height $z$ in the vertical plane.

== Kinematic arm

"Kinematic" means the arm has no dynamics: the commanded velocity _is_ the velocity.
$ q_(t+1) = q_t + Delta t_"ctrl" "clip"(u_t, -2, 2), quad Delta t_"ctrl" = 0.02 $
$"clip"$ clamps each entry to $[-2, 2]$ rad/s. The arm is an infinitely stiff, infinitely heavy pusher: only the puck has dynamics.
] // claude:end

#claude[ // claude:begin
= robopuck, part 2: contact, integrator, friction

*Symbols.* $x, v$ puck position and velocity. $m = 0.1$ mass. $k = 2000$ spring, $c = 10$ damper. $Delta t = 0.005$ substep. $p_"xy"$ EE horizontal position. $F$ force on the puck.

== Penalty contact

Contact only if the sphere's lowest point is below the puck's top face: $p_z - r_e < 0.03$. Penetration depth and push direction:
$ d = max(0, thin r_p + r_e - norm(x - p_"xy")), quad n = (x - p_"xy") / norm(x - p_"xy") $
$max(0, dot)$ is $0$ when not touching; $n$ is the unit vector from EE to puck. Closing speed (positive when EE and puck approach):
$ s = -(v - dot(p)_"xy") dot n, quad F_c = (k d + c thin s) n $
Without $c$ it is a pure spring: the puck bounces off elastically. The damper removes energy: $zeta = c / (2 sqrt(k m)) = 0.35$. A real implementation should keep $F_c dot n >= 0$ so contact never pulls. The pillar is the same spring ($p_"xy" arrow.r$ pillar center, $r_e arrow.r 0.1$), no damper listed.

== Symplectic Euler and the $k$ limit

Update velocity first, then position with the _new_ velocity:
$ v_(n+1) = v_n + Delta t thin F(x_n, v_n) / m, quad x_(n+1) = x_n + Delta t thin v_(n+1) $
For a pure spring, $omega = sqrt(k/m)$ and $a = omega Delta t$:
$ vec(x, v)_(n+1) = mat(1, Delta t; -omega^2 Delta t, 1 - a^2) vec(x, v)_n, quad det = 1, quad "tr" = 2 - a^2 $
Eigenvalues solve $lambda^2 - "tr" thin lambda + 1 = 0$. They lie on the unit circle iff $|"tr"| <= 2$, i.e. $a <= 2$; beyond that one eigenvalue has $|lambda| > 1$ and the simulation blows up. Ours: $a = 0.71$. The cap is $k_"max" = m (2 slash Delta t)^2 = 16000$ N/m.

#let rho(a) = {
  let tr = 2 - a * a
  let disc = tr * tr - 4
  if disc <= 0 { 1 } else { (calc.abs(tr) + calc.sqrt(disc)) / 2 }
}
#let avals = lq.linspace(0, 3, num: 150)
#align(center, lq.diagram(
  width: 9cm, height: 4.5cm,
  xlabel: $a = omega Delta t$, ylabel: [largest $|lambda|$],
  ylim: (0.9, 2.2),
  lq.plot(avals, avals.map(rho), mark: none),
  lq.plot((0.71, 0.71), (0.9, 2.2), mark: none, stroke: (dash: "dashed")),
))

== Friction and the stiffness trap

$ F_f = -mu m g thin v / (norm(v) + epsilon), quad mu = 0.3, thin epsilon = 0.001 $
Fast ($norm(v) >> epsilon$): magnitude $mu m g$, Coulomb friction. Slow: $F_f approx -(mu m g slash epsilon) v$, a linear drag with rate $lambda = mu g slash epsilon = 2943 "s"^(-1)$. Explicit integration of $dot(v) = -lambda v$ needs $lambda Delta t < 2$, but $lambda Delta t = 14.7$: one substep multiplies a small $v$ by $-13.7$. The puck never settles: speed falls to zero and then alternates $plus.minus w$ every substep. A 2-cycle $v arrow.r -v$ needs $2 w = mu g Delta t thin w / (w + epsilon)$, so $w = (mu g Delta t - 2 epsilon) / 2 = 0.0064$ m/s. Position jitter is about $3 times 10^(-5)$ m per substep, harmless against the 3 cm tolerance.

#let eps = 0.001
#let vs = lq.linspace(-0.01, 0.01, num: 201)
#let chat = {
  let v = 0.05
  let out = (v,)
  for i in range(40) {
    v = v - 0.005 * 0.3 * 9.81 * v / (calc.abs(v) + eps)
    out.push(v)
  }
  out
}
#grid(columns: 2, gutter: 8pt,
  lq.diagram(
    width: 6.5cm, height: 4cm,
    xlabel: $v$, ylabel: $F_f / (mu m g)$,
    lq.plot(vs, vs.map(v => -v / (calc.abs(v) + eps)), mark: none),
    lq.plot(vs, vs.map(v => if v > 0 { -1 } else if v < 0 { 1 } else { 0 }), mark: none, stroke: (dash: "dashed")),
  ),
  lq.diagram(
    width: 6.5cm, height: 4cm,
    xlabel: [substep], ylabel: $v$,
    lq.plot(range(41), chat, mark: none),
  ),
)

Left: smoothed friction (solid) against sign function (dashed). Right: speed over 40 substeps from $0.05$ m/s: it crosses zero and flips sign every step.

== Vectorized-sim shape rules

Every state has a leading env dimension: `x` is `(N, 2)`, `q` is `(N, 3)`, `step` is an int tensor `(N,)`. Two idioms do most of the work:
- _Branching_ becomes arithmetic with masks: $max(0, dot)$ is `.clamp(min=0)`; "only if the sphere is low" is `active = (p_z - r_e < 0.03)` then `F = F * active[:, None]`.
- _Resets_ keep shapes fixed: `x = torch.where(done[:, None], x0, x)`. `done[:, None]` is `(N, 1)` and broadcasts against `(N, 2)`. Every env pays for both branches; no variable-length tensors.

== Why `torch.compile` with CUDA graphs

At $N = 16384$ each op is tiny, so the GPU idles while Python launches the next kernel (about 10 $mu$s each, hundreds per step). Inductor _fuses_ chains of elementwise ops into one kernel; `reduce-overhead` records the kernel sequence once as a CUDA graph and replays it with a single launch.
- A replay writes into the _same_ output memory. Keep `x` from call $n$, call again, and call $n+1$ overwrites it: `.clone()` anything held past the next call.
- A graph is recorded for specific shapes and Python-number values. A Python `int` step counter that changes each call looks new every time and recompiles. As a tensor, its value changes but the program doesn't.
] // claude:end

#claude[ // claude:begin
= robopuck, part 3: expert controller

*Symbols.* $J(q) = (partial p) slash (partial q) in RR^(3 times 3)$ Jacobian. $e = p^* - p$ error. $v_"des"$ desired EE velocity. $dot(q)$ joint velocity. $lambda = 0.05$ damping. $sigma_i$ singular values of $J$.

== Jacobian for all envs at once

$J_(i j) = partial p_i / partial q_j$: nudge joint $j$, how does EE coordinate $i$ move? Chain rule: $dot(p) = J dot(q)$.

Code: write `fk(q)` for _one_ `q` of shape `(3,)`. `jacrev(fk)` returns the `(3, 3)` Jacobian (reverse-mode autodiff, one backward pass per output row); `vmap(jacrev(fk))` maps it over the leading dimension, giving `(N, 3, 3)`. `vmap` is not a loop: it rewrites each op in `fk` into its batched version.

== Damped least squares

We want $J dot(q) = v_"des"$. Near full extension $J$ loses rank and $J^(-1)$ sends joint speeds to infinity. Trade tracking error against joint speed:
$ min_(dot(q)) thin norm(J dot(q) - v_"des")^2 + lambda^2 norm(dot(q))^2 $
Gradient zero: $2 J^T (J dot(q) - v_"des") + 2 lambda^2 dot(q) = 0$, so
$ dot(q) = (J^T J + lambda^2 I)^(-1) J^T v_"des" $
(`torch.linalg.solve`, never form the inverse.) With $J = sum_i sigma_i u_i v_i^T$ ($u_i$ output-space unit vectors, $v_i$ joint-space unit vectors):
$ dot(q) = sum_i sigma_i / (sigma_i^2 + lambda^2) (u_i^T v_"des") thin v_i quad "vs. exact:" quad sum_i 1 / sigma_i (u_i^T v_"des") thin v_i $

#let sig = lq.linspace(0.01, 1, num: 200)
#align(center, lq.diagram(
  width: 9cm, height: 4.5cm,
  xlabel: $sigma_i$, ylabel: [gain on direction $i$],
  ylim: (0, 40),
  lq.plot(sig, sig.map(s => 1 / s), mark: none, stroke: (dash: "dashed"), label: [exact $1 / sigma$]),
  lq.plot(sig, sig.map(s => s / (s * s + 0.0025)), mark: none, label: [damped]),
))

The damped gain peaks at $sigma_i = lambda$ with value $1 / (2 lambda) = 10$ rad/s per m/s: the worst-case joint speed per unit of requested EE speed.

== Capped proportional controller

$ v_"des" = K e thin min(1, thin v_max / (K norm(e))), quad v_max = 0.3 "m/s" $
Small error: $v_"des" = K e$. Large error: same direction, length capped. `min(1, ...)` is the vectorized `if`.

== Route logic

Per env: side $s in {+1, -1}$ (the coin) and waypoint index $w in {0, 1, 2}$ (see top view in part 1: $(0.3, 0.2 s)$, $(0.5, 0.2 s)$, goal $(0.6, 0)$). Advance when the puck's $x$ passes the waypoint's: $w arrow.l w + [x_1 > W_(w,1)]$ ($[dot]$ is 1 if true, else 0).

To push toward $W$, stand behind the puck: with $hat(u) = (W - x) / norm(W - x)$,
$ p^*_"park" = x - 0.08 thin hat(u) $
Travel at height $0.08$ (above puck top $0.03$ and sphere $0.02$). Phases, all boolean masks evaluated every step: _travel_ (go to $p^*_"park"$ at $z = 0.08$), _drop_ (once close horizontally, target $z$ goes to the table), _push_ ($p^*_"park"$ plus $0.15 thin hat(u)$), _lift_ if $norm(p_"xy" - p^*_"park") > 0.03$.

Clearance: the waypoint segment is $0.2$ from the pillar center, minus $0.15$, gives $0.05$ m, slightly more than the text's "about 3 cm". The coin is what makes demos bimodal; the second mode is the first with $q_1 arrow.r -q_1$.
] // claude:end

#claude[ // claude:begin
= robopuck, part 4: data

*Symbols.* $T = 400$ steps. $H = 16$ chunk length. $o in RR^7$ obs. $a in RR^3$ action. $A = (a_s, dots, a_(s+H-1)) in RR^48$ chunk from step $s$. $mu, sigma$ per-dimension mean and std.

== Observation leak

$o = (q, x, v)$: $3 + 2 + 2$. If the waypoints or the coin were in $o$, the network could read the side off its input, $p(A | o)$ would have one mode, and MSE would work. The experiment needs $p(A | o)$ bimodal. "Audit" means: confirm the tensor has exactly these 7 columns and none derives from the coin. At the start of an episode all envs have nearly the same $o$, and half go left, half right.

== The buffer

Preallocated `obs: (T, N, 7)`, `act: (T, N, 3)`, filled one control step at a time. Size: $T N (7 + 3) times 4 "bytes" = 262$ MB. Failed episodes are dropped by a boolean mask `good: (N,)`; `good.nonzero()` gives env indices to sample from.

== Chunk sampling by indexing

Sample start $s$ uniformly from $0, dots, T - H = 384$ (385 values; the last chunk ends at step $399$) and env $e$ from the good ones. Index tensor `s[:, None] + arange(H)` has shape `(B, H)`.
- `obs[s, e]`: `(B, 7)`.
- `act[s[:, None] + arange(H), e[:, None]]`: the two index tensors broadcast to `(B, H)`, the trailing 3 rides along, giving `(B, H, 3)`; `.flatten(1)` gives `(B, 48)`.

No `DataLoader`: the data are already one GPU tensor, and a `DataLoader` would add worker processes and a CPU to GPU copy per batch.

== Standardization

$ hat(o)_j = (o_j - mu_j) / (sigma_j + 10^(-6)), quad hat(A)_j = (A_j - mu^A_j) / (sigma^A_j + 10^(-6)) $
The 7 obs dimensions have different units and magnitudes; plain MSE would be dominated by the numerically largest. `register_buffer` makes $mu, sigma$ part of `state_dict()` and moves them with `.to(device)`, but they aren't parameters, so the optimizer ignores them. The policy un-normalizes its output before it goes to the sim.
] // claude:end

#claude[ // claude:begin
= robopuck, part 5: MSE regresses to the mean

*Symbols.* $f_theta$ network. $A$ target chunk. $EE$ average over the data. $S = "diag"(-1, 1, 1)$ mirror map on joint velocities. $c$ the network's output at one fixed $o$.

== Mean minimizes squared error

Fix $o$; the network outputs one vector $c$. Loss: $EE[norm(A - c)^2 | o]$. With $overline(A) = EE[A | o]$:
$ EE[norm(A - c)^2] = norm(c - overline(A))^2 + EE[norm(A - overline(A))^2] $
(cross term $2 (c - overline(A))^T EE[A - overline(A)] = 0$; second term has no $c$.) Best answer: $c = overline(A)$, whatever the distribution looks like.

One coordinate with two modes $plus.minus 1$: the loss against each mode alone is $(c - 1)^2$ or $(c + 1)^2$; the average is $c^2 + 1$.

#let cs = lq.linspace(-2, 2, num: 100)
#align(center, lq.diagram(
  width: 9cm, height: 4.5cm,
  xlabel: $c$, ylabel: [loss],
  lq.plot(cs, cs.map(c => calc.pow(c - 1, 2)), mark: none, stroke: (dash: "dashed"), label: [mode $+1$]),
  lq.plot(cs, cs.map(c => calc.pow(c + 1, 2)), mark: none, stroke: (dash: "dashed"), label: [mode $-1$]),
  lq.plot(cs, cs.map(c => c * c + 1), mark: none, label: [average]),
))

The minimum of the average is at $c = 0$, between the modes. Here the modes are mirror images, $A_minus = S A_plus$:
$ overline(A) = 1/2 (I + S) A_plus = (0, thin a_2, thin a_3, dots) $
The yaw component is exactly zero at every step: the policy pushes straight along the axis into the pillar and stalls. The loss curve looks healthy because $EE[norm(A - overline(A))^2]$ is a non-zero floor that SGD reaches smoothly. Loss cannot tell you the answer is wrong; pillar-contact fraction can.

== Training pieces

*AdamW:* Adam with decoupled weight decay, $theta arrow.l theta - eta lambda_w theta$, separate from the gradient step.

*EMA:* $overline(theta) arrow.l overline(theta) + (1 - beta)(theta - overline(theta))$, $beta = 0.999$. That is `lerp(ema, params, weight = 1 - beta)`: the `_foreach_lerp_` weight is $0.001$, not $0.999$. It averages roughly the last $1 / (1 - beta) = 1000$ steps. Roll out with these weights.

*bf16 autocast:* same 8 exponent bits as float32 (same range, no loss scaling) but 7 mantissa bits (2 to 3 decimal digits). Matmuls run in bf16; master weights, optimizer state and the loss stay float32.

*Chunked rollout:* predict $H = 16$ steps, execute the first 8, re-observe, re-predict. Open loop for 8 steps (0.16 s) smooths motion; replanning keeps feedback.
] // claude:end

#claude[ // claude:begin
= robopuck, part 6: flow policy

*Symbols.* $A_1 in RR^48$ data chunk. $A_0 tilde cal(N)(0, I)$ noise chunk. $t in [0, 1]$ time. $A_t$ point on the path. $u_theta (A_t, t, o)$ network output (a velocity).

== Path and target

$ A_t = (1 - t) A_0 + t A_1, quad dot(A)_t = (d A_t) / (d t) = A_1 - A_0 $
Sample $t tilde "Uniform"(0, 1)$, fresh $A_0$, $A_1$ from the buffer, and regress
$ cal(L) = EE norm(u_theta (A_t, t, o) - (A_1 - A_0))^2 $
(`F.mse_loss` against `A1 - A0`; network input is $A_t$ concatenated with $o$ and 16 time features.)

== Why this does not collapse to the mean

Same decomposition as part 5: the optimum is the conditional mean of the target,
$ u^*(A, t, o) = EE[A_1 - A_0 | A_t = A, o] $
but now it averages only over (noise, data) pairs whose straight line passes through the _point_ $A$ at time $t$. At $t = 0$ the point is pure noise and the average covers both modes; as $t$ grows $A_t$ sits nearer one mode's endpoint and the average narrows to that mode. The ODE $d A / d t = u^*$ transports $cal(N)(0, I)$ at $t = 0$ to $p(A | o)$ at $t = 1$ (continuity equation $partial_t p_t + nabla dot (p_t u^*) = 0$), so samples land _on_ the modes.

One coordinate with modes $a_1 = s in {plus.minus 1}$, probability $1/2$ each, and $epsilon tilde cal(N)(0, 1)$: $a = (1 - t) epsilon + t s$. Bayes on $s$ given $a$:
$ P(s = 1 | a) = sigma_"sig" (2 a t / (1 - t)^2) quad arrow.r.double quad EE[s | a] = tanh(a t / (1 - t)^2) $
($sigma_"sig"$ = logistic sigmoid.) The target is $s - epsilon = (s - a) / (1 - t)$, so
$ u^*(a, t) = (tanh(a t / (1 - t)^2) - a) / (1 - t) $

#let ustar(x, t) = (calc.tanh(t * x / calc.pow(1 - t, 2)) - x) / (1 - t)
#let traj(x0, K) = {
  let h = 1 / K
  let x = x0
  let xs = (x0,)
  for k in range(K) {
    x = x + h * ustar(x, k * h)
    xs.push(x)
  }
  xs
}
#let x0s = (-2.2, -1.5, -1, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 1, 1.5, 2.2)
#align(center, lq.diagram(
  width: 9cm, height: 5cm,
  xlabel: $t$, ylabel: $a$,
  ..x0s.map(x0 => lq.plot(lq.linspace(0, 1, num: 11), traj(x0, 10), mark: none, stroke: 0.7pt)),
))

Twelve noise samples integrated with the 10 Euler steps the policy uses (kinks are the steps). Trajectories start spread over the whole line and end exactly at $plus.minus 1$; none ends between.

== Sampling

Euler, $K = 10$ steps, $h = 0.1$:
$ A^((k+1)) = A^((k)) + h thin u_theta (A^((k)), t_k, o), quad t_k = k h, quad A^((0)) tilde cal(N)(0, I) $
Fresh noise for all $N$ envs in one batched tensor each replan.

*Time features:* $[sin(2 pi f_i t), cos(2 pi f_i t)]_(i=1..8)$, 16 numbers, so an MLP can use $t$ at several scales. Input size $48 + 7 + 16 = 71$, output $48$.

== Side switching and evaluation numbers

Fresh noise each replan is a new draw of the mode. Once the puck is 5 cm off axis the obs says which side, so the draw almost always agrees, but not always. Side switch: $|y|$ exceeds $0.05$, then $y$ crosses $0$ before $x = 0.5$.

Success rate over $N = 16384$: standard error $sqrt(p (1 - p) / N) = 0.0017$ at $p = 0.95$. Differences of 1% between policies are real; 0.1% is noise. Pillar-contact fraction: running OR `touched |= (x - c_P).norm(dim=-1) < 0.15` ($0.05 + 0.1$).
] // claude:end

#claude[ // claude:begin
= robopuck, part 7: program structure

Arrows carry tensors; the label is the shape. $N$ envs, $B$ batch, $H = 16$.

#align(center, diagram(
  node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg, node-inset: 6pt, spacing: (3.6cm, 2.4cm),
  node((0, 0), [`Sim`\ q, x, v, step], name: <sim>),
  node((1, 0), [`Expert`], name: <exp>),
  node((2, 0), [`DemoBuffer`\ $(T, N, dots)$], name: <buf>),
  node((0, 1), [`evaluate()`], name: <ev>),
  node((1, 1), [`Policy`\ MSE or flow], name: <pol>),
  node((2, 1), [`train()`], name: <trn>),
  edge(<sim>, <exp>, "->", [obs $(N,7)$], bend: 25deg),
  edge(<exp>, <sim>, "->", [action $(N,3)$], bend: 25deg),
  edge(<exp>, <buf>, "->", [obs, action]),
  edge(<buf>, <trn>, "->", [chunks $(B, 48)$], label-side: left),
  edge(<trn>, <pol>, "->", [EMA weights]),
  edge(<pol>, <ev>, "->", [`act(obs)`]),
  edge(<ev>, <sim>, "<->", [`reset`, `step`], label-side: left),
))

== Who owns what

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.5pt + fg, inset: 6pt,
  [*class*], [*owns (state)*], [*does*],
  [`Sim`], [`q, x, v, step`, each `(N, ...)`], [`reset(mask)`, `step(u)`, `obs()`],
  [`Expert`], [side, waypoint index, phase masks], [`act(sim) -> (N, 3)`],
  [`DemoBuffer`], [`obs (T,N,7)`, `act (T,N,3)`, `good (N,)`], [`record`, `stats`, `sample(B)`],
  [`Policy`], [trunk MLP, stat buffers, EMA copy], [`loss(obs, A)`, `act(obs)`],
  [`Config`], [hyperparameters], [nothing (a dataclass)],
)

== Skeletons

Signatures and one hint per body; `...` is where the code goes.

```python
@dataclass
class Sim:
    n: int = 16384
    def reset(self, mask: Tensor) -> None:
        ...  # hint: every state var gets torch.where(mask[:, None], fresh, old)
    def contact_force(self) -> Tensor:        # (N, 2) on the puck
        ...  # hint: depth.clamp(min=0), active mask, multiply, never `if`
    def step(self, u: Tensor) -> None:        # u: (N, 3)
        ...  # hint: clip, move arm, then 4 substeps: force -> v -> x (that order)
    def obs(self) -> Tensor:                  # (N, 7)
        ...

def fk(q: Tensor) -> Tensor:                  # q: (3,) -> p: (3,), ONE arm
    ...
jac = vmap(jacrev(fk))                        # (N, 3) -> (N, 3, 3)

class DemoBuffer:
    def sample(self, b: int, h: int = 16) -> tuple[Tensor, Tensor]:
        ...  # hint: s (b,), e (b,); index with s[:, None] + arange(h) and e[:, None]

class Policy(nn.Module):
    def __init__(self, cfg): ...  # hint: register_buffer for the 4 stat tensors
    def loss(self, obs, chunk) -> Tensor: ...
    @torch.no_grad()
    def act(self, obs) -> Tensor: ...         # (N, 16, 3), un-normalized
```

== Python features by piece

#table(
  columns: (auto, 1fr),
  stroke: 0.5pt + fg, inset: 6pt,
  [*piece*], [*features worth knowing*],
  [config], [`@dataclass`, `tyro.cli(TrainConfig)`, type hints as the CLI schema],
  [sim], [`torch.where`, `.clamp`, `norm(dim=-1, keepdim=True)`, broadcasting with `[:, None]`, `torch.compile(mode="reduce-overhead")`, `.clone()`],
  [expert], [`vmap`, `jacrev`, `torch.linalg.solve`, boolean mask indexing, `torch.minimum`],
  [buffer], [preallocated `torch.empty`, advanced indexing, `nonzero`, `register_buffer`],
  [policy], [`nn.Module`, `nn.Sequential`, `torch.autocast("cuda", torch.bfloat16)`, `torch._foreach_lerp_`, `state_dict`, `@torch.no_grad()`, `copy.deepcopy`],
  [eval/logging], [`torch.cuda.Event`, `.cpu()` only every few steps, `rerun` `rr.log`, `rr.save`],
)
] // claude:end
