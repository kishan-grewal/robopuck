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
= robopuck notes

Finished teaching pages, in the order they were written. Each is a `.typ` source and its `.pdf`. `book.pdf` is all of them joined, in this order.

#table(
  columns: (1.3cm, 6.2cm, 1fr),
  stroke: 0.5pt + fg, inset: 5pt, align: horizon,
  [*no.*], [*file*], [*what*],
  [01], [`01_scene_and_physics`], [the scene, reach, contact, symplectic Euler and friction],
  [02], [`02_sim_step`], [from Sim(x, v) to Sim.step: forces and one control step],
  [03], [`03_contact_force`], [the contact force from scratch: touching, which way, how hard],
  [04], [`04_expert_jacobian_and_dls`], [expert 1: the Jacobian, jacrev and vmap, damped least squares],
  [05], [`05_expert_target_to_velocity`], [expert 2: a target position becomes a wanted velocity],
  [06], [`06_expert_where_to_park`], [expert 3: where to park the hand],
  [07], [`07_expert_park_then_drop`], [expert 4: park, then drop],
  [08], [`08_expert_push`], [expert 5: push],
  [09], [`09_expert_coin_and_route`], [expert 6: a coin flip and a route],
  [10], [`10_expert_next_waypoint`], [expert 7: move on to the next waypoint],
  [11], [`11_expert_lift_and_repark`], [expert 8: lift and park again],
  [12], [`12_expert_goal_check`], [expert check: who reaches the goal, when, and how many lifts],
  [13], [`13_check_the_expert`], [step 1: the five summary lines for the expert],
  [14], [`14_episodes`], [step 2: Sim owns t and episode; step returns obs, done, success],
  [15], [`15_why_a_step_is_slow`], [step 3 theory: eager, torch.compile, CUDA graphs],
  [16], [`16_compile_the_slow_step`], [step 3 task sheet: wrap step and act, logs, the ladder],
  [17], [`17_context_managers`], [ways to write a context manager (generator, class, ready-made)],
  [18], [`18_timer`], [the Timer: a with block that times the GPU],
  [19], [`19_profile_act`], [profiling act: where the GPU time goes],
  [20], [`20_dls_without_matmul`], [replacing J.mT @ J in dls with a broadcast sum],
  [21], [`21_demo_buffer`], [step 4: the demo Buffer, recording obs and actions],
  [22], [`22_demos`], [step 5a: finish returns a Demos],
  [23], [`23_demos_sample`], [step 5b: Demos.sample with the pad mask],
  [24], [`24_free_the_buffer`], [step 5c: del the buffer and check memory],
  [25], [`25_normalizer`], [step 6a: the Normalizer module with buffers],
  [26], [`26_mse_policy`], [step 6b: MSEPolicy, forward, act, masked loss],
  [27], [`27_training_loop`], [step 6c: the training loop],
  [28], [`28_contact_findings`], [step 6e: where the hand first touches the puck, expert vs MSE, measured, plus what to add for flow],
)
] // claude:end
