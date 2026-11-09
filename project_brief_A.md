# Project brief A: robopuck

Part 1 builds the sim, the expert, and the MLP MSE vs flow policies. Part 2 swaps the MLP trunk for a hand-written transformer. Both live in this repo.

## Part 1: sim, expert, MLP policies

### Minimum

The minimum is done when the MSE policy stalls against the pillar on most of 16384 envs. Its loss curve should look healthy. The flow policy, trained on the same demos, should land near the expert's 95 percent.

#### Setup

Start with a uv project. torch is the core dependency. Rerun handles every visual, loss curves included. tyro turns a dataclass config into a CLI. On Linux the PyPI torch wheel already bundles CUDA.

```bash
uv init pillar && cd pillar
uv add torch rerun-sdk tyro
```

#### Scene

The arm base sits at the origin. The first link stands vertical, which puts the shoulder at z = 0.4 m. The base joint yaws. The other two pitch. Every link is 0.4 m, so table reach tops out near 0.69 m.

The puck is a disk of radius 0.05 m. It stands 0.03 m tall. Its mass is 0.1 kg. It slides on a table at z = 0. It never rotates. The end effector is a sphere of radius 0.02 m. Nothing else on the arm collides.

Everything sits on the x axis. The puck starts near x = 0.2 m, jittered up to 2 cm per axis. The pillar, radius 0.1 m, stands at x = 0.4 m. The goal sits at x = 0.6 m. The scene is mirror symmetric across the x axis, so neither detour is shorter.

Success means the puck center ends within 3 cm of the goal. Episodes last 400 control steps.

#### Sim

Every state tensor has shape (16384, ...), one row per env. No Python loops over envs. Resets use torch.where masks so shapes never change. The episode step counter is an int tensor too.

Actions are joint velocities at 50 Hz, clipped at 2 rad/s. The arm is kinematic. Joint angles integrate straight from the commanded velocities.

Contact uses a penalty spring with k = 2000 N/m. It acts while the sphere's lowest point sits below the puck's top face. Horizontal overlap sets the depth. A damper of 10 N s/m on the closing speed stops the puck bouncing off. The pillar repels the puck with the same spring.

Each substep lasts 5 ms, from 4 substeps per control step. Symplectic Euler stays stable while `sqrt(k/m) * dt < 2`. For this puck that caps k near 16000 N/m. Update velocity before position.

Friction is a smoothed Coulomb term with μ = 0.3. Its denominator adds eps = 0.001 m/s to the speed. That keeps the force finite at rest.

```python
f_fric = -mu * m * g * v / (v.norm(dim=-1, keepdim=True) + eps)
```

#### Rerun

Wire up Rerun before going further. Draw links as line strips. Log 64 envs every 5th step, since every .cpu() call forces a GPU sync. On a remote box, rr.save writes an .rrd file you open locally. Watch a random policy shove the puck to confirm contact works.

#### Compile

Wrap step() in torch.compile with `mode="reduce-overhead"`, which turns on CUDA graphs. Run with `TORCH_LOGS=graph_breaks,recompiles,perf_hints`. The last flag reports skipped CUDA graphs. Fix everything the logs report.

CUDA graphs reuse output memory. Each replay overwrites what the previous replay returned. Clone any state you hold past the next call. Pass counters as tensors, since a changing Python int forces recompiles.

Time 1000 steps with torch.cuda.Event after a warmup. Do it eager first. The compiled step should beat eager tenfold. A raw million env steps per second proves little, since eager likely clears it.

#### Expert

Write forward kinematics for a single arm. vmap(jacrev(fk)) gives Jacobians for all envs at once. A damped least squares solve with torch.linalg.solve maps end effector velocity to joint velocity. Damping of 0.05 keeps it stable near full extension. End effector velocity comes from a proportional controller capped at 0.3 m/s.

The expert flips a coin each episode to pick a side. The route then runs through 2 waypoints on that side. They sit 0.2 m off the axis, at x = 0.3 m then x = 0.5 m. That keeps the puck about 3 cm clear of the pillar. The goal comes last. Advance once the puck's x passes the current waypoint's x.

The parking spot sits 0.08 m behind the puck, opposite the current waypoint. The end effector reaches it by traveling at z = 0.08 m, clear of the puck's top face. It then drops to table height. Pushing adds 0.15 m/s toward the waypoint on top of tracking the spot. If it drifts 3 cm from the spot, it lifts and parks again. Tune these thresholds until success clears 95 percent. Record nothing before that.

#### Data

The observation never sees the waypoints, since they encode the coin. A leak lets the MSE policy learn the side, which erases the result. Audit the obs tensor before training.

```
obs     (N, 7)   joint angles 3, puck xy 2, puck velocity 2
action  (N, 3)   joint velocities
```

Record demos into one preallocated GPU buffer shaped (T, N, ...) with T = 400. That is about 260 MB in float32. Drop failed episodes with a boolean mask over N. Sample 16 step action chunks with tensor indexing. Draw start indices so every chunk fits inside its episode. Skip DataLoader.

Standardize every obs dimension using stats from the buffer. Do the same for actions. The stats live as registered buffers so they ride along in state_dict.

#### Policy one: MSE

An MLP maps obs to a whole chunk, 16 steps of 3 joint velocities flattened to 48. Train it with F.mse_loss under bf16 autocast. The loss itself runs in float32 there anyway. Use AdamW. Keep EMA weights with torch._foreach_lerp_ at decay 0.999.

At rollout, execute 8 actions of each chunk before replanning. Roll out with the EMA weights.

```python
@dataclass
class TrainConfig:
    hidden: int = 512
    layers: int = 3
    batch: int = 1024
    steps: int = 20_000
    lr: float = 3e-4
    ema: float = 0.999
```

#### Policy two: flow

It reuses the MLP trunk. Inputs grow by a noisy chunk with its time t in [0, 1]. Feed t through 16 sinusoidal features.

The noisy chunk interpolates linearly from Gaussian noise at t = 0 to the data chunk at t = 1. Sample t uniformly. The loss is still F.mse_loss. Its target is the chunk minus the noise, the velocity along that path.

Sampling starts from fresh noise. It runs 10 Euler steps of size 0.1 inside the batched rollout. Eager is fast enough here.

#### Evaluation

Run each policy on all 16384 envs for 400 steps. Log success rate per policy. Log the fraction of episodes where the puck touches the pillar. That number carries the MSE result.

Track side switches for the flow policy. Count episodes where the puck gets 5 cm off the axis, then crosses it before x = 0.5 m. Replanning every 8 steps can flip the sampled side.

Overlay 64 trajectories from each policy in Rerun. The loss curves go there too.

### Extensions

All 5 are optional. None changes the minimum.

**Gaussian mixture head.** Swap the last layer for 2 components over the flattened 48 dim chunk. Build it from MixtureSameFamily with Independent(Normal, 1) as the component. Train on negative log likelihood. Floor the std at 0.01 so the likelihood cannot run away. At rollout, execute the mean of the component with the larger logit. The mixture's .mean averages the components, which drives into the pillar like MSE. Done right, it passes the pillar too.

**Compiled sampling loop.** Compile the 10 step Euler loop together with step(). Rollout speed is already fine, so this is CUDA graph practice. Watch for the overwritten output error here.

**DDPM.** Swap flow matching for DDPMScheduler from diffusers. Chi et al.'s Diffusion Policy code uses that scheduler. Their repo will then read like yours.

**Pixels.** Render a 64×64 overhead image in torch from distance fields. Encode it with a ResNet18 from torchvision. Most visuomotor repos, LeRobot included, take this shape.

**DataLoader.** Wrap the demo buffer in a torch.utils.data.Dataset. Every repo you will read loads data that way. Time it against tensor indexing.

## Part 2: transformer trunk

### Minimum

The minimum is done when a transformer flow policy matches your MLP flow policy on the pillar task across 16384 envs, and a transformer MSE policy still stalls against the pillar. Same sim, same expert, same demos, same eval. Only the trunk changes.

#### Setup

Stay in the robopuck uv project. Add one new file. No new dependencies. Write attention yourself with matmuls. nn.Transformer and nn.MultiheadAttention stay out of the minimum.

#### Tokens

Every input becomes a vector of width d = 128. One token per thing.

```
obs tokens     3   joint angles (3), puck xy (2), puck velocity (2), each its own Linear to d
time token     1   t through 16 sinusoidal features, then a 2 layer MLP to d
action tokens  16  one per chunk step, each 3 joint velocities through a Linear to d
```

The sequence is 20 tokens long. Action tokens get a learned positional embedding of shape (16, d). Obs tokens get a learned type embedding, one per kind, so the model can tell joint angles from puck velocity.

#### Attention

One Linear maps d to 3d for queries, keys and values. Split into 4 heads of 32. Scores are q @ k.mT divided by sqrt(32). Softmax over the last axis. Multiply by v, merge the heads, apply an output Linear.

```
x        (B, 20, 128)
qkv      (B, 20, 3, 4, 32)  then permute to (3, B, 4, 20, 32)
scores   (B, 4, 20, 20)
out      (B, 20, 128)
```

Check it against F.scaled_dot_product_attention on random input. They should agree to about 1e-5 in float32. Only then build on it.

#### Block

Pre-LayerNorm. Attention with a residual, then an MLP of width 4d with GELU and a residual. Stack 4 blocks. A final LayerNorm.

#### Policy one: flow

Feed the 20 tokens through the stack. Keep only the 16 action token outputs. A Linear maps each to 3. That is the velocity prediction, shape (B, 16, 3). Loss, path, t sampling and the 10 step Euler sampler are exactly robopuck's.

#### Policy two: MSE

Same stack. No time token. The 16 action tokens are learned query embeddings instead of a noisy chunk. Their outputs are the chunk. This is how ACT and DETR read out a fixed set of outputs.

#### Training

AdamW, lr 1e-4, weight decay 0.01. Warm up linearly over 1000 steps, then cosine down. Clip gradient norm at 1.0. bf16 autocast and EMA 0.999 as before. Print the parameter count. Size d and depth so it lands near your MLP's count, or the comparison means nothing.

#### Checks

Each one is a fast pass or fail before the next piece.

- Attention matches scaled_dot_product_attention.
- Remove the positional embedding. Permute the action tokens. The outputs permute the same way. This is why positions exist.
- Overfit one batch of 64 to near zero loss.
- Batch independence. Change one row, the others must not move.

#### Evaluation

Same as robopuck. Success rate, pillar touch fraction, side switches. Overlay 64 trajectories per policy in Rerun.

Log attention for env 0 as an image, one per head, at the first and last Euler step. Do the action tokens attend to the puck tokens? Does that change as t goes from 0 to 1?

### Extensions

All optional. None changes the minimum.

**Block mask and prefix cache.** Obs tokens attend only to obs tokens. The time and action tokens attend to everything. Now the obs keys and values never depend on the noise, so compute them once per rollout step and reuse them across all 10 Euler steps. This is pi0's blockwise mask. Time it.

**Cross attention.** Split into an encoder over obs tokens and a decoder over action tokens that cross attends to it. Compare to the single stack at matched parameters.

**adaLN-Zero.** Drop the time token. Condition every block on t by regressing a scale, shift and gate from it, gates initialised to zero. This is how DiT conditions.

**Observation history.** Add the previous 2 steps as extra obs tokens. Does the side switch rate drop?

**Compile.** torch.compile the stack. Watch for graph breaks from your reshapes.
