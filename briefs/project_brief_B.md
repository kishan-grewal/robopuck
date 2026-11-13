# Project brief B: pixels, words, MuJoCo

"Brief 1" below means brief A, the robopuck project, both parts. Its block mask is part 2's block mask extension. It is planned for a new repo, not robopuck.

## Minimum

The minimum is done when a policy that sees only pixels, joint angles and one word clears the pillar task in MuJoCo. Told "left" or "right", it goes that side almost every time. Told "any", it splits close to half and half and rarely touches the pillar. It is your brief 1 transformer with images and a word in the prefix.

### Setup

```bash
uv add mujoco mujoco-warp warp-lang
```

MuJoCo Warp has been officially released since MuJoCo 3.5. It runs many worlds on the GPU in one call. Warp arrays become torch tensors with wp.to_torch, without a copy. It is not differentiable, which you don't need. Read the MJWarp docs page for the current step, data and render calls before writing any. Don't trust remembered names.

### Scene

Rebuild the robopuck scene in MJCF. Same numbers. Arm base at the origin, 3 links of 0.4 m, yaw then two pitch joints. Puck radius 0.05 m, height 0.03 m, mass 0.1 kg. Pillar radius 0.1 m at x = 0.4 m. Goal at x = 0.6 m.

Give the puck two slide joints, x and y, and nothing else. It still never rotates. Drive the arm with velocity actuators on the 3 joints. The arm is now dynamic, not kinematic, so commanded and actual joint velocity differ. Contact and friction are MuJoCo's now, not your spring.

Put the camera overhead, looking straight down at the table. 64x64 RGB.

### Port check

Run your robopuck expert in MuJoCo. It reads puck xy and joint angles from MuJoCo's state instead of your tensors. Your fk, jac and dls carry over unchanged. Tune until success clears 95 percent again. Log the scene to Rerun the same way.

Time 1000 steps of N worlds against your own torch sim. Report the number honestly. Yours may well win on this scene.

### Data

Render only while recording demos. 512 successful episodes of 400 steps at 64x64x3 uint8 is about 2.5 GB. Keep frames as uint8 and convert to float per batch. A 16 GB card ran out of memory near 100 worlds with two 128x128 cameras each, so render in chunks of worlds if you need to.

The observation is the current frame and the previous one, stacked to 6 channels, plus 3 joint angles. One frame can't show the puck's velocity. Two can.

The word is "left", "right" or "any". Label each demo with its side. Swap a random 30 percent of labels to "any" at load time, so the same demo teaches both.

### Encoder

Write a small CNN by hand. 4 conv layers, stride 2, GroupNorm, GELU. 64x64 in, a 4x4 grid of d = 128 out. Flatten the grid into 16 image tokens with a learned positional embedding.

Augment with random shift. Pad 4 pixels with edge values, then crop back to 64x64 at a random offset per sample.

### Tokens

```
prefix   16 image tokens, 1 joint angle token, 1 word token (nn.Embedding over 3 words)
suffix   1 time token, 16 action tokens
```

Use brief 1's block mask. The prefix attends only to itself. The suffix attends to everything. Cache the prefix keys and values once per control step and reuse them across the 10 Euler steps. That is pi0's structure with no pretrained VLM: the prefix is what the VLM would have been, the suffix is the action expert.

### Training

Brief 1's flow loss, optimiser and schedule. The encoder trains end to end with the transformer.

### Checks

- The image tokens are not ignored. Zero the frames at eval and success should collapse.
- The word is not ignored. Shuffle words across a batch and the side should stop following them.
- The state policy from brief 1 in MuJoCo is your ceiling. The pixel policy is measured against it.

### Evaluation

Run all worlds for 400 steps per word. Log success, pillar touch fraction, and which side each episode took. "left" and "right" should follow the word. "any" should split.

Overlay 64 trajectories per word in Rerun. Log the attention from action tokens to image tokens as a 4x4 map over the frame. Does it sit on the puck?

## Extensions

All optional. None changes the minimum.

**Push-T.** The Diffusion Policy benchmark. A T block pushed to a target pose by the end effector. Multimodal, contact rich, rotation matters. Write the scripted expert yourself.

**Pretrained encoder.** Swap in an ImageNet ResNet18 with spatial softmax. Compare data efficiency at 64, 128 and 512 demos.

**Domain randomisation.** Randomise table colour, light direction and camera pose by a few degrees per world. Measure success on held out settings with and without it.

**Wrist camera.** A second 64x64 view on the end effector. 16 more image tokens.

**Real-time chunking.** Start sampling the next chunk while the current one executes, as at Extend. Measure the jerk at chunk boundaries before and after.

**DAgger.** Your scripted expert can label any state the policy reaches. Compare 3 rounds of on-policy relabelling against the same number of extra offline demos.
