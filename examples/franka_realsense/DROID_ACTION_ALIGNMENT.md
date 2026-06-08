# Franka RealSense DROID Action Alignment

This note records the action-space decision for the Franka + RealSense dataset and
how it should be used for training and inference with DROID checkpoints.

## Why This Exists

The original Franka RealSense collection script stored:

```python
actions[:7] = robot_state.dq
actions[7] = gripper_action
```

That is a measured joint velocity from the Franka runtime. It matches the name
`joint_velocity`, but it is not the same action semantics used by the official
DROID policy stack.

The official DROID code uses `joint_velocity` as an action-space name. In
execution, it is better understood as a normalized one-step joint delta:

```python
joint_delta = joint_velocity_action * 0.2
target_q = current_q + joint_delta
```

The target sent to the low-level robot controller is therefore a desired joint
position, not a continuously held physical velocity command in rad/s.

## Official DROID Chain

During DROID data collection, the VR controller produces a Cartesian velocity
action. `RobotEnv.step(action)` converts that command into an `action_info`
dictionary that includes several equivalent representations:

```text
cartesian_velocity
cartesian_position
joint_velocity
joint_position
gripper_position
```

For DROID training in OpenPI, the loader uses:

```python
actions = [
    action_dict["joint_velocity"] or action_dict["joint_position"],
    action_dict["gripper_position"],
]
```

For the pi05-DROID velocity-action checkpoint, the first seven action dimensions
follow the DROID `joint_velocity` action convention, and the eighth dimension is
absolute gripper position.

At inference, the OpenPI DROID example creates:

```python
RobotEnv(action_space="joint_velocity", gripper_action_space="position")
```

and executes one predicted action per 15 Hz tick. Inside DROID, the first seven
dimensions are converted to a joint-position target through:

```python
target_q = current_q + action[:7] * 0.2
```

The gripper dimension is treated as absolute position:

```text
0.0 = fully open
1.0 = fully closed
```

The official inference example thresholds the gripper output at 0.5 before
sending it.

## Our Aligned Dataset

To align the Franka RealSense data with the official DROID checkpoint, we create
a separate dataset:

```text
/home/nvidia/lixu_thor/franka_realsense_droid_video_droid_action
```

The conversion is:

```python
actions[:7] = (joint_position[t + 1] - joint_position[t]) / 0.2
actions[7] = gripper_position[t + 1]
```

The final frame repeats the final state, so its joint action is zero.

The source dataset was checked before conversion:

```text
frames: 6556
max(abs((q[t+1] - q[t]) / 0.2)): 0.2844
rows outside [-1, 1]: 0
```

So the aligned joint actions already fit the official DROID command envelope.
No clipping or vector scaling is needed for the current data.

## Training Config

The aligned training config is:

```text
pi05_franka_realsense_droid_action_full_align_full_finetune
```

Important properties:

```text
repo_root: /home/nvidia/lixu_thor/franka_realsense_droid_video_droid_action
actions[:7]: already normalized DROID-style joint deltas
actions[7]: absolute gripper position
DeltaActions: not used
normalization stats: computed from this aligned dataset
initialization: pi05_droid PyTorch checkpoint
```

Because the actions are already in the DROID-style normalized delta convention,
do not apply `DeltaActions` again in this config.

## Inference Contract

An inference script for this aligned policy should:

1. Run at 15 Hz for action execution.
2. Pass the current observation with DROID keys:
   ```text
   observation/exterior_image_1_left
   observation/wrist_image_left
   observation/joint_position
   observation/gripper_position
   prompt
   ```
3. Interpret policy output as:
   ```python
   normalized_delta = action[:7]
   gripper_position = action[7]
   ```
4. Convert arm action to a joint-position target:
   ```python
   target_q = current_q + normalized_delta * 0.2
   ```
5. Treat the gripper as absolute DROID position. If following the official
   DROID inference behavior, threshold it:
   ```python
   gripper_position = 1.0 if gripper_position > 0.5 else 0.0
   ```

If policy inference blocks, the official OpenPI DROID example does not continue
issuing old actions while waiting for a new chunk. For our Franka deployment,
the control loop should define an explicit safe behavior for stale or missing
actions instead of relying on controller hold behavior.

## Commands

Convert the dataset:

```bash
PYTHONPATH=$PWD/src:$PWD:$PYTHONPATH \
/home/nvidia/lixu_thor/openpi_bak/.venv/bin/python \
  examples/franka_realsense/convert_to_droid_action_targets.py \
  --overwrite
```

Compute normalization stats after conversion:

```bash
PYTHONPATH=$PWD/src:$PWD:$PYTHONPATH \
/home/nvidia/lixu_thor/openpi_bak/.venv/bin/python \
  scripts/compute_norm_stats.py \
  --config-name pi05_franka_realsense_droid_action_full_align_full_finetune
```

Then launch training with the configured pi05-DROID PyTorch checkpoint:

```bash
PYTHONPATH=$PWD/src:$PWD:$PYTHONPATH \
/home/nvidia/lixu_thor/openpi_bak/.venv/bin/python \
  scripts/train_pytorch.py \
  pi05_franka_realsense_droid_action_full_align_full_finetune \
  --exp_name <experiment_name> \
  --num_train_steps <steps> \
  --save_interval 1000 \
  --overwrite
```
