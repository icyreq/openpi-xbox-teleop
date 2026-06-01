# Franka + RealSense + Xbox 采集与 pi05-DROID 微调

这个目录用于在 Franka 机械臂上用两路 Intel RealSense RGB 相机替代原始 DROID 的相机栈，并复用 OpenPI 的 `pi05_droid` 策略。

当前代码支持两件事：

- 用 Xbox 手柄遥操作 Franka，并把数据直接采集成 LeRobot/DROID schema。
- 在服务器上从 `pi05_droid` checkpoint 做全参数微调，然后把训练好的 checkpoint 拷回机器人侧推理。

数据和策略输入的映射如下：

- 外部 RealSense 相机 -> `observation/exterior_image_1_left`
- 腕部 RealSense 相机 -> `observation/wrist_image_left`
- Franka 7 维关节角 -> `observation/joint_position`
- Franka Hand 宽度归一化 -> `observation/gripper_position`，其中 `0=open`，`1=closed`
- 策略输出 -> 7 维关节速度 + 1 维夹爪命令

## 当前这版数据的图像路径

这次已经采好的数据使用的是当前 `record_teleop_lerobot.py` 默认路径：

```text
RealSense 640x480
  -> 存储前 resize 到 320x180
  -> 训练时 resize_with_pad 到 224x224
```

为了让训练和推理看到一致的图像分布，`examples/franka_realsense/main.py` 默认也会走同样的推理预处理：

```text
RealSense 640x480
  -> resize 到 320x180
  -> resize_with_pad 到 224x224
  -> 送入 policy
```

这和官方 DROID client 的直接 `resize_with_pad(224, 224)` 有区别，但只影响本目录的 Franka + RealSense client，不影响 `examples/droid/main.py`。

如果要测试官方原版 `pi05_droid` checkpoint，而不是我们这次 fine-tune 出来的模型，可以加：

```bash
--policy-image-preprocess direct
```

这样推理路径会恢复为：

```text
RealSense 640x480
  -> resize_with_pad 到 224x224
```

下次重新采集数据时，建议把 RealSense 原始流改成 16:9，例如 `640x360`，再存成 `320x180`，这样可以避免 `640x480 -> 320x180` 带来的宽高比拉伸。

## 采集数据

采集脚本：

```text
examples/franka_realsense/record_teleop_lerobot.py
```

它会直接写出 LeRobot 数据集，字段和 OpenPI 的 DROID fine-tune 配置匹配：

- `exterior_image_1_left`：外部 RealSense RGB，存成 `320x180`
- `exterior_image_2_left`：外部图像的拷贝，仅用于满足 DROID LeRobot schema
- `wrist_image_left`：腕部 RealSense RGB，存成 `320x180`
- `joint_position`：Franka 7 维关节角
- `gripper_position`：Franka Hand 宽度归一化，`0=open`，`1=closed`
- `actions`：Franka 7 维关节速度 `dq` + 1 维夹爪命令，15 Hz

当前这批 video 数据的采集命令示例：

```bash
uv run python examples/franka_realsense/record_teleop_lerobot.py \
  --repo-id mani1/franka_realsense_droid_video \
  --dataset-root ~/franka_realsense_lerobot/mani1/franka_realsense_droid_video \
  --robot-ip 172.16.0.8 \
  --external-camera-serial 215322076954 \
  --wrist-camera-serial 233622071841 \
  --task "pick up all the object and put into the green" \
  --append \
  --velocity-duration-ms 70 \
  --use-videos
```

采集完成后的数据目录是：

```text
~/franka_realsense_lerobot/mani1/franka_realsense_droid_video
```

已确认当前视频实际分辨率是：

```text
320x180 @ 15 fps
```

Xbox 手柄映射：

- 左摇杆：XY 平移
- 右摇杆上下：Z 平移
- 右摇杆左右：yaw 旋转
- LT / RT：pitch 旋转
- D-pad 左右：roll 旋转
- LB / RB：关闭 / 打开夹爪
- Y：开始录制，或停止并保存当前 episode
- X：录制中丢弃当前 episode；未录制时打印关节状态
- B：速度停止开关
- A：回 home
- START：回 home 并打开夹爪
- BACK：recover

如果手柄没有响应，先跑 debug：

```bash
uv run python examples/franka_realsense/record_teleop_lerobot.py \
  --repo-id mani1/franka_realsense_droid_video \
  --dataset-root ~/franka_realsense_lerobot/mani1/franka_realsense_droid_video \
  --robot-ip 172.16.0.8 \
  --external-camera-serial 215322076954 \
  --wrist-camera-serial 233622071841 \
  --task "pick up all the object and put into the green" \
  --append \
  --use-videos \
  --debug-joystick
```

如果终端里 `events=0`，说明选错了 joystick 设备，需要用 `--joy-device` 指定实际设备。如果左右摇杆正常但 trigger 不对，可以尝试：

```bash
--joystick-layout xbox_bluetooth
```

## 服务器上全参数微调

当前配置名：

```text
pi05_franka_realsense_video_full_finetune
```

配置位置：

```text
src/openpi/training/config.py
```

这个配置会：

- 从 `gs://openpi-assets/checkpoints/pi05_droid/params` 加载 `pi05_droid`
- 使用数据集 `mani1/franka_realsense_droid_video`
- 复用 DROID normalization stats
- 使用 `action_horizon=15`
- 不设置 `freeze_filter`，所以是全参数微调，不是 LoRA

### 1. 上传代码到服务器

在本地把代码推到 GitHub 后，服务器上 clone：

```bash
git clone git@github.com:icyreq/openpi-xbox-teleop.git
cd openpi-xbox-teleop
```

然后按 OpenPI 原项目方式安装依赖。通常是：

```bash
uv sync
```

如果服务器训练时要下载 `gs://openpi-assets/...`，需要保证服务器能访问 Google Cloud Storage。

### 2. 上传数据到服务器

把本地数据目录：

```text
~/franka_realsense_lerobot/mani1/franka_realsense_droid_video
```

拷到服务器的 LeRobot cache 路径：

```text
~/.cache/huggingface/lerobot/mani1/franka_realsense_droid_video
```

示例，用 `rsync`：

```bash
rsync -avh --progress \
  ~/franka_realsense_lerobot/mani1/franka_realsense_droid_video/ \
  <server_user>@<server_host>:~/.cache/huggingface/lerobot/mani1/franka_realsense_droid_video/
```

注意最后的 `/`：它表示把目录内容同步到目标目录。

服务器上检查数据是否到位：

```bash
ls ~/.cache/huggingface/lerobot/mani1/franka_realsense_droid_video/meta
find ~/.cache/huggingface/lerobot/mani1/franka_realsense_droid_video/videos -name '*.mp4' | head
```

应该能看到：

```text
meta/info.json
meta/episodes.jsonl
meta/tasks.jsonl
videos/chunk-000/...
```

### 3. 启动训练

在服务器仓库根目录运行：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_franka_realsense_video_full_finetune \
  --exp-name franka_realsense_droid_video_v1 \
  --overwrite
```

checkpoint 会保存在：

```text
checkpoints/pi05_franka_realsense_video_full_finetune/franka_realsense_droid_video_v1/
```

当前配置：

- `num_train_steps=20_000`
- `batch_size=32`
- `save_interval=1000`
- `keep_period=5000`

如果显存不够，优先在 `src/openpi/training/config.py` 里把这个 config 的 `batch_size` 调小，例如从 `32` 改成 `16` 或 `8`。

## 训练完后回到机器人侧推理

训练完成后，选择一个 checkpoint step，例如：

```text
checkpoints/pi05_franka_realsense_video_full_finetune/franka_realsense_droid_video_v1/20000
```

把它从服务器拷回机器人侧同样的相对路径，例如：

```bash
rsync -avh --progress \
  <server_user>@<server_host>:~/openpi-xbox-teleop/checkpoints/pi05_franka_realsense_video_full_finetune/franka_realsense_droid_video_v1/20000/ \
  checkpoints/pi05_franka_realsense_video_full_finetune/franka_realsense_droid_video_v1/20000/
```

同时确保机器人侧代码包含这些改动：

```text
src/openpi/training/config.py
examples/franka_realsense/
```

### 1. 启动 policy server

在有 GPU 的机器上启动 fine-tuned policy：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_franka_realsense_video_full_finetune \
  --policy.dir=checkpoints/pi05_franka_realsense_video_full_finetune/franka_realsense_droid_video_v1/20000 \
  --port 8000
```

如果 policy server 不在机器人控制电脑上，记下服务器 IP，下面用 `--policy-host` 指向它。

### 2. 先 dry run

机器人侧先不执行动作，只连接相机、机器人和 policy server，并保存预览图：

```bash
python examples/franka_realsense/main.py \
  --policy-host <policy_server_ip> \
  --policy-port 8000 \
  --robot-ip 172.16.0.8 \
  --external-camera-serial 215322076954 \
  --wrist-camera-serial 233622071841 \
  --prompt "pick up all the object and put into the green"
```

检查：

```text
franka_realsense_views.jpg
franka_realsense_policy_inputs.jpg
```

其中 `franka_realsense_policy_inputs.jpg` 是实际送给 policy 的两路 `224x224` 输入预览。

### 3. 小速度执行

确认预览图和动作输出合理后，再执行：

```bash
python examples/franka_realsense/main.py \
  --policy-host <policy_server_ip> \
  --policy-port 8000 \
  --robot-ip 172.16.0.8 \
  --external-camera-serial 215322076954 \
  --wrist-camera-serial 233622071841 \
  --prompt "pick up all the object and put into the green" \
  --execute \
  --max-joint-velocity 0.15 \
  --franky-dynamics-factor 0.03
```

脚本会要求输入：

```text
EXECUTE
```

才会真正给机器人发动作。第一次测试时建议保持急停可触达。

如果机械臂几乎不动，逐步增大：

```bash
--franky-dynamics-factor
```

如果动作太猛或出现 discontinuity error，减小：

```bash
--max-joint-velocity
--policy-velocity-scale
--franky-dynamics-factor
```

逐步单步执行可以加：

```bash
--step-on-space
```

示例：

```bash
python examples/franka_realsense/main.py \
  --policy-host <policy_server_ip> \
  --policy-port 8000 \
  --robot-ip 172.16.0.8 \
  --external-camera-serial 215322076954 \
  --wrist-camera-serial 233622071841 \
  --prompt "pick up all the object and put into the green" \
  --execute \
  --yes \
  --step-on-space
```

每按一次空格执行一个 action，按 `q` 退出。

## 常用排查

列出 RealSense 相机：

```bash
python examples/franka_realsense/main.py --list-cameras
```

如果预览图发绿/发蓝，增加 RealSense warmup：

```bash
--camera-warmup-frames 90
```

如果白平衡仍然不对，可以固定白平衡：

```bash
--white-balance 4500
```

如果看起来是红蓝通道反了，加：

```bash
--assume-realsense-rgb
```

如果左右相机反了，加：

```bash
--swap-cameras
```

如果要测试官方原版 `pi05_droid` 而不是我们的 fine-tuned 模型，加：

```bash
--policy-image-preprocess direct
```
