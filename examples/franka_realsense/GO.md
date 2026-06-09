采数据
uv run python examples/franka_realsense/record_teleop_lerobot.py \
      --repo-id mani1/franka_realsense_droid_video \
      --dataset-root ~/franka_realsense_lerobot/mani1/AAA_franka_data \
      --robot-ip 172.16.0.8 \
      --external-camera-serial 215322076954 \
      --wrist-camera-serial 233622071841 \
      --task "pick up all the object and put into the green" \
      --append \
      --velocity-duration-ms 70 \
      --use-videos


数据保存位置：
/home/nvidia/lixu_thor/franka_realsense_droid_video


启动训练(服务器可以使用，推理侧电脑无训练环境)
     PYTHONPATH=$PWD/src:$PWD:$PYTHONPATH \
  /home/nvidia/lixu_thor/openpi_bak/.venv/bin/python \
    scripts/train_pytorch.py \
    pi05_franka_realsense_droid_action_full_align_full_finetune \
    --exp_name AAA_franka_droid_action_new_data \
    --num_train_steps 1600 \
    --save_interval 1600 \
    --overwrite
推理代码：

  PYTHONPATH=$PWD/src:$PWD:$PYTHONPATH \
  uv run python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
    --policy.config=pi05_franka_realsense_droid_action_full_align_full_finetune \
    --policy.dir=/home/mani1/openpi/1600

 

  PYTHONPATH=$PWD/src:$PWD:$PYTHONPATH \
  uv run python examples/franka_realsense/infer_franky_realsense.py \
    --policy-host 127.0.0.1 \
    --policy-port 8000 \
    --robot-ip 172.16.0.8 \
    --external-camera-serial 215322076954 \
    --wrist-camera-serial 233622071841 \
    --prompt "pick up all the object and put into the green"





