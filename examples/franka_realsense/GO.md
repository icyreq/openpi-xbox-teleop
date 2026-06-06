采数据
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


数据保存位置：
/home/nvidia/lixu_thor/franka_realsense_droid_video


启动训练
  PYTHONPATH=$PWD/src:$PWD:$PYTHONPATH \
  /home/nvidia/lixu_thor/openpi_bak/.venv/bin/python \
    scripts/train_pytorch.py \
    pi05_franka_realsense_video_joint_full_finetune \
    --exp_name franka_realsense_joint_action_from_pi05_base_v1 \
    --num_train_steps 3003 \
    --save_interval 1000 \
    --overwrite

