"""Run a remote pi05 Franka RealSense policy on a Franka robot through franky.

The policy was trained with DROID-style actions:
    action[:7] = (q[t + 1] - q[t]) / 0.2
    action[7] = absolute gripper position, where 0 is open and 1 is closed

At inference time we therefore convert each predicted arm action back to the
same one-step joint target used by the DROID control stack:
    target_q = current_q + action[:7] * 0.2
"""

from __future__ import annotations

import argparse
import time

import cv2
from franky import Gripper
from franky import JointMotion
from franky import Robot
import numpy as np
from openpi_client import websocket_client_policy
from PIL import Image
import pyrealsense2 as rs


DROID_CONTROL_HZ = 15.0
DROID_MAX_JOINT_DELTA = 0.2
DROID_IMAGE_WIDTH = 320
DROID_IMAGE_HEIGHT = 180


class RealSenseCamera:
    """Small RGB-only RealSense wrapper matching the collection-time camera path."""

    def __init__(self, *, serial: str, width: int, height: int, fps: int, warmup_frames: int) -> None:
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self._pipeline.start(cfg)

        for _ in range(warmup_frames):
            self.read_rgb()

    def read_rgb(self) -> np.ndarray:
        frames = self._pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("RealSense color frame is empty.")
        return np.ascontiguousarray(np.asarray(color.get_data())[..., :3])

    def close(self) -> None:
        self._pipeline.stop()


def resize_for_droid(image: np.ndarray) -> np.ndarray:
    """Match dataset collection: direct resize to 320x180 before OpenPI pads to 224x224."""
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    return np.asarray(Image.fromarray(image).resize((DROID_IMAGE_WIDTH, DROID_IMAGE_HEIGHT), resampling), dtype=np.uint8)


def show_camera_preview(*, window_name: str, external_image: np.ndarray, wrist_image: np.ndarray) -> bool:
    """Display the current RGB camera pair. Returns True when the user presses q."""
    if external_image.shape != wrist_image.shape:
        raise ValueError(f"Camera image shapes differ: external={external_image.shape}, wrist={wrist_image.shape}.")
    preview_rgb = np.concatenate([external_image, wrist_image], axis=1)
    preview_bgr = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, preview_bgr)
    return (cv2.waitKey(1) & 0xFF) == ord("q")


def gripper_width_to_droid_position(width: float, *, open_width: float, closed_width: float) -> np.ndarray:
    """Convert franky gripper width to the DROID convention used in training."""
    span = open_width - closed_width
    closed_amount = (open_width - width) / span
    return np.asarray([np.clip(closed_amount, 0.0, 1.0)], dtype=np.float32)


def make_observation(
    *,
    robot: Robot,
    gripper: Gripper | None,
    external_image: np.ndarray,
    wrist_image: np.ndarray,
    prompt: str,
    gripper_open_width: float,
    gripper_closed_width: float,
) -> dict:
    """Build the exact DROID policy observation keys consumed by DroidInputs."""
    joint_position = np.asarray(robot.current_joint_state.position, dtype=np.float32)
    if gripper is None:
        gripper_position = np.asarray([0.0], dtype=np.float32)
    else:
        gripper_position = gripper_width_to_droid_position(
            float(gripper.width),
            open_width=gripper_open_width,
            closed_width=gripper_closed_width,
        )

    return {
        "observation/exterior_image_1_left": resize_for_droid(external_image),
        "observation/wrist_image_left": resize_for_droid(wrist_image),
        "observation/joint_position": joint_position,
        "observation/gripper_position": gripper_position,
        "prompt": prompt,
    }


def execute_action(
    *,
    robot: Robot,
    gripper: Gripper | None,
    action: np.ndarray,
    gripper_open_width: float,
    gripper_closed_width: float,
    gripper_speed: float,
    last_gripper_target: float | None,
) -> float | None:
    """Decode one DROID-style action and send it through franky."""
    action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

    current_q = np.asarray(robot.current_joint_state.position, dtype=np.float64)
    target_q = current_q + action[:7] * DROID_MAX_JOINT_DELTA
    robot.move(JointMotion(target_q.tolist()), asynchronous=True)

    if gripper is None:
        return None

    gripper_target = 1.0 if float(action[7]) > 0.5 else 0.0
    if gripper_target != last_gripper_target:
        width = gripper_closed_width if gripper_target > 0.5 else gripper_open_width
        gripper.move_async(width, gripper_speed)
    return gripper_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8000)

    parser.add_argument("--robot-ip", default="172.16.0.8")
    parser.add_argument("--external-camera-serial", required=True)
    parser.add_argument("--wrist-camera-serial", required=True)

    parser.add_argument("--prompt", required=True)

    parser.add_argument("--max-timesteps", type=int, default=999999999)
    parser.add_argument("--open-loop-horizon", type=int, default=8)
    parser.add_argument("--control-hz", type=float, default=DROID_CONTROL_HZ)

    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-warmup-frames", type=int, default=90)
    parser.add_argument("--disable-preview", action="store_true")
    parser.add_argument("--preview-window-name", default="Franka RealSense inference")

    parser.add_argument("--relative-dynamics-factor", type=float, default=0.1)
    parser.add_argument("--disable-gripper", action="store_true")
    parser.add_argument("--gripper-open-width", type=float, default=0.08)
    parser.add_argument("--gripper-closed-width", type=float, default=0.0)
    parser.add_argument("--gripper-speed", type=float, default=0.08)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    policy = websocket_client_policy.WebsocketClientPolicy(
        host=args.policy_host,
        port=args.policy_port,
    )
    print(f"Connected to policy server: {policy.get_server_metadata()}")

    # Create hardware handles only after the remote policy server is reachable.
    robot = Robot(args.robot_ip)
    robot.relative_dynamics_factor = args.relative_dynamics_factor
    gripper = None if args.disable_gripper else Gripper(args.robot_ip)
    external_camera = RealSenseCamera(
        serial=args.external_camera_serial,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        warmup_frames=args.camera_warmup_frames,
    )
    wrist_camera = RealSenseCamera(
        serial=args.wrist_camera_serial,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        warmup_frames=args.camera_warmup_frames,
    )

    action_chunk = None
    chunk_index = 0
    last_gripper_target = None
    control_period = 1.0 / args.control_hz

    if not args.disable_preview:
        cv2.namedWindow(args.preview_window_name, cv2.WINDOW_NORMAL)

    try:
        for _ in range(args.max_timesteps):
            step_start = time.monotonic()
            external_image = external_camera.read_rgb()
            wrist_image = wrist_camera.read_rgb()

            if not args.disable_preview and show_camera_preview(
                window_name=args.preview_window_name,
                external_image=external_image,
                wrist_image=wrist_image,
            ):
                break

            # Query a new action chunk, then execute a short open-loop prefix at 15 Hz.
            if action_chunk is None or chunk_index >= min(args.open_loop_horizon, len(action_chunk)):
                obs = make_observation(
                    robot=robot,
                    gripper=gripper,
                    external_image=external_image,
                    wrist_image=wrist_image,
                    prompt=args.prompt,
                    gripper_open_width=args.gripper_open_width,
                    gripper_closed_width=args.gripper_closed_width,
                )
                action_chunk = policy.infer(obs)["actions"]
                chunk_index = 0

            last_gripper_target = execute_action(
                robot=robot,
                gripper=gripper,
                action=action_chunk[chunk_index],
                gripper_open_width=args.gripper_open_width,
                gripper_closed_width=args.gripper_closed_width,
                gripper_speed=args.gripper_speed,
                last_gripper_target=last_gripper_target,
            )
            chunk_index += 1

            elapsed = time.monotonic() - step_start
            if elapsed < control_period:
                time.sleep(control_period - elapsed)
    finally:
        if not args.disable_preview:
            cv2.destroyWindow(args.preview_window_name)
        external_camera.close()
        wrist_camera.close()


if __name__ == "__main__":
    main()
