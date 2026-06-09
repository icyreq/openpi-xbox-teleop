# ruff: noqa: T201

import contextlib
import dataclasses
import pathlib
import select
import sys
import termios
import time
import tty
from typing import Any, Literal

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from PIL import Image
import tyro

DROID_CONTROL_HZ = 15.0
DROID_IMAGE_WIDTH = 320
DROID_IMAGE_HEIGHT = 180
POLICY_IMAGE_SIZE = 224


@dataclasses.dataclass
class Args:
    # Policy server. Start it with: uv run scripts/serve_policy.py --env DROID --port 8000
    policy_host: str = "127.0.0.1"
    policy_port: int = 8000

    # Franka FCI IP address.
    robot_ip: str = "10.90.90.1"

    # RealSense serial numbers. Use --list-cameras to print connected RealSense devices.
    external_camera_serial: str | None = None
    wrist_camera_serial: str | None = None
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    camera_warmup_frames: int = 30
    auto_exposure: bool = True
    auto_white_balance: bool = True
    white_balance: float | None = None
    assume_realsense_rgb: bool = False
    swap_cameras: bool = False
    list_cameras: bool = False

    # Rollout.
    prompt: str | None = None
    max_timesteps: int = 600
    open_loop_horizon: int = 8
    control_hz: float = DROID_CONTROL_HZ
    preview_path: pathlib.Path | None = pathlib.Path("franka_realsense_views.jpg")
    policy_preview_path: pathlib.Path | None = pathlib.Path("franka_realsense_policy_inputs.jpg")
    policy_image_preprocess: Literal["match_collected_dataset", "direct"] = "match_collected_dataset"
    step_on_space: bool = False

    # By default this script only runs inference and prints actions.
    execute: bool = False
    franky_dynamics_factor: float = 0.3

    # Franka Hand gripper. OpenPI/DROID convention is 0=open, 1=closed.
    enable_gripper: bool = True
    gripper_open_width: float = 0.08
    gripper_closed_width: float = 0.0
    gripper_speed: float = 0.03
    gripper_command_period: float = 0.75


class RealSenseCamera:
    def __init__(
        self,
        *,
        serial: str,
        width: int,
        height: int,
        fps: int,
        warmup_frames: int,
        auto_exposure: bool,
        auto_white_balance: bool,
        white_balance: float | None,
        assume_rgb: bool,
    ):
        import pyrealsense2 as rs

        self._rs = rs
        self._assume_rgb = assume_rgb
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = self._pipeline.start(config)

        color_sensor = profile.get_device().first_color_sensor()
        self._set_option(color_sensor, rs.option.enable_auto_exposure, 1.0 if auto_exposure else 0.0)
        self._set_option(color_sensor, rs.option.enable_auto_white_balance, 1.0 if auto_white_balance else 0.0)
        if white_balance is not None:
            self._set_option(color_sensor, rs.option.enable_auto_white_balance, 0.0)
            self._set_option(color_sensor, rs.option.white_balance, white_balance)

        for _ in range(warmup_frames):
            self._pipeline.wait_for_frames(timeout_ms=2000)

    def read_rgb(self) -> np.ndarray:
        frames = self._pipeline.wait_for_frames(timeout_ms=2000)
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("RealSense color frame is empty.")
        bgr = np.asanyarray(color_frame.get_data())
        if self._assume_rgb:
            return np.ascontiguousarray(bgr)
        return np.ascontiguousarray(bgr[..., ::-1])

    def close(self) -> None:
        self._pipeline.stop()

    @staticmethod
    def _set_option(sensor, option, value: float) -> None:
        if sensor.supports(option):
            sensor.set_option(option, value)


class FrankaController:
    def __init__(self, args: Args):
        import franky

        self._franky = franky
        self._robot = franky.Robot(args.robot_ip)
        self._robot.relative_dynamics_factor = args.franky_dynamics_factor
        self._gripper = franky.Gripper(args.robot_ip) if args.enable_gripper else None
        self._franky_dynamics_factor = args.franky_dynamics_factor
        self._gripper_open_width = args.gripper_open_width
        self._gripper_closed_width = args.gripper_closed_width
        self._gripper_speed = args.gripper_speed
        self._gripper_command_period = args.gripper_command_period
        self._last_gripper_closed: bool | None = None
        self._last_gripper_command_time = 0.0
        self._gripper_future: Any | None = None

    def joint_position(self) -> np.ndarray:
        return np.asarray(self._robot.current_joint_state.position, dtype=np.float64)

    def gripper_position(self) -> np.ndarray:
        if self._gripper is None:
            return np.zeros(1, dtype=np.float64)

        width = float(self._gripper.width)
        span = max(self._gripper_open_width - self._gripper_closed_width, 1e-6)
        closed_amount = (self._gripper_open_width - width) / span
        return np.asarray([np.clip(closed_amount, 0.0, 1.0)], dtype=np.float64)

    def step(self, action: np.ndarray, *, asynchronous: bool) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        joint_target = self._joint_target(action)
        motion = self._franky.JointMotion(
            joint_target.tolist(),
            relative_dynamics_factor=self._franky_dynamics_factor,
        )
        self._robot.move(motion, asynchronous=asynchronous)
        self._maybe_command_gripper(float(action[7]))
        return joint_target

    def execute_action_sequence(self, actions: np.ndarray, *, period_s: float) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float64)
        joint_targets = np.stack([self._joint_target(action) for action in actions])
        period_ms = max(1, round(period_s * 1000))
        waypoints = [
            self._franky.JointWaypoint(
                target.tolist(),
                hold_target_duration=self._franky.Duration(period_ms),
                relative_dynamics_factor=self._franky_dynamics_factor,
            )
            for target in joint_targets
        ]
        motion = self._franky.JointWaypointMotion(waypoints)
        self._robot.move(motion, asynchronous=True)

        next_tick = time.monotonic()
        for action in actions:
            self._maybe_command_gripper(float(action[7]))
            next_tick += period_s
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
        self._robot.join_motion()
        return joint_targets

    def stop(self) -> None:
        with contextlib.suppress(Exception):
            self._robot.move(self._franky.JointStopMotion())
        if self._gripper is not None:
            with contextlib.suppress(Exception):
                self._gripper.stop()

    def _maybe_command_gripper(self, gripper_action: float) -> None:
        if self._gripper is None:
            return

        closed = gripper_action > 0.5
        now = time.monotonic()
        if closed == self._last_gripper_closed and now - self._last_gripper_command_time < self._gripper_command_period:
            return

        if self._gripper_future is not None and not self._gripper_future.wait(0.0):
            return

        target_width = self._gripper_closed_width if closed else self._gripper_open_width
        self._gripper_future = self._gripper.move_async(target_width, self._gripper_speed)
        self._last_gripper_closed = closed
        self._last_gripper_command_time = now

    def _joint_target(self, action: np.ndarray) -> np.ndarray:
        joint_target = np.asarray(action[:7], dtype=np.float64)
        if joint_target.shape != (7,):
            raise ValueError(f"Expected 7D joint target, got shape {joint_target.shape}.")
        return joint_target


def main(args: Args) -> None:
    if args.list_cameras:
        list_realsense_cameras()
        return

    if not args.external_camera_serial or not args.wrist_camera_serial:
        raise ValueError("Set both --external-camera-serial and --wrist-camera-serial. Use --list-cameras first.")
    if args.external_camera_serial == args.wrist_camera_serial:
        raise ValueError("External and wrist camera serial numbers must be different.")

    instruction = args.prompt or input("Enter instruction: ")

    policy_client = websocket_client_policy.WebsocketClientPolicy(args.policy_host, args.policy_port)
    print(f"Connected to policy server: {policy_client.get_server_metadata()}")
    if args.swap_cameras:
        print("Camera inputs are swapped: wrist serial will be sent as exterior, external serial as wrist.")

    action_chunk: np.ndarray | None = None
    actions_from_chunk_completed = 0
    period_s = 1.0 / args.control_hz
    controller: FrankaController | None = None
    external_camera: RealSenseCamera | None = None
    wrist_camera: RealSenseCamera | None = None

    try:
        controller = FrankaController(args)
        external_camera = RealSenseCamera(
            serial=args.external_camera_serial,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            warmup_frames=args.camera_warmup_frames,
            auto_exposure=args.auto_exposure,
            auto_white_balance=args.auto_white_balance,
            white_balance=args.white_balance,
            assume_rgb=args.assume_realsense_rgb,
        )
        wrist_camera = RealSenseCamera(
            serial=args.wrist_camera_serial,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            warmup_frames=args.camera_warmup_frames,
            auto_exposure=args.auto_exposure,
            auto_white_balance=args.auto_white_balance,
            white_balance=args.white_balance,
            assume_rgb=args.assume_realsense_rgb,
        )

        if args.execute and not args.step_on_space:
            run_continuous_rollout(
                args=args,
                controller=controller,
                external_camera=external_camera,
                wrist_camera=wrist_camera,
                policy_client=policy_client,
                instruction=instruction,
                period_s=period_s,
            )
            return

        with KeyboardStepper(enabled=args.step_on_space) as stepper:
            if args.step_on_space:
                print("Manual stepping enabled: press SPACE to execute one action, q to quit.")

            for t_step in range(args.max_timesteps):
                if stepper.wait_for_step() == "quit":
                    break

                start_time = time.monotonic()

                external_image, wrist_image = read_camera_pair(args, external_camera, wrist_camera)
                if t_step == 0 and args.preview_path is not None:
                    save_preview(args.preview_path, external_image, wrist_image)

                if action_chunk is None or actions_from_chunk_completed >= args.open_loop_horizon:
                    request_data = build_policy_request(
                        args,
                        external_image,
                        wrist_image,
                        controller,
                        instruction,
                        save_policy_preview=t_step == 0,
                    )
                    action_chunk = np.asarray(policy_client.infer(request_data)["actions"], dtype=np.float64)
                    if action_chunk.ndim != 2 or action_chunk.shape[1] != 8:
                        raise RuntimeError(f"Expected action chunk shape [N, 8], got {action_chunk.shape}.")
                    actions_from_chunk_completed = 0
                    print(
                        f"step={t_step} action_chunk={action_chunk.shape} "
                        f"first_joint_target={np.round(action_chunk[0, :7], 3)} gripper={action_chunk[0, 7]:.3f}"
                    )

                action = action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                if args.execute:
                    sent_target = controller.step(action, asynchronous=not args.step_on_space)
                    if args.step_on_space:
                        print(f"executed joint_target={np.round(sent_target, 3)} gripper={action[7]:.3f}")
                else:
                    sent_target = np.asarray(action[:7], dtype=np.float64)

                elapsed = time.monotonic() - start_time
                if not args.step_on_space and elapsed < period_s:
                    time.sleep(period_s - elapsed)
                if not args.execute and (args.step_on_space or t_step % args.open_loop_horizon == 0):
                    print(f"dry_run joint_target={np.round(sent_target, 3)} gripper={action[7]:.3f}")
    except KeyboardInterrupt:
        print("Interrupted, stopping robot.")
    finally:
        if args.execute and controller is not None:
            controller.stop()
        if external_camera is not None:
            external_camera.close()
        if wrist_camera is not None:
            wrist_camera.close()


def list_realsense_cameras() -> None:
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("No RealSense devices found.")
        return
    for device in devices:
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        firmware = device.get_info(rs.camera_info.firmware_version)
        print(f"{serial}\t{name}\tfirmware={firmware}")


def read_camera_pair(
    args: Args,
    external_camera: RealSenseCamera,
    wrist_camera: RealSenseCamera,
) -> tuple[np.ndarray, np.ndarray]:
    external_image = external_camera.read_rgb()
    wrist_image = wrist_camera.read_rgb()
    if args.swap_cameras:
        return wrist_image, external_image
    return external_image, wrist_image


def build_policy_request(
    args: Args,
    external_image: np.ndarray,
    wrist_image: np.ndarray,
    controller: FrankaController,
    instruction: str,
    *,
    save_policy_preview: bool,
) -> dict:
    exterior_policy_image = preprocess_policy_image(external_image, mode=args.policy_image_preprocess)
    wrist_policy_image = preprocess_policy_image(wrist_image, mode=args.policy_image_preprocess)
    if save_policy_preview and args.policy_preview_path is not None:
        save_preview(args.policy_preview_path, exterior_policy_image, wrist_policy_image)

    return {
        "observation/exterior_image_1_left": exterior_policy_image,
        "observation/wrist_image_left": wrist_policy_image,
        "observation/joint_position": controller.joint_position(),
        "observation/gripper_position": controller.gripper_position(),
        "prompt": instruction,
    }


def preprocess_policy_image(image: np.ndarray, *, mode: Literal["match_collected_dataset", "direct"]) -> np.ndarray:
    if mode == "direct":
        return image_tools.resize_with_pad(image, POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE)

    resampling = getattr(Image, "Resampling", Image).BICUBIC
    droid_storage_image = np.asarray(
        Image.fromarray(image).resize((DROID_IMAGE_WIDTH, DROID_IMAGE_HEIGHT), resampling),
        dtype=np.uint8,
    )
    return image_tools.resize_with_pad(droid_storage_image, POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE)


def run_continuous_rollout(
    *,
    args: Args,
    controller: FrankaController,
    external_camera: RealSenseCamera,
    wrist_camera: RealSenseCamera,
    policy_client: websocket_client_policy.WebsocketClientPolicy,
    instruction: str,
    period_s: float,
) -> None:
    t_step = 0
    while t_step < args.max_timesteps:
        external_image, wrist_image = read_camera_pair(args, external_camera, wrist_camera)
        if t_step == 0 and args.preview_path is not None:
            save_preview(args.preview_path, external_image, wrist_image)

        request_data = build_policy_request(
            args,
            external_image,
            wrist_image,
            controller,
            instruction,
            save_policy_preview=t_step == 0,
        )
        action_chunk = np.asarray(policy_client.infer(request_data)["actions"], dtype=np.float64)
        if action_chunk.ndim != 2 or action_chunk.shape[1] != 8:
            raise RuntimeError(f"Expected action chunk shape [N, 8], got {action_chunk.shape}.")

        horizon = min(args.open_loop_horizon, action_chunk.shape[0], args.max_timesteps - t_step)
        actions = action_chunk[:horizon]
        print(
            f"step={t_step} action_chunk={action_chunk.shape} executing={horizon} "
            f"first_joint_target={np.round(actions[0, :7], 3)} gripper={actions[0, 7]:.3f}"
        )
        sent_targets = controller.execute_action_sequence(actions, period_s=period_s)
        print(
            f"executed first_joint_target={np.round(sent_targets[0], 3)} "
            f"last_joint_target={np.round(sent_targets[-1], 3)}"
        )
        t_step += horizon


class KeyboardStepper:
    def __init__(self, *, enabled: bool):
        self._enabled = enabled
        self._old_settings = None

    def __enter__(self):
        if self._enabled:
            if not sys.stdin.isatty():
                raise RuntimeError("--step-on-space requires an interactive terminal.")
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def wait_for_step(self) -> str:
        if not self._enabled:
            return "step"

        print("Press SPACE for one action, q to quit.", flush=True)
        while True:
            readable, _, _ = select.select([sys.stdin], [], [])
            if not readable:
                continue
            char = sys.stdin.read(1)
            if char == " ":
                return "step"
            if char.lower() == "q":
                return "quit"


def save_preview(path: pathlib.Path, external_image: np.ndarray, wrist_image: np.ndarray) -> None:
    height = min(external_image.shape[0], wrist_image.shape[0])
    external = external_image[:height]
    wrist = wrist_image[:height]
    combined = np.concatenate([external, wrist], axis=1)
    Image.fromarray(combined).save(path)
    print(f"Saved camera preview to {path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
