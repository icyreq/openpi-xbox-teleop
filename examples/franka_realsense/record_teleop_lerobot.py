# ruff: noqa: T201

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import pathlib
import queue
import shutil
import struct
import threading
import time
from typing import Any, Literal

import cv2
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from main import RealSenseCamera
from main import list_realsense_cameras
from main import save_preview
import numpy as np
from PIL import Image
import tyro

DROID_CONTROL_HZ = 15
DROID_IMAGE_WIDTH = 320
DROID_IMAGE_HEIGHT = 180
DEFAULT_DATASET_ROOT = pathlib.Path.home() / "franka_realsense_lerobot" / "mani1" / "franka_realsense_droid"
DELETE_CONFIRM_WINDOW_S = 3.0

JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80
JS_EVENT_FORMAT = "IhBB"
JS_EVENT_SIZE = struct.calcsize(JS_EVENT_FORMAT)


class XboxIndex:
    LEFT_X = 0
    LEFT_Y = 1
    LT = 2
    RIGHT_X = 3
    RIGHT_Y = 4
    RT = 5
    DPAD_X = 6
    DPAD_Y = 7

    A = 0
    B = 1
    X = 2
    Y = 3
    LB = 4
    RB = 5
    BACK = 6
    START = 7


@dataclasses.dataclass(frozen=True)
class JoystickAxisMap:
    left_x: int
    left_y: int
    lt: int
    right_x: int
    right_y: int
    rt: int
    dpad_x: int


def joystick_axis_map(layout: str) -> JoystickAxisMap:
    if layout == "xbox_bluetooth":
        return JoystickAxisMap(left_x=0, left_y=1, lt=4, right_x=2, right_y=3, rt=5, dpad_x=6)
    return JoystickAxisMap(left_x=0, left_y=1, lt=2, right_x=3, right_y=4, rt=5, dpad_x=6)


@dataclasses.dataclass
class Args:
    # Dataset.
    repo_id: str = "mani1/franka_realsense_droid"
    dataset_root: pathlib.Path = DEFAULT_DATASET_ROOT
    task: str = "pick up the object"
    append: bool = False
    overwrite: bool = False
    max_episodes: int = 50
    min_episode_frames: int = 10
    max_episode_seconds: float | None = None
    use_videos: bool = False
    image_writer_threads: int = 10
    image_writer_processes: int = 0
    record_queue_size: int = 450
    max_camera_frame_age_s: float = 1.0

    # Franka FCI.
    robot_ip: str = "172.16.0.8"
    franky_dynamics_factor: float = 0.1
    home_dynamics_factor: float = 0.3
    initial_reset: bool = False
    home_pose: tuple[float, float, float, float, float, float] = (0.4, 0.0, 0.5, math.pi, 0.0, 0.0)

    # Xbox controller through Linux joystick API.
    joy_device: pathlib.Path = pathlib.Path("/dev/input/js0")
    control_hz: float = 50.0
    velocity_duration_ms: int | None = 100
    robot_command_hz: float = 15.0
    v_max: float = 0.12
    w_max: float = 0.35
    dead_zone: float = 0.005
    trigger_positive_pressed: bool = True
    joystick_layout: Literal["xbox360", "xbox_bluetooth"] = "xbox360"
    invert_z_axis: bool = True
    debug_joystick: bool = False
    debug_joystick_period: float = 0.5

    # Franka Hand. OpenPI/DROID convention is 0=open, 1=closed.
    enable_gripper: bool = True
    gripper_open_width: float = 0.08
    gripper_closed_width: float = 0.0
    gripper_speed: float = 0.08

    # RealSense.
    external_camera_serial: str | None = None
    wrist_camera_serial: str | None = None
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    camera_warmup_frames: int = 90
    auto_exposure: bool = True
    auto_white_balance: bool = True
    white_balance: float | None = 4500
    assume_realsense_rgb: bool = False
    swap_cameras: bool = False
    list_cameras: bool = False
    preview_path: pathlib.Path | None = pathlib.Path("franka_realsense_record_views.jpg")
    show_preview: bool = True
    preview_hz: float = 15.0
    preview_window_name: str = "Franka RealSense teleop"

    # Stored action cleanup.
    max_recorded_joint_velocity: float = 1.0


class LinuxJoystick:
    def __init__(
        self,
        path: pathlib.Path,
        *,
        num_axes: int = 8,
        num_buttons: int = 12,
        trigger_positive_pressed: bool = True,
    ):
        self._path = pathlib.Path(path)
        self._axes = np.zeros(num_axes, dtype=np.float32)
        self._buttons = [0] * num_buttons
        if trigger_positive_pressed:
            self._axes[XboxIndex.LT] = -1.0
            self._axes[XboxIndex.RT] = -1.0
        else:
            self._axes[XboxIndex.LT] = 1.0
            self._axes[XboxIndex.RT] = 1.0

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._file = None
        self._event_count = 0

    def start(self) -> None:
        try:
            self._file = self._path.open("rb", buffering=0)
        except OSError as exc:
            raise RuntimeError(f"无法打开手柄设备 {self._path}: {exc}") from exc
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._file is not None:
            with contextlib.suppress(OSError):
                self._file.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def snapshot(self) -> tuple[np.ndarray, list[int]]:
        with self._lock:
            return self._axes.copy(), list(self._buttons)

    def event_count(self) -> int:
        with self._lock:
            return self._event_count

    def _read_loop(self) -> None:
        assert self._file is not None
        while not self._stop.is_set():
            try:
                data = self._file.read(JS_EVENT_SIZE)
            except OSError:
                break
            if not data or len(data) != JS_EVENT_SIZE:
                time.sleep(0.001)
                continue

            _event_time_ms, value, event_type, number = struct.unpack(JS_EVENT_FORMAT, data)
            event_type &= ~JS_EVENT_INIT
            with self._lock:
                if event_type == JS_EVENT_AXIS and number < len(self._axes):
                    denom = 32767.0 if value >= 0 else 32768.0
                    self._axes[number] = float(np.clip(value / denom, -1.0, 1.0))
                    self._event_count += 1
                elif event_type == JS_EVENT_BUTTON and number < len(self._buttons):
                    self._buttons[number] = int(value)
                    self._event_count += 1


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float64,
    )


class FrankaTeleopController:
    def __init__(self, args: Args):
        import franky

        self._franky = franky
        self._robot = franky.Robot(args.robot_ip)
        self._robot.recover_from_errors()
        self._robot.relative_dynamics_factor = args.franky_dynamics_factor
        self._gripper = franky.Gripper(args.robot_ip) if args.enable_gripper else None
        self._robot_lock = threading.Lock()
        self._gripper_lock = threading.Lock()

        self._home_pose = args.home_pose
        self._home_dynamics_factor = args.home_dynamics_factor
        self._default_dynamics_factor = args.franky_dynamics_factor
        self._open_width = args.gripper_open_width
        self._closed_width = args.gripper_closed_width
        self._gripper_speed = args.gripper_speed
        self._gripper_action = 0.0
        self._last_q = np.zeros(7, dtype=np.float32)
        self._last_dq = np.zeros(7, dtype=np.float32)
        self._missed_velocity_commands = 0
        self._last_velocity_command_time = 0.0
        self._robot_command_period = 1.0 / max(float(args.robot_command_hz), 1e-6)

    def send_cartesian_velocity(self, values: np.ndarray, *, duration_ms: int) -> None:
        now = time.monotonic()
        if now - self._last_velocity_command_time < self._robot_command_period:
            return
        self._last_velocity_command_time = now
        lin = np.asarray(values[:3], dtype=np.float64).reshape(3, 1)
        ang = np.asarray(values[3:], dtype=np.float64).reshape(3, 1)
        twist = self._franky.Twist(lin, ang)
        motion = self._franky.CartesianVelocityMotion(
            self._franky.RobotVelocity(twist),
            self._franky.Duration(max(1, int(duration_ms))),
            relative_dynamics_factor=self._robot.relative_dynamics_factor,
            ee_frame=None,
        )
        if not self._robot_lock.acquire(blocking=False):
            self._missed_velocity_commands += 1
            if self._missed_velocity_commands % 50 == 1:
                print(f"机器人控制忙, 跳过速度命令。累计跳过={self._missed_velocity_commands}")
            return
        try:
            self._robot.move(motion, asynchronous=True)
            self._missed_velocity_commands = 0
        finally:
            self._robot_lock.release()

    def stop_velocity(self, *, blocking: bool = True) -> None:
        acquired = self._robot_lock.acquire(blocking=blocking)
        if not acquired:
            print("机器人控制忙, 跳过 stop_velocity。")
            return
        try:
            self._robot.move(
                self._franky.CartesianVelocityStopMotion(self._robot.relative_dynamics_factor),
                asynchronous=True,
            )
        finally:
            self._robot_lock.release()

    def recover(self) -> None:
        if not self._robot_lock.acquire(timeout=1.0):
            raise RuntimeError("机器人控制忙, 无法 recover。")
        try:
            with contextlib.suppress(Exception):
                self._robot.move(
                    self._franky.CartesianVelocityStopMotion(self._robot.relative_dynamics_factor),
                    asynchronous=False,
                )
            self._robot.recover_from_errors()
        finally:
            self._robot_lock.release()

    def move_home(self, *, open_gripper: bool) -> None:
        x, y, z, roll, pitch, yaw = self._home_pose
        motion = self._franky.CartesianMotion(
            self._franky.Affine([x, y, z], quat_from_rpy(roll, pitch, yaw)),
            relative_dynamics_factor=self._home_dynamics_factor,
        )
        if not self._robot_lock.acquire(timeout=1.0):
            raise RuntimeError("机器人控制忙, 无法 Home。")
        try:
            old_dynamics = self._robot.relative_dynamics_factor
            self._robot.relative_dynamics_factor = self._home_dynamics_factor
            try:
                with contextlib.suppress(Exception):
                    self._robot.move(
                        self._franky.CartesianVelocityStopMotion(self._home_dynamics_factor),
                        asynchronous=False,
                    )
                self._robot.move(motion, asynchronous=False)
            finally:
                self._robot.relative_dynamics_factor = old_dynamics or self._default_dynamics_factor
        finally:
            self._robot_lock.release()
        if open_gripper:
            self.open_gripper()

    def close_gripper(self) -> None:
        self._gripper_action = 1.0
        if self._gripper is None:
            return
        with self._gripper_lock:
            self._gripper.move_async(self._closed_width, self._gripper_speed)

    def open_gripper(self) -> None:
        self._gripper_action = 0.0
        if self._gripper is None:
            return
        with self._gripper_lock:
            self._gripper.move_async(self._open_width, self._gripper_speed)

    def gripper_action(self) -> float:
        return float(self._gripper_action)

    def gripper_position(self) -> np.ndarray:
        if self._gripper is None:
            return np.zeros(1, dtype=np.float32)
        with self._gripper_lock:
            width = float(self._gripper.width)
        span = max(self._open_width - self._closed_width, 1e-6)
        closed_amount = (self._open_width - width) / span
        return np.asarray([np.clip(closed_amount, 0.0, 1.0)], dtype=np.float32)

    def state_sample(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._robot_lock.acquire(blocking=False):
            return self._last_q.copy(), self._last_dq.copy()
        try:
            state = self._robot.state
            q = np.asarray(state.q, dtype=np.float32).reshape(-1).copy()
            dq = np.asarray(state.dq, dtype=np.float32).reshape(-1).copy()
        finally:
            self._robot_lock.release()
        if q.size != 7 or dq.size != 7:
            raise RuntimeError(f"Expected 7 Franka joints, got q={q.shape}, dq={dq.shape}.")
        self._last_q = q.copy()
        self._last_dq = dq.copy()
        return q, dq

    def shutdown(self) -> None:
        with contextlib.suppress(Exception):
            self.stop_velocity(blocking=False)
        if self._gripper is not None:
            with contextlib.suppress(Exception):
                self._gripper.stop()


class XboxTeleopRuntime:
    def __init__(self, args: Args, joystick: LinuxJoystick, controller: FrankaTeleopController):
        self._args = args
        self._joystick = joystick
        self._controller = controller
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._events: list[str] = []
        self._events_lock = threading.Lock()
        self._recording = False
        self._recording_lock = threading.Lock()
        self._prev_buttons = [0] * 12
        self._estop = False
        self._delete_last_requested_at = 0.0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def set_recording(self, *, value: bool) -> None:
        with self._recording_lock:
            self._recording = value

    def is_recording(self) -> bool:
        with self._recording_lock:
            return self._recording

    def pop_events(self) -> list[str]:
        with self._events_lock:
            events = list(self._events)
            self._events.clear()
        return events

    def _push_event(self, event: str) -> None:
        with self._events_lock:
            self._events.append(event)

    def _run(self) -> None:
        period_s = 1.0 / float(self._args.control_hz)
        duration_ms = self._args.velocity_duration_ms or max(1, round(period_s * 1000))
        next_tick = time.monotonic()
        started_at = next_tick
        last_debug_print = 0.0
        warned_no_events = False

        while not self._stop.is_set():
            axes, buttons = self._joystick.snapshot()
            self._handle_buttons(buttons)

            if self._estop:
                velocity = np.zeros(6, dtype=np.float64)
            else:
                velocity = joystick_to_cartesian_velocity(axes, self._args)

            now = time.monotonic()
            event_count = self._joystick.event_count()
            if not warned_no_events and now - started_at > 2.0 and event_count == 0:
                print(
                    "警告: 还没有收到任何手柄事件。请确认 --joy-device 指向 Xbox, "
                    "例如 /dev/input/js0 或 /dev/input/by-id/...-joystick。"
                )
                warned_no_events = True

            if self._args.debug_joystick and now - last_debug_print >= self._args.debug_joystick_period:
                axis_text = np.array2string(axes[:8], precision=2, suppress_small=True)
                button_text = buttons[:10]
                vel_text = np.array2string(velocity, precision=3, suppress_small=True)
                print(f"手柄调试 axes={axis_text} buttons={button_text} cart_vel={vel_text} events={event_count}")
                last_debug_print = now

            if not self._estop:
                try:
                    self._controller.send_cartesian_velocity(velocity, duration_ms=duration_ms)
                except Exception as exc:
                    print(f"速度控制命令失败: {exc}")

            next_tick += period_s
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()

    def _handle_buttons(self, buttons: list[int]) -> None:
        def rising(idx: int) -> bool:
            return idx < len(buttons) and idx < len(self._prev_buttons) and buttons[idx] == 1 and self._prev_buttons[idx] == 0

        if rising(XboxIndex.Y):
            self._push_event("toggle_record")

        if rising(XboxIndex.X):
            if self.is_recording():
                self._push_event("discard_episode")
            else:
                now = time.monotonic()
                if now - self._delete_last_requested_at <= DELETE_CONFIRM_WINDOW_S:
                    self._delete_last_requested_at = 0.0
                    self._push_event("delete_last_episode")
                else:
                    self._delete_last_requested_at = now
                    print(f"再次按 X 将删除上一条已保存 episode。{DELETE_CONFIRM_WINDOW_S:.0f} 秒内有效。")

        if rising(XboxIndex.B):
            self._estop = not self._estop
            if self._estop:
                with contextlib.suppress(Exception):
                    self._controller.stop_velocity()
                print("已启用速度急停。再次按 B 解除。")
            else:
                print("已解除速度急停。")

        if rising(XboxIndex.BACK):
            try:
                self._controller.recover()
                print("已尝试从机器人错误状态恢复。")
            except Exception as exc:
                print(f"恢复失败: {exc}")

        if rising(XboxIndex.LB):
            self._controller.close_gripper()
            print("夹爪关闭命令。")

        if rising(XboxIndex.RB):
            self._controller.open_gripper()
            print("夹爪打开命令。")

        if not self.is_recording():
            if rising(XboxIndex.A):
                try:
                    self._controller.move_home(open_gripper=False)
                    print("已回到 Home。")
                except Exception as exc:
                    print(f"Home 失败: {exc}")
            if rising(XboxIndex.START):
                try:
                    self._controller.move_home(open_gripper=True)
                    print("已复位: Home + 打开夹爪。")
                except Exception as exc:
                    print(f"复位失败: {exc}")

        self._prev_buttons = list(buttons)


def joystick_to_cartesian_velocity(axes: np.ndarray, args: Args) -> np.ndarray:
    mapping = joystick_axis_map(args.joystick_layout)

    def axis(idx: int, default: float = 0.0) -> float:
        return float(axes[idx]) if idx < axes.size else default

    def dead(v: float) -> float:
        return v if abs(v) > args.dead_zone else 0.0

    def trigger(v: float) -> float:
        if args.trigger_positive_pressed:
            return dead(0.5 * (v + 1.0))
        return dead(0.5 * (1.0 - v))

    lx = dead(axis(mapping.left_x))
    ly = dead(axis(mapping.left_y))
    rx = dead(axis(mapping.right_x))
    ry = dead(axis(mapping.right_y))
    dpad_x = dead(axis(mapping.dpad_x))
    lt = trigger(axis(mapping.lt, -1.0 if args.trigger_positive_pressed else 1.0))
    rt = trigger(axis(mapping.rt, -1.0 if args.trigger_positive_pressed else 1.0))

    return np.asarray(
        [
            -ly * args.v_max,
            -lx * args.v_max,
            (-ry if args.invert_z_axis else ry) * args.v_max,
            -dpad_x * args.w_max,
            (rt - lt) * args.w_max,
            -rx * args.w_max,
        ],
        dtype=np.float64,
    )


@dataclasses.dataclass
class CameraPairSample:
    external_image: np.ndarray
    wrist_image: np.ndarray
    timestamp: float


class CameraPairReader:
    def __init__(self, args: Args, external_camera: RealSenseCamera, wrist_camera: RealSenseCamera):
        self._args = args
        self._external_camera = external_camera
        self._wrist_camera = wrist_camera
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sample: CameraPairSample | None = None
        self._error: Exception | None = None
        self._frame_count = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def latest(self, *, max_age_s: float | None = None) -> CameraPairSample | None:
        with self._lock:
            if self._error is not None:
                raise RuntimeError(f"RealSense 采集线程失败: {self._error}") from self._error
            sample = self._sample
        if sample is None:
            return None
        if max_age_s is not None and time.monotonic() - sample.timestamp > max_age_s:
            return None
        return sample

    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                external_image, wrist_image = read_camera_pair(self._args, self._external_camera, self._wrist_camera)
            except Exception as exc:
                with self._lock:
                    self._error = exc
                return

            sample = CameraPairSample(external_image=external_image, wrist_image=wrist_image, timestamp=time.monotonic())
            with self._lock:
                self._sample = sample
                self._frame_count += 1


@dataclasses.dataclass
class RecordFrameJob:
    frame: dict[str, Any]


class AsyncEpisodeRecorder:
    def __init__(
        self,
        *,
        dataset: LeRobotDataset,
        max_queue_size: int,
    ):
        self._dataset = dataset
        self._queue: queue.Queue[RecordFrameJob | None] = queue.Queue(maxsize=max_queue_size)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._written_frames = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            with contextlib.suppress(queue.Full):
                self._queue.put_nowait(None)
            self._thread.join(timeout=5.0)

    def enqueue(self, job: RecordFrameJob) -> None:
        self.raise_if_failed()
        self._queue.put(job, timeout=2.0)

    def wait_until_done(self) -> None:
        self.raise_if_failed()
        self._queue.join()
        self.raise_if_failed()

    def pending_frames(self) -> int:
        return self._queue.qsize()

    def written_frames(self) -> int:
        return self._written_frames

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"录制写入线程失败: {self._error}") from self._error

    def _run(self) -> None:
        while not self._stop.is_set():
            job = self._queue.get()
            try:
                if job is None:
                    return
                self._dataset.add_frame(job.frame)
                self._written_frames += 1
            except Exception as exc:
                self._error = exc
                self._drain_queue()
                return
            finally:
                self._queue.task_done()

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
            else:
                self._queue.task_done()


def dataset_features(*, use_videos: bool) -> dict[str, Any]:
    image_feature = {
        "dtype": "video" if use_videos else "image",
        "shape": (DROID_IMAGE_HEIGHT, DROID_IMAGE_WIDTH, 3),
        "names": ["height", "width", "channel"],
    }
    return {
        "exterior_image_1_left": dict(image_feature),
        "exterior_image_2_left": dict(image_feature),
        "wrist_image_left": dict(image_feature),
        "joint_position": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["joint_position"],
        },
        "gripper_position": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["gripper_position"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["actions"],
        },
    }


def dataset_root(args: Args) -> pathlib.Path:
    root = pathlib.Path(args.dataset_root).expanduser()
    root.parent.mkdir(parents=True, exist_ok=True)
    return root


def is_lerobot_dataset_root(root: pathlib.Path) -> bool:
    meta = root / "meta"
    return all((meta / name).is_file() for name in ("info.json", "tasks.jsonl", "episodes.jsonl"))


def has_saved_episode_files(root: pathlib.Path) -> bool:
    data_dir = root / "data"
    videos_dir = root / "videos"
    return (data_dir.is_dir() and any(data_dir.rglob("*.parquet"))) or (
        videos_dir.is_dir() and any(videos_dir.rglob("*.mp4"))
    )


def load_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def delete_last_saved_episode(dataset: LeRobotDataset) -> tuple[int, int]:
    root = pathlib.Path(dataset.root)
    info_path = root / "meta" / "info.json"
    episodes_path = root / "meta" / "episodes.jsonl"
    stats_path = root / "meta" / "episodes_stats.jsonl"

    # 删除只允许作用在最后一个连续 episode，避免破坏 LeRobot 的全局帧索引和文件命名约定。
    info = load_json(info_path)
    episodes = load_jsonl(episodes_path)
    episode_stats = load_jsonl(stats_path)
    total_episodes = int(info["total_episodes"])
    if total_episodes <= 0 or not episodes:
        print("当前数据集没有可删除的已保存 episode。")
        return -1, 0

    last_episode = episodes[-1]
    last_index = int(last_episode["episode_index"])
    if last_index != total_episodes - 1:
        raise RuntimeError(
            f"最后一条 episode_index={last_index}, 但 total_episodes={total_episodes}。"
            "为避免破坏数据集索引，已拒绝删除。"
        )
    last_length = int(last_episode["length"])

    data_path = root / dataset.meta.get_data_file_path(last_index)
    data_path.unlink(missing_ok=True)
    for video_key in dataset.meta.video_keys:
        video_path = root / dataset.meta.get_video_file_path(last_index, video_key)
        video_path.unlink(missing_ok=True)

    episodes = episodes[:-1]
    episode_stats = [row for row in episode_stats if int(row["episode_index"]) != last_index]
    info["total_episodes"] = total_episodes - 1
    info["total_frames"] = max(0, int(info["total_frames"]) - last_length)
    info["total_videos"] = max(0, int(info.get("total_videos", 0)) - len(dataset.meta.video_keys))
    info["total_chunks"] = max(1, (info["total_episodes"] + int(info["chunks_size"]) - 1) // int(info["chunks_size"]))
    info["splits"] = {"train": f"0:{info['total_episodes']}"}

    write_json(info_path, info)
    write_jsonl(episodes_path, episodes)
    write_jsonl(stats_path, episode_stats)

    return last_index, last_length


def refresh_dataset_metadata(dataset: LeRobotDataset) -> None:
    root = pathlib.Path(dataset.root)
    info = load_json(root / "meta" / "info.json")
    episodes = {int(row["episode_index"]): row for row in load_jsonl(root / "meta" / "episodes.jsonl")}
    episode_stats = {int(row["episode_index"]): row["stats"] for row in load_jsonl(root / "meta" / "episodes_stats.jsonl")}

    # 同步内存中的 meta，保证删除后继续保存新 episode 时索引和全局帧数接着磁盘状态走。
    dataset.meta.info = info
    dataset.meta.episodes = episodes
    dataset.meta.episodes_stats = episode_stats
    dataset.meta.stats = {}
    dataset.episode_buffer = dataset.create_episode_buffer()


def validate_dataset_storage_mode(dataset: LeRobotDataset, args: Args) -> None:
    expected_dtype = "video" if args.use_videos else "image"
    camera_keys = ("exterior_image_1_left", "exterior_image_2_left", "wrist_image_left")
    mismatched = {
        key: dataset.features.get(key, {}).get("dtype")
        for key in camera_keys
        if dataset.features.get(key, {}).get("dtype") != expected_dtype
    }
    if mismatched:
        actual = ", ".join(f"{key}={dtype}" for key, dtype in mismatched.items())
        raise RuntimeError(
            f"已有数据集的相机存储格式和当前参数不一致: {actual}, 期望 {expected_dtype}。"
            "如果要生成 mp4, 请换一个 --dataset-root 或加 --overwrite 重建, 并使用 --use-videos。"
        )


def create_or_load_dataset(args: Args) -> LeRobotDataset:
    root = dataset_root(args)
    if root.exists():
        if args.overwrite:
            shutil.rmtree(root)
        elif args.append:
            if not is_lerobot_dataset_root(root):
                if has_saved_episode_files(root):
                    raise RuntimeError(
                        f"目录 {root} 里有 episode 文件, 但 LeRobot meta 不完整。"
                        "请先手动备份/检查该目录, 然后用 --overwrite 重建, 或换一个 --dataset-root。"
                    )
                print(f"检测到不完整的本地数据集目录, 将重新创建: {root}")
                shutil.rmtree(root)
            else:
                dataset = LeRobotDataset(args.repo_id, root=root)
                validate_dataset_storage_mode(dataset, args)
                dataset.episode_buffer = dataset.create_episode_buffer()
                if args.image_writer_processes or args.image_writer_threads:
                    dataset.start_image_writer(args.image_writer_processes, args.image_writer_threads)
                return dataset
        elif not is_lerobot_dataset_root(root):
            raise FileExistsError(
                f"目录已经存在但不是完整 LeRobot 数据集: {root}。"
                "如果这里没有要保留的数据, 请加 --overwrite 重建。"
            )
        else:
            raise FileExistsError(f"数据集已经存在: {root}。继续追加请加 --append, 重新创建请加 --overwrite。")

    if root.exists():
        # Happens when --append saw an incomplete local root and removed it above.
        shutil.rmtree(root)

    if args.append and root.exists():
        try:
            dataset = LeRobotDataset(args.repo_id, root=root)
            validate_dataset_storage_mode(dataset, args)
            dataset.episode_buffer = dataset.create_episode_buffer()
            if args.image_writer_processes or args.image_writer_threads:
                dataset.start_image_writer(args.image_writer_processes, args.image_writer_threads)
            return dataset
        except Exception as exc:
            raise RuntimeError(f"加载已有 LeRobot 数据集失败: {root}") from exc

    return LeRobotDataset.create(
        repo_id=args.repo_id,
        root=root,
        robot_type="franka_realsense",
        fps=DROID_CONTROL_HZ,
        features=dataset_features(use_videos=args.use_videos),
        use_videos=args.use_videos,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )


def resize_for_droid(image: np.ndarray) -> np.ndarray:
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    return np.asarray(Image.fromarray(image).resize((DROID_IMAGE_WIDTH, DROID_IMAGE_HEIGHT), resampling), dtype=np.uint8)


def read_camera_pair(args: Args, external_camera: RealSenseCamera, wrist_camera: RealSenseCamera) -> tuple[np.ndarray, np.ndarray]:
    external_image = external_camera.read_rgb()
    wrist_image = wrist_camera.read_rgb()
    if args.swap_cameras:
        return wrist_image, external_image
    return external_image, wrist_image


def make_display_preview(
    external_image: np.ndarray,
    wrist_image: np.ndarray,
    *,
    recording: bool,
    active_frames: int,
) -> np.ndarray:
    display_height = min(360, external_image.shape[0], wrist_image.shape[0])

    def resize_to_height(image: np.ndarray) -> np.ndarray:
        scale = display_height / float(image.shape[0])
        display_width = max(1, round(image.shape[1] * scale))
        return cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_AREA)

    external = resize_to_height(external_image)
    wrist = resize_to_height(wrist_image)
    combined_rgb = np.concatenate([external, wrist], axis=1)
    combined_bgr = np.ascontiguousarray(combined_rgb[..., ::-1])

    status = f"{'RECORDING' if recording else 'READY'} frames={active_frames}"
    cv2.putText(combined_bgr, "LEFT: exterior_image_1_left", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 255, 40), 2)
    cv2.putText(
        combined_bgr,
        "RIGHT: wrist_image_left",
        (external.shape[1] + 12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (40, 255, 40),
        2,
    )
    cv2.putText(combined_bgr, status, (12, display_height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    return combined_bgr


def show_camera_preview(
    args: Args,
    external_image: np.ndarray,
    wrist_image: np.ndarray,
    *,
    recording: bool,
    active_frames: int,
) -> None:
    preview = make_display_preview(
        external_image,
        wrist_image,
        recording=recording,
        active_frames=active_frames,
    )
    cv2.imshow(args.preview_window_name, preview)
    cv2.waitKey(1)


def print_operation_help(args: Args, dataset: LeRobotDataset) -> None:
    mapping = joystick_axis_map(args.joystick_layout)
    print("\n=== Franka + RealSense 遥操作数据采集 ===")
    print(f"数据集目录: {dataset.root}")
    print(f"已有 episode 数: {dataset.meta.total_episodes}")
    print(f"相机存储格式: {'mp4 videos' if args.use_videos else 'parquet embedded images'}")
    print(f"后台录制队列上限: {args.record_queue_size} 帧")
    print(f"本次任务文本: {args.task!r}")
    print(f"左侧/外部相机 serial: {args.external_camera_serial}")
    print(f"右侧/腕部相机 serial: {args.wrist_camera_serial}")
    print(f"手柄设备: {args.joy_device}")
    print(f"手柄布局: {args.joystick_layout}")
    print(
        "手柄轴映射: "
        f"LX={mapping.left_x}, LY={mapping.left_y}, RX={mapping.right_x}, RY={mapping.right_y}, "
        f"LT={mapping.lt}, RT={mapping.rt}, DPAD_X={mapping.dpad_x}"
    )
    print(f"速度命令保持时间: {args.velocity_duration_ms} ms")
    print(f"机器人速度命令发送频率: {args.robot_command_hz} Hz")
    print("\n手柄操作:")
    print("  左摇杆: XY 平移")
    print("  右摇杆上下: Z 平移")
    print("  右摇杆左右: Yaw 角速度")
    print("  LT / RT: Pitch 角速度")
    print("  D-pad 左右: Roll 角速度")
    print("  LB / RB: 关闭 / 打开夹爪")
    print("  Y: 开始录制; 录制中再次按 Y 保存当前 episode")
    print("  X: 仅在录制中丢弃当前还没保存的 episode; 保存后的 episode 不会被 X 删除")
    print("  B: 速度急停开关")
    print("  A: Home")
    print("  START: Home + 打开夹爪")
    print("  BACK: 从机器人错误状态恢复")
    print(f"  Z 方向反向: {args.invert_z_axis}。如果方向不符合习惯可加 --no-invert-z-axis")
    print("\n相机预览:")
    print("  预览窗口左半边 = exterior_image_1_left, 右半边 = wrist_image_left")
    print("  如果左右相机反了, 重新启动时加 --swap-cameras")
    print("  如果摇杆不动, 先加 --debug-joystick 看 axes 和 cart_vel 是否变化")
    print("  如果右摇杆或扳机轴不对, 试试 --joystick-layout xbox_bluetooth")
    print("  如果希望生成 videos/*.mp4, 用新的 --dataset-root 或 --overwrite 重建, 并加 --use-videos")
    print("  退出脚本按 Ctrl+C\n")


def add_record_frame(
    *,
    args: Args,
    dataset: LeRobotDataset,
    controller: FrankaTeleopController,
    external_image: np.ndarray,
    wrist_image: np.ndarray,
    frame_index: int,
) -> None:
    if frame_index == 0 and args.preview_path is not None:
        save_preview(args.preview_path, external_image, wrist_image)

    dataset.add_frame(
        make_record_frame(
            args=args,
            controller=controller,
            external_image=external_image,
            wrist_image=wrist_image,
        )
    )


def make_record_frame(
    *,
    args: Args,
    controller: FrankaTeleopController,
    external_image: np.ndarray,
    wrist_image: np.ndarray,
) -> dict[str, Any]:
    joint_position, joint_velocity = controller.state_sample()
    clipped_joint_velocity = np.clip(
        joint_velocity,
        -float(args.max_recorded_joint_velocity),
        float(args.max_recorded_joint_velocity),
    )
    action = np.concatenate(
        [clipped_joint_velocity.astype(np.float32), np.asarray([controller.gripper_action()], dtype=np.float32)]
    )

    exterior = resize_for_droid(external_image)
    wrist = resize_for_droid(wrist_image)
    return {
        "exterior_image_1_left": exterior,
        "exterior_image_2_left": exterior.copy(),
        "wrist_image_left": wrist,
        "joint_position": joint_position.astype(np.float32),
        "gripper_position": controller.gripper_position(),
        "actions": action.astype(np.float32),
        "task": args.task,
    }


def main(args: Args) -> None:
    if args.list_cameras:
        list_realsense_cameras()
        return
    if not args.external_camera_serial or not args.wrist_camera_serial:
        raise ValueError("请同时设置 --external-camera-serial 和 --wrist-camera-serial。")
    if args.external_camera_serial == args.wrist_camera_serial:
        raise ValueError("外部相机和腕部相机的 serial 不能相同。")

    dataset = create_or_load_dataset(args)
    print_operation_help(args, dataset)

    joystick = LinuxJoystick(args.joy_device, trigger_positive_pressed=args.trigger_positive_pressed)
    controller: FrankaTeleopController | None = None
    runtime: XboxTeleopRuntime | None = None
    external_camera: RealSenseCamera | None = None
    wrist_camera: RealSenseCamera | None = None
    camera_reader: CameraPairReader | None = None
    recorder: AsyncEpisodeRecorder | None = None

    episodes_saved = 0
    recording = False
    active_frames = 0
    episode_started_at = 0.0
    next_record_time = 0.0
    next_preview_time = 0.0
    record_period = 1.0 / DROID_CONTROL_HZ
    preview_period = 1.0 / max(float(args.preview_hz), 1e-6)
    latest_sample: CameraPairSample | None = None
    preview_enabled = bool(args.show_preview)

    def finish_episode(*, discard: bool) -> None:
        nonlocal recording, active_frames, episodes_saved
        if recorder is not None:
            print(f"等待后台录制写入完成... pending={recorder.pending_frames()}")
            recorder.wait_until_done()
        if discard or active_frames < args.min_episode_frames:
            dataset.clear_episode_buffer()
            reason = "用户丢弃" if discard else f"帧数少于最小值 {args.min_episode_frames}"
            print(f"已丢弃当前未保存 episode。原因: {reason}, 帧数={active_frames}")
        else:
            dataset.save_episode()
            episodes_saved += 1
            print(
                f"已保存 episode。帧数={active_frames} "
                f"本次已保存={episodes_saved}/{args.max_episodes} 总数={dataset.meta.total_episodes}"
            )
        recording = False
        active_frames = 0
        if runtime is not None:
            runtime.set_recording(value=False)

    def reload_dataset_after_delete(new_dataset: LeRobotDataset) -> None:
        nonlocal dataset
        with contextlib.suppress(Exception):
            dataset.stop_image_writer()
        dataset = new_dataset
        if args.image_writer_processes or args.image_writer_threads:
            dataset.start_image_writer(args.image_writer_processes, args.image_writer_threads)

    try:
        joystick.start()
        controller = FrankaTeleopController(args)
        if args.initial_reset:
            print("正在执行初始复位...")
            controller.move_home(open_gripper=True)

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

        runtime = XboxTeleopRuntime(args, joystick, controller)
        runtime.start()
        camera_reader = CameraPairReader(args, external_camera, wrist_camera)
        camera_reader.start()
        recorder = AsyncEpisodeRecorder(
            dataset=dataset,
            max_queue_size=args.record_queue_size,
        )
        recorder.start()

        print("等待 RealSense 首帧...")
        while camera_reader.latest() is None:
            time.sleep(0.01)
        print("RealSense 后台采集已启动。")

        while episodes_saved < args.max_episodes:
            now = time.monotonic()
            if recorder is not None:
                recorder.raise_if_failed()
            for event in runtime.pop_events():
                if event == "toggle_record":
                    if recording:
                        finish_episode(discard=False)
                    else:
                        if camera_reader.latest(max_age_s=args.max_camera_frame_age_s) is None:
                            print("相机最新帧过旧或还未就绪, 暂不开始录制。")
                            continue
                        dataset.clear_episode_buffer()
                        recording = True
                        active_frames = 0
                        episode_started_at = now
                        next_record_time = now
                        runtime.set_recording(value=True)
                        print(f"开始录制当前 episode。任务: {args.task!r}")
                elif event == "discard_episode" and recording:
                    finish_episode(discard=True)
                elif event == "delete_last_episode" and not recording:
                    previous_total_episodes = dataset.meta.total_episodes
                    deleted_index, deleted_frames = delete_last_saved_episode(dataset)
                    remaining_episodes = max(0, previous_total_episodes - 1)
                    if deleted_index >= 0 and remaining_episodes > 0:
                        new_dataset = LeRobotDataset(args.repo_id, root=dataset.root)
                        new_dataset.episode_buffer = new_dataset.create_episode_buffer()
                        reload_dataset_after_delete(new_dataset)
                    else:
                        refresh_dataset_metadata(dataset)
                    if deleted_index >= 0:
                        print(
                            f"已删除上一条 episode: index={deleted_index}, frames={deleted_frames}, "
                            f"当前总数={remaining_episodes}"
                        )

            if recording and args.max_episode_seconds is not None and now - episode_started_at >= args.max_episode_seconds:
                finish_episode(discard=False)
                continue

            if preview_enabled and now >= next_preview_time:
                try:
                    assert camera_reader is not None
                    latest_sample = camera_reader.latest(max_age_s=args.max_camera_frame_age_s)
                    if latest_sample is None:
                        raise RuntimeError("RealSense 最新帧过旧。")
                    show_camera_preview(
                        args,
                        latest_sample.external_image,
                        latest_sample.wrist_image,
                        recording=recording,
                        active_frames=active_frames,
                    )
                except Exception as exc:
                    preview_enabled = False
                    print(f"实时相机预览启动失败, 已关闭预览窗口。错误: {exc}")
                next_preview_time = now + preview_period

            if recording and now >= next_record_time:
                assert camera_reader is not None
                assert recorder is not None
                latest_sample = camera_reader.latest(max_age_s=args.max_camera_frame_age_s)
                if latest_sample is None:
                    print("RealSense 帧超时, 暂停录制当前帧; 如果持续出现, 检查 USB 带宽/线缆/相机。")
                    next_record_time = time.monotonic() + record_period
                    continue
                if active_frames == 0 and args.preview_path is not None:
                    save_preview(args.preview_path, latest_sample.external_image, latest_sample.wrist_image)
                frame = make_record_frame(
                    args=args,
                    controller=controller,
                    external_image=latest_sample.external_image,
                    wrist_image=latest_sample.wrist_image,
                )
                recorder.enqueue(
                    RecordFrameJob(
                        frame=frame,
                    )
                )
                active_frames += 1
                if active_frames % DROID_CONTROL_HZ == 0:
                    print(
                        f"正在录制... 当前 episode 帧数={active_frames} "
                        f"pending_write={recorder.pending_frames()}"
                    )
                next_record_time += record_period
                if next_record_time < time.monotonic() - record_period:
                    next_record_time = time.monotonic()

            time.sleep(0.003)

    except KeyboardInterrupt:
        print("收到 Ctrl+C, 正在退出。")
        if recording:
            finish_episode(discard=False)
    finally:
        if runtime is not None:
            runtime.stop()
        if recorder is not None:
            with contextlib.suppress(Exception):
                recorder.wait_until_done()
            recorder.stop()
        if camera_reader is not None:
            camera_reader.stop()
        if controller is not None:
            controller.shutdown()
        joystick.stop()
        if external_camera is not None:
            external_camera.close()
        if wrist_camera is not None:
            wrist_camera.close()
        if args.show_preview:
            with contextlib.suppress(Exception):
                cv2.destroyWindow(args.preview_window_name)
        with contextlib.suppress(Exception):
            dataset.stop_image_writer()
        print(f"数据集目录: {dataset.root}")


if __name__ == "__main__":
    main(tyro.cli(Args))
