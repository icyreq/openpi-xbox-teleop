"""Convert Franka RealSense actions to DROID-style normalized joint-delta actions.

The raw Franka RealSense dataset stores measured joint velocities in ``actions``.
For better alignment with the DROID policy checkpoint, this script builds a new
LeRobot dataset where:

    actions[:7] = (joint_position[t + 1] - joint_position[t]) / 0.2
    actions[7] = gripper_position[t + 1]

The first seven dimensions are therefore normalized one-step joint deltas. The
gripper dimension remains an absolute DROID gripper position, where 0 is open
and 1 is closed.
"""

from __future__ import annotations

import json
import pathlib
import shutil
from typing import Any

import numpy as np
import pandas as pd
import tyro


DROID_MAX_JOINT_DELTA = 0.2
DEFAULT_SOURCE_ROOT = pathlib.Path("/home/nvidia/lixu_thor/franka_realsense_droid_video")
DEFAULT_OUTPUT_ROOT = pathlib.Path("/home/nvidia/lixu_thor/franka_realsense_droid_video_droid_action")


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    """Load LeRobot jsonl metadata while preserving episode order."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    """Write LeRobot jsonl metadata with one compact JSON object per line."""
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _as_vector_batch(values: pd.Series, *, dim: int, key: str) -> np.ndarray:
    """Convert a parquet vector column into a dense float32 matrix."""
    batch = np.stack([np.asarray(value, dtype=np.float32).reshape(-1) for value in values.to_list()])
    if batch.ndim != 2 or batch.shape[1] != dim:
        raise ValueError(f"Expected {key} to have shape [N, {dim}], got {batch.shape}.")
    return batch


def _as_gripper_batch(values: pd.Series) -> np.ndarray:
    """Convert scalar or length-one gripper entries into a dense [N, 1] matrix."""
    parsed = []
    for value in values.to_list():
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size != 1:
            raise ValueError(f"Expected gripper_position entries to be scalar or length 1, got shape {array.shape}.")
        parsed.append(array)
    return np.stack(parsed)


def _feature_stats(array: np.ndarray) -> dict[str, list[float] | list[int]]:
    """Compute LeRobot-compatible per-episode statistics for one feature."""
    return {
        "min": np.min(array, axis=0).astype(float).tolist(),
        "max": np.max(array, axis=0).astype(float).tolist(),
        "mean": np.mean(array, axis=0).astype(float).tolist(),
        "std": np.std(array, axis=0).astype(float).tolist(),
        "count": [int(array.shape[0])],
    }


def _episode_index_from_path(path: pathlib.Path) -> int:
    """Parse an episode index from a standard LeRobot parquet filename."""
    stem = path.stem
    prefix = "episode_"
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected episode parquet name: {path.name}")
    return int(stem[len(prefix) :])


def _next_step_targets(joints: np.ndarray, gripper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build next-step joint and gripper targets with terminal state repetition."""
    next_joints = joints.copy()
    next_gripper = gripper.copy()
    if len(joints) > 1:
        next_joints[:-1] = joints[1:]
        next_gripper[:-1] = gripper[1:]
    next_joints[-1] = joints[-1]
    next_gripper[-1] = gripper[-1]
    return next_joints, next_gripper


def _scale_overrange_joint_actions(joint_actions: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Project overrange joint commands with DROID's full-vector scaling rule."""
    max_abs = np.max(np.abs(joint_actions), axis=1)
    scale = np.maximum(1.0, max_abs)
    scaled = joint_actions / scale[:, None]
    return scaled.astype(np.float32), int(np.sum(max_abs > 1.0)), float(np.max(max_abs))


def _validate_joint_action_range(joint_actions: np.ndarray, *, path: pathlib.Path) -> tuple[int, float]:
    """Require normalized joint-delta actions to fit DROID's [-1, 1] command range."""
    max_abs = np.max(np.abs(joint_actions), axis=1)
    overrange_count = int(np.sum(max_abs > 1.0))
    max_observed = float(np.max(max_abs))
    if overrange_count:
        raise ValueError(
            f"{path} has {overrange_count} DROID joint actions outside [-1, 1]; "
            f"max row abs={max_observed:.6f}. Inspect the action scale, or rerun with "
            "--scale-overrange-actions to apply DROID-style full-vector scaling."
        )
    return overrange_count, max_observed


def _convert_episode(path: pathlib.Path, *, scale_overrange_actions: bool) -> tuple[dict[str, list[float] | list[int]], dict[str, Any]]:
    """Replace one episode's actions with DROID-style joint delta plus gripper position."""
    df = pd.read_parquet(path)
    joints = _as_vector_batch(df["joint_position"], dim=7, key="joint_position")
    gripper = _as_gripper_batch(df["gripper_position"])

    # Use observed next states to define the command that would advance one 15 Hz step.
    next_joints, next_gripper = _next_step_targets(joints, gripper)
    joint_actions = (next_joints - joints) / DROID_MAX_JOINT_DELTA

    # Keep the official command envelope explicit; current data should not need scaling.
    if scale_overrange_actions:
        joint_actions, overrange_count, max_observed = _scale_overrange_joint_actions(joint_actions)
    else:
        overrange_count, max_observed = _validate_joint_action_range(joint_actions, path=path)

    # Store normalized joint-delta actions and absolute next gripper position.
    actions = np.concatenate([joint_actions.astype(np.float32), next_gripper.astype(np.float32)], axis=1)
    df["actions"] = [row for row in actions]
    df.to_parquet(path, index=False)

    report = {
        "frames": int(len(df)),
        "overrange_count": overrange_count,
        "max_abs_joint_action": max_observed,
        "p99_abs_joint_action": float(np.percentile(np.max(np.abs(joint_actions), axis=1), 99)),
    }
    return _feature_stats(actions), report


def _copy_dataset(source_root: pathlib.Path, output_root: pathlib.Path, *, overwrite: bool) -> None:
    """Copy the source LeRobot tree so the original dataset remains untouched."""
    if source_root.resolve() == output_root.resolve():
        raise ValueError("source_root and output_root must be different.")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)
    shutil.copytree(source_root, output_root)


def main(
    source_root: pathlib.Path = DEFAULT_SOURCE_ROOT,
    output_root: pathlib.Path = DEFAULT_OUTPUT_ROOT,
    *,
    overwrite: bool = False,
    scale_overrange_actions: bool = False,
) -> None:
    """Create a DROID-action-aligned Franka RealSense LeRobot dataset."""
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not (source_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"LeRobot metadata not found under {source_root}")

    _copy_dataset(source_root, output_root, overwrite=overwrite)

    # Convert all episode parquet files and collect replacement action statistics.
    action_stats_by_episode: dict[int, dict[str, list[float] | list[int]]] = {}
    reports: dict[int, dict[str, Any]] = {}
    episode_paths = sorted((output_root / "data").glob("chunk-*/episode_*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(f"No episode parquet files found under {output_root / 'data'}")
    for episode_path in episode_paths:
        episode_index = _episode_index_from_path(episode_path)
        stats, report = _convert_episode(episode_path, scale_overrange_actions=scale_overrange_actions)
        action_stats_by_episode[episode_index] = stats
        reports[episode_index] = report

    # Replace only action statistics; observation/video metadata stays unchanged.
    stats_path = output_root / "meta" / "episodes_stats.jsonl"
    stats_rows = _load_jsonl(stats_path)
    for row in stats_rows:
        episode_index = int(row["episode_index"])
        row["stats"]["actions"] = action_stats_by_episode[episode_index]
    _write_jsonl(stats_path, stats_rows)

    total_frames = sum(report["frames"] for report in reports.values())
    total_overrange = sum(report["overrange_count"] for report in reports.values())
    max_abs = max(report["max_abs_joint_action"] for report in reports.values())
    print(f"Converted {len(episode_paths)} episodes / {total_frames} frames.")
    print(f"Source dataset: {source_root}")
    print(f"Output dataset: {output_root}")
    print(f"Overrange joint-action rows: {total_overrange}")
    print(f"Max abs normalized joint action: {max_abs:.6f}")


if __name__ == "__main__":
    tyro.cli(main)
