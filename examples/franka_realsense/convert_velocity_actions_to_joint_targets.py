"""Convert Franka RealSense LeRobot actions from joint velocity to next joint state.

The collected dataset stores ``actions`` as 7 joint velocities plus one gripper command.
This script creates a new LeRobot dataset whose ``actions`` are absolute next-step
joint/gripper states:

    actions[:-1] = states[1:]
    actions[-1] = states[-1]

Training configs can then apply OpenPI's DeltaActions transform to train on joint
deltas relative to the current observation while keeping inference outputs absolute.
"""

from __future__ import annotations

import json
import pathlib
import shutil
from typing import Any

import numpy as np
import pandas as pd
import tyro


DEFAULT_SOURCE_ROOT = pathlib.Path("/home/nvidia/lixu_thor/franka_realsense_droid_video")
DEFAULT_OUTPUT_ROOT = pathlib.Path("/home/nvidia/lixu_thor/franka_realsense_droid_video_joint_action")


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    """Read a jsonl metadata file into dictionaries while preserving row order."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries back to jsonl using one compact JSON object per line."""
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _as_vector_batch(values: pd.Series, *, dim: int, key: str) -> np.ndarray:
    """Convert a parquet vector column into a dense float32 array with the expected width."""
    batch = np.stack([np.asarray(value, dtype=np.float32) for value in values.to_list()])
    if batch.ndim != 2 or batch.shape[1] != dim:
        raise ValueError(f"Expected {key} to have shape [N, {dim}], got {batch.shape}.")
    return batch


def _as_gripper_batch(values: pd.Series) -> np.ndarray:
    """Convert scalar or one-element gripper values into a dense [N, 1] float32 array."""
    parsed = []
    for value in values.to_list():
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size != 1:
            raise ValueError(f"Expected gripper_position entries to be scalar or length 1, got shape {array.shape}.")
        parsed.append(array)
    return np.stack(parsed)


def _feature_stats(array: np.ndarray) -> dict[str, list[float] | list[int]]:
    """Compute LeRobot-compatible per-episode statistics for a dense feature array."""
    return {
        "min": np.min(array, axis=0).astype(float).tolist(),
        "max": np.max(array, axis=0).astype(float).tolist(),
        "mean": np.mean(array, axis=0).astype(float).tolist(),
        "std": np.std(array, axis=0).astype(float).tolist(),
        "count": [int(array.shape[0])],
    }


def _episode_index_from_path(path: pathlib.Path) -> int:
    """Parse the numeric episode index from a standard LeRobot parquet filename."""
    stem = path.stem
    prefix = "episode_"
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected episode parquet name: {path.name}")
    return int(stem[len(prefix) :])


def _convert_episode(path: pathlib.Path) -> dict[str, list[float] | list[int]]:
    """Replace one episode's velocity actions with next-step absolute joint/gripper states."""
    df = pd.read_parquet(path)
    joints = _as_vector_batch(df["joint_position"], dim=7, key="joint_position")
    gripper = _as_gripper_batch(df["gripper_position"])

    # Build the absolute state trajectory used as the target action sequence.
    states = np.concatenate([joints, gripper], axis=1).astype(np.float32)
    actions = states.copy()
    if len(actions) > 1:
        actions[:-1] = states[1:]
    actions[-1] = states[-1]

    # Store each action row as an 8D float32 vector, matching the existing LeRobot schema.
    df["actions"] = [row for row in actions]
    df.to_parquet(path, index=False)
    return _feature_stats(actions)


def _copy_dataset(source_root: pathlib.Path, output_root: pathlib.Path, *, overwrite: bool) -> None:
    """Create the output dataset by copying the source tree before modifying parquet files."""
    if source_root.resolve() == output_root.resolve():
        raise ValueError("source_root and output_root must be different to preserve the original dataset.")
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
) -> None:
    """Convert a local Franka RealSense LeRobot dataset into joint-target action form."""
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not (source_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"LeRobot metadata not found under {source_root}")

    _copy_dataset(source_root, output_root, overwrite=overwrite)

    # Convert every episode parquet and collect updated action statistics.
    action_stats_by_episode: dict[int, dict[str, list[float] | list[int]]] = {}
    episode_paths = sorted((output_root / "data").glob("chunk-*/episode_*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(f"No episode parquet files found under {output_root / 'data'}")
    for episode_path in episode_paths:
        episode_index = _episode_index_from_path(episode_path)
        action_stats_by_episode[episode_index] = _convert_episode(episode_path)

    # Preserve all existing metadata stats and replace only the action statistics.
    stats_path = output_root / "meta" / "episodes_stats.jsonl"
    stats_rows = _load_jsonl(stats_path)
    for row in stats_rows:
        episode_index = int(row["episode_index"])
        row["stats"]["actions"] = action_stats_by_episode[episode_index]
    _write_jsonl(stats_path, stats_rows)

    print(f"Converted {len(episode_paths)} episodes.")
    print(f"Source dataset: {source_root}")
    print(f"Output dataset: {output_root}")


if __name__ == "__main__":
    tyro.cli(main)
