"""
Script to convert multiple Aloha hdf5 data directories to a single LeRobot dataset v2.0 format.

Example usage:
uv run convert_aloha.py --raw-dirs /path/to/task1 /path/to/task2 --repo-id <org>/<dataset-name> --fps 30 --prompts "instruction 1" "instruction 2"
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    cameras: list[str],
    mode: Literal["video", "image"] = "video",
    fps: int = 50,
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        # 新增 language_instruction 字段用于保存 prompt
        "language_instruction": {
            "dtype": "string",
            "shape": (1,),
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4
        if uncompressed:
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                imgs_array.append(cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BGR2RGB))
            imgs_array = np.array(imgs_array)
        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
    cameras: list[str],
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:])
        action = torch.from_numpy(ep["/action"][:])
        velocity = torch.from_numpy(ep["/observations/qvel"][:]) if "/observations/qvel" in ep else None
        effort = torch.from_numpy(ep["/observations/effort"][:]) if "/observations/effort" in ep else None
        imgs_per_cam = load_raw_images_per_camera(ep, cameras)
    return imgs_per_cam, state, action, velocity, effort


def get_prompt(ep_path: Path, default_prompt: str) -> str:
    """尝试从 HDF5 中读取 prompt"""
    with h5py.File(ep_path, "r") as f:
        if "language_instruction" in f.attrs:
            return str(f.attrs["language_instruction"])
        if "language_instruction" in f:
            data = f["language_instruction"][()]
            return data.decode("utf-8") if isinstance(data, bytes) else str(data)
    return default_prompt


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    cameras: list[str],
    task: str,
    fixed_prompt: str | None = None,
) -> LeRobotDataset:
    for ep_idx in tqdm.tqdm(range(len(hdf5_files))):
        ep_path = hdf5_files[ep_idx]
        
        # 确定当前 Episode 的 Prompt
        prompt = fixed_prompt if fixed_prompt else get_prompt(ep_path, task)

        imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path, cameras)
        num_frames = state.shape[0]

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "task": task,
                "language_instruction": prompt,
            }

            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            # 如果全局启用了这些字段但当前 episode 缺失，则补零
            if "observation.velocity" in dataset.features:
                frame["observation.velocity"] = velocity[i] if velocity is not None else torch.zeros_like(state[i])
            if "observation.effort" in dataset.features:
                frame["observation.effort"] = effort[i] if effort is not None else torch.zeros_like(state[i])

            dataset.add_frame(frame)

        dataset.save_episode()

    return dataset


def port_aloha_multi(
    raw_dirs: list[Path],
    repo_id: str,
    fps: int = 50,
    prompts: list[str] | None = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if prompts and len(prompts) != len(raw_dirs):
        raise ValueError(f"Prompts 数量 ({len(prompts)}) 必须与 raw_dirs 数量 ({len(raw_dirs)}) 一致")

    # 1. 扫描所有目录，确定全局配置
    all_tasks_data = []
    global_has_velocity = False
    global_has_effort = False
    common_cameras = None

    for i, r_dir in enumerate(raw_dirs):
        if not r_dir.exists():
            raise ValueError(f"目录 {r_dir} 不存在")
        
        hdf5_files = sorted(r_dir.glob("episode_*.hdf5"))
        if not hdf5_files:
            print(f"警告: 目录 {r_dir} 中没有找到 episode_*.hdf5 文件")
            continue

        # 检查相机一致性
        current_cameras = get_cameras(hdf5_files)
        if common_cameras is None:
            common_cameras = current_cameras
        elif common_cameras != current_cameras:
            raise ValueError(f"相机不匹配! 期望 {common_cameras}, 但在 {r_dir} 发现 {current_cameras}")

        global_has_velocity = global_has_velocity or has_velocity(hdf5_files)
        global_has_effort = global_has_effort or has_effort(hdf5_files)
        
        all_tasks_data.append({
            "task": r_dir.name,
            "files": hdf5_files,
            "fixed_prompt": prompts[i] if prompts else None
        })

    if not all_tasks_data:
        raise ValueError("没有任何有效的数据可供转换")

    # 2. 创建统一的数据集
    print(f"正在创建数据集: {repo_id} (FPS: {fps}, Cameras: {common_cameras})")
    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        cameras=common_cameras,
        mode=mode,
        fps=fps,
        has_effort=global_has_effort,
        has_velocity=global_has_velocity,
        dataset_config=dataset_config,
    )

    # 3. 循环填充数据
    for task_info in all_tasks_data:
        print(f"正在处理任务: {task_info['task']}")
        dataset = populate_dataset(
            dataset,
            task_info["files"],
            cameras=common_cameras,
            task=task_info["task"],
            fixed_prompt=task_info["fixed_prompt"]
        )

    if push_to_hub:
        dataset.push_to_hub()
    
    print("所有数据集转换完成！")


if __name__ == "__main__":
    tyro.cli(port_aloha_multi)