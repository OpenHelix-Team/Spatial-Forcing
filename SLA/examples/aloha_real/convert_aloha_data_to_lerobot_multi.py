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

# --- 100% 保留原始辅助函数，不做任何修改 ---

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

# --- 修改后的核心逻辑，支持多文件夹 ---

def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    cameras: list[str],
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    # 完全保留原始的电机命名列表
    motors = ["right_waist", "right_shoulder", "right_elbow", "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate", "right_gripper", "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate", "left_gripper"]
    
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(motors),), "names": [motors]},
        "action": {"dtype": "float32", "shape": (len(motors),), "names": [motors]},
    }
    if has_velocity:
        features["observation.velocity"] = {"dtype": "float32", "shape": (len(motors),), "names": [motors]}
    if has_effort:
        features["observation.effort"] = {"dtype": "float32", "shape": (len(motors),), "names": [motors]}
    
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode, "shape": (3, 480, 640), "names": ["channels", "height", "width"]
        }

    # 仅在开始时删除一次
    dest_path = HF_LEROBOT_HOME / repo_id
    if dest_path.exists():
        shutil.rmtree(dest_path)

    return LeRobotDataset.create(
        repo_id=repo_id, fps=30, robot_type=robot_type, features=features,
        use_videos=dataset_config.use_videos, tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )

def port_aloha_multi(
    raw_dirs: list[Path], # 改为列表
    repo_id: str,
    *,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    # 1. 扫描所有文件夹并提取元数据
    all_data = []
    global_has_velocity = False
    global_has_effort = False
    common_cameras = None

    for r_dir in raw_dirs:
        if not r_dir.exists():
            continue
        hdf5_files = sorted(r_dir.glob("episode_*.hdf5"))
        if not hdf5_files:
            continue
        
        # 检查特征对齐情况
        current_cameras = get_cameras(hdf5_files)
        if common_cameras is None:
            common_cameras = current_cameras
        elif common_cameras != current_cameras:
            raise ValueError(f"Camera mismatch in {r_dir}. Expected {common_cameras}, got {current_cameras}")

        global_has_velocity = global_has_velocity or has_velocity(hdf5_files)
        global_has_effort = global_has_effort or has_effort(hdf5_files)
        
        # 将文件夹名称作为该部分数据的 task 标签
        all_data.append({"task": r_dir.name, "files": hdf5_files})

    if not all_data:
        raise ValueError("No valid data found in provided directories.")

    # 2. 创建统一的数据集结构
    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        cameras=common_cameras,
        mode=mode,
        has_velocity=global_has_velocity,
        has_effort=global_has_effort,
        dataset_config=dataset_config,
    )

    # 3. 顺序填充数据
    for task_item in all_data:
        task_label = task_item["task"]
        files = task_item["files"]
        print(f"Processing task: {task_label}")

        for ep_path in tqdm.tqdm(files):
            imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path, common_cameras)
            num_frames = state.shape[0]

            for i in range(num_frames):
                frame = {
                    "observation.state": state[i],
                    "action": action[i],
                    "task": task_label,
                }
                for camera, img_array in imgs_per_cam.items():
                    frame[f"observation.images.{camera}"] = img_array[i]
                
                # 只有在数据集启用了这些字段时才添加
                if global_has_velocity:
                    frame["observation.velocity"] = velocity[i] if velocity is not None else torch.zeros_like(state[i])
                if global_has_effort:
                    frame["observation.effort"] = effort[i] if effort is not None else torch.zeros_like(state[i])

                dataset.add_frame(frame)
            dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub()

if __name__ == "__main__":
    tyro.cli(port_aloha_multi)