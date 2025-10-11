
# LIBERO Dataset Preparation
Clone and install the [LIBERO repo](https://github.com/Lifelong-Robot-Learning/LIBERO) and required packages:
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt  # From openvla-oft base dir
```

(Optional, if you plan to launch training) To download the [LIBERO datasets](https://huggingface.co/datasets/openvla/modified_libero_rlds) that we used in our fine-tuning experiments, run the command below or download them manually. This will download the LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-10 datasets in RLDS data format (~10 GB total). You can use these to fine-tune Spatial-Forcing or train other methods like OpenVLA. Note that these are the same datasets used in the original OpenVLA project. If needed, see details on how to download the original non-RLDS datasets [here](https://github.com/openvla/openvla?tab=readme-ov-file#libero-setup).
```bash
git clone git@hf.co:datasets/openvla/modified_libero_rlds ./data/libero  # or download manually
```

# Directory structure
```
Spatial-Forcing
    ├── data
    ·   ├── libero
        │   ├── libero_10_no_noops
        │   │   └── 1.0.0  (It contains some json files and 32 tfrecord files)
        │   ├── libero_goal_no_noops
        │   │   └── 1.0.0  (It contains some json files and 16 tfrecord files)
        │   ├── libero_object_no_noops
        │   │   └── 1.0.0  (It contains some json files and 32 tfrecord files)
        │   ├── libero_spatial_no_noops
        │   │   └── 1.0.0  (It contains some json files and 16 tfrecord files)
        │
        └── other benchmarks ...
```
<!-- todo 树结构里Spatial-Forcing的名字 -->