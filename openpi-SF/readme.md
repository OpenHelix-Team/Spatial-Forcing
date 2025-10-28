# Run Spatial-Forcing on Real World Robots
> This example demonstrates how to run on the **real-world** environments.<br/>
> We choose **Pi_0 (torch version)** as base model for this deployment.

⭐ Please select this dir here `Spatial-Forcing/openpi-SF/` as **root dir** for the following steps.
```bash
cd ./openpi-SF  # If your terminal dir is still at Spatial-Forcing/
```
As the [original openpi repo](https://github.com/Physical-Intelligence/openpi/) has provided very detailed instructions, here we only provide key steps for your quick deployment.


## 1. Environment Setup
We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
source .venv/bin/activate
```

NOTE: `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot as a dependency.


## 2. Data Preparation
First, you need to collect the task-specific raw data with your own robot, and save it in the `.hdf5` format.

Then, convert the data to LeRobot dataset format.
```bash
uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
# By default, The converted data is stored in ~/.cache/huggingface/lerobot/<org>/<dataset-name>/
```


## 3. Training
First, define your task-specific config in [config.py](src/openpi/training/config.py). And we provide an example of our real-world task [here](src/openpi/training/config.py#L772-L809).

Then, convert a JAX model checkpoint to PyTorch format:
```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir gs://openpi-assets/checkpoints/pi0_base \
    --config_name <config_name> \
    --output_path ./checkpoints/pi0_base_full_torch
# This command will automatically download pi0_base checkpoint to ~/.cache/openpi/openpi-assets/checkpoints/pi0_base/
# Otherwise you can download it manually and modify the --checkpoint_dir
```

Then, download the [VGGT](https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt) foundation models and place it in the `./checkpoints/vggt/` folder. 

The directory structure is as below:
```
openpi-SF
    ├── checkpoints
    ·   ├── pi0_base_full_torch
        │   ├── config.json
        │   ├── model.safetensors
        │   └── ...
        ├── vggt
        ·   └── model.pt
```

Next, you need to compute the normalization statistics for the training data.
```bash
uv run scripts/compute_norm_stats.py --config-name <config_name>
```

Finally, launch training using one of these modes:
```bash
# Single GPU training:
uv run scripts/train_align_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
# Example:
uv run scripts/train_align_pytorch.py debug --exp_name pytorch_test
uv run scripts/train_align_pytorch.py debug --exp_name pytorch_test --resume  # Resume from latest checkpoint
uv run scripts/train_align_pytorch.py debug --exp_name pytorch_test --overwrite  # Overwrite existing checkpoints

# Multi-GPU training (single node):
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_align_pytorch.py <config_name> --exp_name <run_name>
# Example:
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_align_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test

# Multi-Node Training:
uv run torchrun \
    --nnodes=<num_nodes> \
    --nproc_per_node=<gpus_per_node> \
    --node_rank=<rank_of_node> \
    --master_addr=<master_ip> \
    --master_port=<port> \
    scripts/train_align_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>
```


## 4. Inference
Real-world inference is executed in the server-client form.

First, launch a model server (we use the checkpoint for iteration 20,000 for this example, modify as needed):
```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config_name> --policy.dir=checkpoints/<config_name>/<run_name>/20000
```

This will spin up a server that listens on port 8000 and waits for observations to be sent to it.

Then, We can then run an client robot script that queries the server.

You need to write your client script according to your robot. A simple [client exmaple](examples/simple_client/main.py) is as below:
```bash
uv run examples/simple_client/main.py --env ALOHA
```