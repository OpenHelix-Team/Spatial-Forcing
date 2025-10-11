
# Setup Instructions

```bash
# Create and activate conda environment
conda create -n spatialforcing python=3.10.16 -y
conda activate spatialforcing

# Install PyTorch
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# Clone spatial-forcing repo and pip install to download dependencies
git clone https://github.com/moojink/openvla-oft.git
cd Spatial-Forcing
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```
- If you are uncertain about the version of a dependency, please ref to our [**complete envs list**](../envs_list.txt).
<!-- TODO 修改git clone地址、cd的路径、pyproject里面的project.urls -->