# Spatial Forcing: Implicit Spatial Representation Alignment for Vision-language-action Model
<div align="center">

[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](http://arxiv.org/abs/2510.12276) [![Page](https://img.shields.io/badge/Project--Page-blue?style=for-the-badge&logo=homepage&logoColor=white)](https://spatial-forcing.github.io/) [![Hugging Face Collection](https://img.shields.io/badge/Models-fcd022?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/collections/haofuly/spatial-forcing-68ea1bf0f1ac2c60e2ec6caa) [![WeChat](https://img.shields.io/badge/WeChat--Group-07C160?style=for-the-badge&logo=wechat&logoColor=white)]([https://github.com/OpenHelix-Team/VLA-Adapter/issues/1](https://github.com/OpenHelix-Team/Spatial-Forcing/issues/1))

</div>

## :loudspeaker: News!
- **[2025/10/28]** We released our codes based on Pi_0 in real world! Everyone is welcome to use it!🎉
- **[2025/10/24]** Congradulations to Jialong! He and our SF got the second place in Agibot World Challenge as well as 5000$ prize💰!
- **[2025/10/18]** Our paper won the 🥇**first place** in the [daily list](https://huggingface.co/papers/2510.12276) and 🥉**third place** in the [weekly list](https://huggingface.co/papers/week/2025-W42) in HF! ⭐
- **[2025/10/12]** We released our paper on [ArXiv](http://arxiv.org/abs/2510.12276).

## 🌟 Key Features of Spatial-Forcing (SF)

1. **Universality**: SF is a **plug-and-play** 3D finetune strategy that can be seamlessly integrated with any VLA training process, requiring only **30 lines** of code modifications. It substantially enhances spatial reasoning and manipulation capabilities. We provide implementations based on **OpenVLA** and **Pi0**, along with a **quick-start guide** for adapting SF to other VLA models.

2. **Strong Performance**: SF achieves **state-of-the-art (SOTA)** results on both **LIBERO** and **RoboTwin** benchmarks.  
In real-world experiments involving complex spatial structures, SF improves task success rates by **up to 50%**.

3. **Efficient Training**: SF requires only **3% of the training steps** or **5% of the training data** to reach a 66% success rate on LIBERO-Long. Moreover, it achieves strong real-world performance with as few as **20 demonstrations**.


## 📃 Overview
![teaser](./figs/teaser.png)

Our Spatial-Forcing (SF) model aligns the intermediate visual embeddings of VLAs with geometric representations produced by pretrained 3D foundation models. This alignment demonstrates enhanced performance, training efficiency, and data effificency.


<!-- ## Table of Contents
- [🚀 Get Started](#-get-started)
  - [Environment Setup](#environment-setup)
  - [Data Preparation](#data-preparation)
  - [Inference](#inference)
  - [Training](#training) -->


## 🚀 Get Started
- To reproduce our simulation results, ref to our [openvla-SF folder](openvla-SF).

- To deploy policy in real-world robots, ref to our [openpi-SF folder](openpi-SF).

- To integrate Spatial-Forcing strategy into your VLA model, ref to [Simulation Training Scripts Line373-Line400](./openvla-SF/vla-scripts/finetune_align.py#L373-L399).


## 🔥 TODO List
✅ Training and inference code on LIBERO (Base model: OpenVLA)<br>
✅ Checkpoints on LIBERO (Base model: OpenVLA)<br>
✅ Deployment code in real world (Base model: Pi_0 torch version)<br>


## 🌏 Contact
For further discussion and collaboration, please feel free to contact us via Email and WeChat:

| Author | Email | WeChat |
|:---:|:---:|:---:|
| Fuhao Li | lfh23@mails.tsinghua.edu.cn | haofuly |
| Wenxuan Song | songwenxuan0115@gmail.com | swx0757 |
> WeChat Communication Group is at [here](https://github.com/OpenHelix-Team/Spatial-Forcing/issues/1)


## ❤️ Acknowledgement
We thank these great works and open-source codebases: [OpenVLA-OFT](https://github.com/moojink/openvla-oft) & [OpenPI](https://github.com/Physical-Intelligence/openpi) & [VGGT](https://github.com/facebookresearch/vggt) & [REPA](https://github.com/sihyun-yu/REPA)


## 🖊 Citation
If you find this work useful, please cite:

```bibtex
@article{spatialforcing2025,
  author    = {Li Fuhao, Song Wenxuan, Zhao Han, Wang Jingbo, Ding Pengxiang, Wang Donglin, Zeng Long, Li Haoang},
  title     = {Spatial Forcing: Implicit Spatial Representation Alignment For Vision-Language-Action Model},
  journal   = {arXiv preprint arXiv:2510.12276},
  year      = {2025},
}
```
