# DriveMA: Driving Vision-Language-Action Models with Verifiable Meta-Actions

<p align="center">
  <a href="https://www.corl.org/"><img src="https://img.shields.io/badge/CoRL_2026-Accepted-2E7DD1" alt="Accepted at CoRL 2026"></a>
  <a href="https://arxiv.org/pdf/2605.31271"><img src="https://img.shields.io/badge/Paper-arXiv-B31B1B?logo=arxiv&logoColor=white" alt="Paper"></a>
  <a href="https://tsinghua-mars-lab.github.io/DriveMA"><img src="https://img.shields.io/badge/Project%20Page-DriveMA-0A7E8C?logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href="https://huggingface.co/zwc2003/DriveMA-2B"><img src="https://img.shields.io/badge/Model-DriveMA--2B-FFD21E?logo=huggingface&logoColor=black" alt="DriveMA-2B Model"></a>
</p>

DriveMA is a vision-language-action (VLA) framework for end-to-end driving planning. It introduces **verifiable meta-actions**—compact, interpretable decisions that bridge visual observations and future trajectories. By deriving the same interface from expert and predicted motion, DriveMA can supervise high-level decisions before reinforcement learning (RL) and explicitly reward consistency between a predicted meta-action and its resulting trajectory.

This repository is developed from [ms-swift](https://github.com/modelscope/ms-swift). DriveMA adds the driving datasets, three-stage training pipeline, trajectory-aware rewards, and turn-level RL credit assignment used in our work.

## Updates

- **2026-09-05:** 🎉 DriveMA is accepted to **CoRL 2026**!
- **2026-07-18:** Released the [DriveMA-2B model weights](https://huggingface.co/zwc2003/DriveMA-2B).
- **2026-07-10:** Initial release of the DriveMA codebase, training annotations, and three-stage training instructions.
- Dataset image files are **not** redistributed; follow [Data Preparation](#data-preparation) to download the annotations and original datasets, then update local image paths.

## Release Status

- [x] Training code
- [x] Testing code
- [x] Training and validation annotations
- [x] [DriveMA-2B model weights](https://huggingface.co/zwc2003/DriveMA-2B)

## Contents

- [Release Status](#release-status)
- [Method](#method)
- [Results](#results)
- [Installation](#installation)
- [Model Weights](#model-weights)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Repository Structure](#repository-structure)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Citation](#citation)

## Method

DriveMA formulates planning as a two-turn generation process:

1. **Predict a meta-action.** Given multi-view driving observations and vehicle state, the model predicts a concise longitudinal and lateral action.
2. **Generate future waypoints.** The model predicts a future trajectory conditioned on this meta-action.

The training pipeline aligns these two turns in three stages:

1. **Action-centric pretraining (ACP):** learns expert meta-actions and driving-VQA knowledge, including intention, action, and risk reasoning.
2. **Action-conditioned trajectory SFT:** learns to generate future waypoints conditioned on the predicted decision.
3. **Turn-level RL:** assigns meta-action correctness to the first turn and trajectory quality plus trajectory–action consistency to the second turn. This avoids applying one sequence-level reward uniformly to both decisions and trajectories.

## Results

### Waymo Open Dataset: vision-based end-to-end planning

DriveMA establishes state-of-the-art results on the Waymo Open Dataset vision-based end-to-end benchmark. DriveMA-4B achieves the best **RFS Overall**, while DriveMA-2B achieves the best **RFS Spotlight**, **ADE@5s**, and **ADE@3s**.

| Method | RFS Overall ↑ | RFS Spotlight ↑ | ADE@5s ↓ | ADE@3s ↓ |
| --- | ---: | ---: | ---: | ---: |
| RAP | 8.043 | 7.204 | 2.646 | 1.174 |
| DriveMA-2B | 8.060 | **7.251** | **2.616** | **1.154** |
| DriveMA-4B | **8.079** | 7.169 | 2.670 | 1.166 |

For the complete comparison, ablations, qualitative cases, and videos, visit the [project page](https://tsinghua-mars-lab.github.io/DriveMA).

## Installation

```bash
git clone <YOUR_DRIVEMA_REPOSITORY_URL>
cd DriveMA/code

conda create -n drivema python=3.11 -y
conda activate drivema
pip install -e .
```

The training pipeline uses CUDA, PyTorch, DeepSpeed, and vLLM. Install versions compatible with your CUDA driver and hardware before launching training.

## Model Weights

The trained DriveMA-2B model is available on [Hugging Face](https://huggingface.co/zwc2003/DriveMA-2B):

```bash
hf download zwc2003/DriveMA-2B --local-dir <DRIVEMA_2B_PATH>
```

### Base Model Weights

DriveMA does not redistribute the base model weights. To train DriveMA, download the official [Qwen3.5-2B model weights](https://huggingface.co/Qwen/Qwen3.5-2B) to a local directory:

```bash
hf download Qwen/Qwen3.5-2B --local-dir <QWEN3_5_2B_PATH>
```

Set `--model` in the Stage 1 training script to `<QWEN3_5_2B_PATH>`. The following stages must be initialized from the checkpoint selected by the previous stage.

## Data Preparation

DriveMA JSONL annotations are hosted in the [DriveMA Datasets](https://huggingface.co/datasets/zwc2003/DriveMA_Datasets) repository. Image and video assets are not included. Download the annotations and extract them from the `code/` directory:

```bash
hf download zwc2003/DriveMA_Datasets DriveMA_Datasets.zip --repo-type dataset --local-dir .
unzip DriveMA_Datasets.zip
```

The archive extracts a `data/` directory containing all training and validation annotations. Download each source dataset under its original license and update the absolute paths in every JSONL record's `images` field so that they point to files on your machine.

> **Important:** The released annotations contain paths from the authors' training environment (for example, `/nfs_zhaohang/...`). They are not portable. Training or evaluation will fail until every referenced image path is replaced with your local path.

### 1. Waymo Open Dataset

Download the Waymo Open Dataset following the official [download instructions](https://waymo.com/open/download/) and [repository](https://github.com/waymo-research/waymo-open-dataset). Extract the camera images into the following directory structure:

```text
<WAYMO_IMAGE_ROOT>/
└── images/
    ├── train/
    │   └── <sceneID>/
    │       └── <sceneID>-<frameID>_<camera>.jpg
    ├── val/
    │   └── <sceneID>/
    │       └── <sceneID>-<frameID>_<camera>.jpg
    └── test/
        └── <sceneID>/
            └── <sceneID>-<frameID>_<camera>.jpg
```

For example:

```text
images/train/1c1442ae77af9a1727f0f912ae7648c3/
├── 1c1442ae77af9a1727f0f912ae7648c3-206_front_left.jpg
├── 1c1442ae77af9a1727f0f912ae7648c3-206_front.jpg
└── 1c1442ae77af9a1727f0f912ae7648c3-206_front_right.jpg
```

Update the Waymo paths referenced by the action-conditioned trajectory, RL, and validation JSONL files to `<WAYMO_IMAGE_ROOT>/images/...`.

### 2. AD-VQA

The AD-VQA data used for action-centric pretraining combines the following two public datasets. Download their image assets from the official repositories and preserve a directory layout that matches the paths you write into the annotations.

- **IDKB:** [4DVLab/IDKB](https://github.com/4DVLab/IDKB)
- **LingoQA:** [wayveai/LingoQA](https://github.com/wayveai/LingoQA)

Then update the `images` entries in:

```text
data/train_data/sft_data/action-centric-pretraining/AD_VQA_v1_clean_nfs_longitudinal.jsonl
```

### 3. WaymoQA

Download WaymoQA from its official repository:

- **WaymoQA:** [sjyu001/WaymoQA](https://github.com/sjyu001/WaymoQA)

After downloading its image and video assets, update the image paths in:

```text
data/train_data/sft_data/action-centric-pretraining/WaymoQA_train_image.jsonl
data/train_data/sft_data/action-centric-pretraining/WaymoQA_train_video.jsonl
```

### Archive contents

```text
data/
├── train_data/
│   ├── sft_data/
│   │   ├── action-centric-pretraining/
│   │   │   ├── AD_VQA_v1_clean_nfs_longitudinal.jsonl
│   │   │   ├── WaymoQA_train_image.jsonl
│   │   │   └── WaymoQA_train_video.jsonl
│   │   └── action-conditioned-traj/
│   │       └── 20pc_train_samples_mutil_turn_random_mask.jsonl
│   ├── rfs_rl_data/
│   │   └── total_434_RFS_RL.jsonl
│   └── meta_data/
│       ├── train_samples_filtered_20pc_downsample.jsonl
│       ├── val_samples_434.jsonl
│       ├── val_samples_479.jsonl
│       └── test_samples_last_frame.jsonl
└── val_data_w_action_label/
    └── val_samples_479_meta_action_label.jsonl
```

## Training

DriveMA is trained sequentially in three stages:

| Stage | Objective | Script | Primary annotation |
| --- | --- | --- | --- |
| 1 | Action-centric pretraining | `examples/train/full/5s/pretrained.sh` | `data/train_data/sft_data/action-centric-pretraining/` |
| 2 | Action-conditioned trajectory SFT | `examples/train/full/5s/posttrained_SFT.sh` | `data/train_data/sft_data/action-conditioned-traj/20pc_train_samples_mutil_turn_random_mask.jsonl` |
| 3 | Turn-level GSPO/GRPO RL | `examples/train/grpo/internal/gspo.sh` | `data/train_data/rfs_rl_data/total_434_RFS_RL.jsonl` |

Set the model, dataset, output, reward-plugin, and checkpoint paths in each script to valid local locations before use. Run the stages in order: select the best Stage 1 checkpoint by meta-action accuracy, use it to initialize Stage 2, then initialize Stage 3 from the best Stage 2 checkpoint.

See [README_DriveMA_Pipeline.md](README_DriveMA_Pipeline.md) for the full training and checkpoint-selection procedure.

## Repository Structure

```text
code/
├── data/                         # DriveMA JSONL annotations
├── examples/train/full/5s/       # Stage 1 and Stage 2 training scripts
├── examples/train/grpo/          # Stage 3 RL scripts and reward plugin
├── swift/                        # Training framework and DriveMA RL extensions
├── tools/                        # Inference or evaluation tools
└── README_DriveMA_Pipeline.md    # Detailed three-stage training guide
```

## Acknowledgments

DriveMA is built upon [ms-swift](https://github.com/modelscope/ms-swift). We thank the authors and maintainers of ms-swift, the Waymo Open Dataset, IDKB, LingoQA, and WaymoQA for making their resources available to the research community.

## License

This codebase is released under the [Apache License 2.0](LICENSE). The datasets referenced above are governed by their respective licenses and terms of use.

## Citation

If you find DriveMA useful, please cite our paper:

```bibtex
@inproceedings{zheng2026drivema,
  title={DriveMA: Driving Vision-Language-Action Models with verifiable Meta-Actions},
  author={Zheng, Weicheng and Huang, Yixin and Sun, Qiao and Li, Derun and Zhao, Hang},
  booktitle={Proceedings of the Conference on Robot Learning (CoRL)},
  year={2026}
}
```
