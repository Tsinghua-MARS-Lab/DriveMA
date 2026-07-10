# DriveMA Three-Stage Training Pipeline

This document records the current three-stage training pipeline for DriveMA. All commands are expected to be executed from the repository root:

```bash
cd /cephfs/zhengwc/FluidDrive/ms-swift-3.5
conda activate mining
```

## Overview

| Stage | Objective | Training entrypoint | Training data | Checkpoint selection |
| --- | --- | --- | --- | --- |
| Stage 1 | Action-centric pretraining | `examples/train/full/5s/pretrained.sh` | `data/train_data/sft_data/action-centric-pretraining` | Highest `meta action` accuracy on validation |
| Stage 2 | Action-conditioned trajectory SFT | `examples/train/full/5s/posttrained_SFT.sh` | `data/train_data/sft_data/action-conditioned-traj/20pc_train_samples_mutil_turn_random_mask.jsonl` | Lowest `ADE` on validation |
| Stage 3 | GSPO / GRPO RL | `examples/train/grpo/internal/gspo.sh` | RL data configured in the script | Use the checkpoint around `620 steps` |

The three stages are sequentially dependent:

```text
base model
  -> Stage 1 best meta-action ckpt
  -> Stage 2 best ADE ckpt
  -> Stage 3 GSPO ckpt around step 620
```

## Stage 1: Action-Centric Pretraining

Training script:

```bash
bash /cephfs/zhengwc/FluidDrive/ms-swift-3.5/examples/train/full/5s/pretrained.sh
```

Training dataset:

```text
/cephfs/zhengwc/FluidDrive/ms-swift-3.5/data/train_data/sft_data/action-centric-pretraining
```

The current script runs full SFT on 8 GPUs with `save_steps=50`. Checkpoints are written to the run directory under `--output_dir outputs`. After training starts, launch the checkpoint monitor on a separate idle GPU; a 3090 is sufficient.

Monitor script:

```bash
CUDA_VISIBLE_DEVICES=<3090_GPU_ID> python /cephfs/zhengwc/FluidDrive/ms-swift-3.5/tools/monitor/eval_watch_meta_action_checkpoints.py \
  --checkpoint_root <stage1_run_dir> \
  --output_dir <stage1_eval_dir> \
  --input_jsonl data/train_data/meta_data/val_samples_434.jsonl \
  --label_jsonl data/val_data_w_action_label/val_samples_479_meta_action_label.jsonl \
  --best_metric joint_acc \
  --tensor_parallel_size 1
```

Outputs:

- `stage1_eval_dir/best_meta_action.json`: records the checkpoint with the best meta-action metric on validation.
- `stage1_eval_dir/history.jsonl`: records the historical metrics of evaluated checkpoints.
- By default, evaluated non-best checkpoints are deleted. Add `--no_delete_non_best` explicitly if all checkpoints need to be preserved.

After Stage 1 finishes, use `best_checkpoint_path` from `best_meta_action.json` as the initialization model for Stage 2.

## Stage 2: Action-Conditioned Trajectory SFT

Training script:

```bash
bash /cephfs/zhengwc/FluidDrive/ms-swift-3.5/examples/train/full/5s/posttrained_SFT.sh
```

This stage uses only the following training file:

```text
/cephfs/zhengwc/FluidDrive/ms-swift-3.5/data/train_data/sft_data/action-conditioned-traj/20pc_train_samples_mutil_turn_random_mask.jsonl
```

Before running, verify the following in `posttrained_SFT.sh`:

- `--model` points to the `best_checkpoint_path` selected from Stage 1.
- `--dataset` points to the single jsonl file above, not the whole directory.

After training starts, launch the ADE monitor on another idle GPU:

```bash
CUDA_VISIBLE_DEVICES=<3090_GPU_ID> python /cephfs/zhengwc/FluidDrive/ms-swift-3.5/tools/monitor/eval_watch_meta_action_ade_checkpoints.py \
  --checkpoint_root <stage2_run_dir> \
  --output_dir <stage2_eval_dir> \
  --input_jsonl data/train_data/meta_data/val_samples_434.jsonl \
  --label_jsonl data/val_data_w_action_label/val_samples_479_meta_action_label.jsonl \
  --best_metric ade_5s \
  --tensor_parallel_size 1
```

Outputs:

- `stage2_eval_dir/best_model.json`: records the checkpoint with the lowest ADE on validation. By default, `ade_5s` is used, and lower is better.
- `stage2_eval_dir/history.jsonl`: records historical meta-action and ADE metrics.
- By default, evaluated non-best checkpoints are deleted. Add `--no_delete_non_best` explicitly if all checkpoints need to be preserved.

After Stage 2 finishes, use `best_checkpoint_path` from `best_model.json` as the initialization model for Stage 3.

## Stage 3: GSPO / GRPO RL

Training script:

```bash
bash /cephfs/zhengwc/FluidDrive/ms-swift-3.5/examples/train/grpo/internal/gspo.sh
```

Before running, verify the following in `gspo.sh`:

```bash
--model <stage2_best_checkpoint_path>
```

Here, `<stage2_best_checkpoint_path>` comes from the Stage 2 `best_model.json`.

The current script uses 8 GPUs, vLLM colocate mode, and `save_steps=20`. It also enables two-turn dialogue GRPO credit:

```bash
--multi_turn_scheduler fixed_dialogue
--max_turns 2
--per_turn_grpo_credit true
--reward_funcs meta_action_acc rfs traj_consistency
--reward_weights 1.0 1.0 0.5
```

For the final model, use the checkpoint around `620 steps`, for example:

```text
<stage3_run_dir>/checkpoint-620
```

If the actual training directory does not contain `checkpoint-620`, choose the checkpoint closest to 620 steps based on the save interval.

## Key Notes

- The monitors for Stage 1 and Stage 2 must be launched separately after training starts, and they must point to the current run directory where checkpoints are being written.
- The monitor considers a checkpoint ready only when the checkpoint directory contains `config.json` and model weight files, which avoids reading checkpoints that are still being written.
- Stage 1 selects the best checkpoint only by meta-action accuracy; Stage 2 selects the best checkpoint only by ADE; Stage 3 uses the checkpoint around 620 steps based on empirical selection.
